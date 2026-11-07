# syntax=docker/dockerfile:1

# ==============================================================================
# 1. BASE CHEF STAGE (Shared compilation toolchain for AMD64 & ARM64)
# ==============================================================================
FROM lukemathwalker/cargo-chef:latest-rust-1-slim-trixie AS chef
WORKDIR /app

ARG TARGETARCH
ARG ORT_VERSION=1.31.1

# Install standard C-toolchain dependencies needed for compiling native crates
# and download official prebuilt ONNX Runtime shared libraries for dynamic loading.
RUN apt-get update && apt-get install -y --no-install-recommends \
    pkg-config \
    libssl-dev \
    ca-certificates \
    build-essential \
    cmake \
    curl \
    tar \
    binutils \
    && rm -rf /var/lib/apt/lists/* \
    && mkdir -p /opt/ort/lib /opt/ort-cuda/lib \
    && if [ "$TARGETARCH" = "arm64" ]; then \
         curl -fSL "https://github.com/microsoft/onnxruntime/releases/download/v${ORT_VERSION}/onnxruntime-linux-aarch64-${ORT_VERSION}.tgz" \
           | tar -xz -C /opt/ort --strip-components=1; \
         cp -a /opt/ort/lib/* /opt/ort-cuda/lib/; \
       else \
         curl -fSL "https://github.com/microsoft/onnxruntime/releases/download/v${ORT_VERSION}/onnxruntime-linux-x64-${ORT_VERSION}.tgz" \
           | tar -xz -C /opt/ort --strip-components=1; \
         curl -fSL "https://github.com/microsoft/onnxruntime/releases/download/v${ORT_VERSION}/onnxruntime-linux-x64-gpu_cuda13-${ORT_VERSION}.tgz" \
           | tar -xz -C /opt/ort-cuda --strip-components=1; \
       fi

# ==============================================================================
# 2. PLANNER STAGE (Shared dependency recipe extraction)
# ==============================================================================
FROM chef AS planner
COPY . .
RUN cargo chef prepare --recipe-path recipe.json

# ==============================================================================
# 3. UNIFIED BUILDER STAGE (Compiled once per architecture, shared by slim/cpu/cuda)
# ==============================================================================
FROM chef AS builder
WORKDIR /app
COPY --from=planner /app/recipe.json recipe.json
RUN --mount=type=cache,target=/usr/local/cargo/registry \
    --mount=type=cache,target=/usr/local/cargo/git \
    cargo chef cook --release \
    --features "default,cuda,tensorrt,streams-all" \
    --bin snooze \
    --recipe-path recipe.json

COPY . .
RUN --mount=type=cache,target=/usr/local/cargo/registry \
    --mount=type=cache,target=/usr/local/cargo/git \
    cargo build --release \
    --features "default,cuda,tensorrt,streams-all" \
    --bin snooze && \
    if readelf -d /app/target/release/snooze | grep -E "NEEDED.*(libcuda|libcudart|libcublas|libcurand|libnvrtc|libonnxruntime|libstdc\+\+)"; then \
      echo "ERROR: snooze binary has forbidden hard DT_NEEDED link dependency!"; \
      exit 1; \
    fi && \
    mkdir -p /app/dist/bin /app/dist/lib /app/dist/lib-cuda && \
    cp /app/target/release/snooze /app/dist/bin/snooze && \
    cp -a /opt/ort/lib/libonnxruntime* /app/dist/lib/ && \
    cp -a /opt/ort-cuda/lib/libonnxruntime* /app/dist/lib-cuda/

# Backwards-compatible stage aliases
FROM builder AS builder-slim
FROM builder AS builder-cpu

# ==============================================================================
# 4. SHARED BASE RUNTIME (Common OS packages, user, volumes & healthcheck)
# ==============================================================================
FROM debian:trixie-slim AS base-runtime

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    curl \
    libssl3t64 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Security: Create non-privileged system user and group (UID 10001)
RUN useradd -u 10001 -m -U -s /sbin/nologin appuser && \
    mkdir -p /models /app /home/appuser/.cache && \
    chown -R appuser:appuser /models /app /home/appuser

ENV SNOOZE_MODELS_DIR="/models" \
    RUST_LOG="info" \
    HF_HOME="/home/appuser/.cache/huggingface" \
    ORT_DYLIB_PATH="/usr/local/lib/libonnxruntime.so" \
    LD_LIBRARY_PATH="/usr/local/lib:/app"

WORKDIR /app

EXPOSE 8080 50051

VOLUME ["/models"]

HEALTHCHECK --interval=10s --timeout=3s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8080/health/live || exit 1

ENTRYPOINT ["snooze"]
CMD ["serve", "--host", "0.0.0.0", "--port", "8080", "--models-dir", "/models"]

# ==============================================================================
# 4A. TARGET: SLIM (Smallest footprint, zero Python)
# ==============================================================================
FROM base-runtime AS slim

COPY --from=builder /app/dist/bin/snooze /usr/local/bin/snooze
COPY --from=builder /app/dist/lib/ /usr/local/lib/
RUN ldconfig

USER appuser

# ==============================================================================
# 4B. SHARED PYTHON LAYER (Built once, shared by CPU and CUDA targets)
# ==============================================================================
FROM base-runtime AS base-python

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-numpy \
    python3-requests \
    && rm -rf /var/lib/apt/lists/*

# Bundle the pure-Python snoozeml package from source (zero pip/venv bloat,
# always version-locked to the daemon commit)
COPY python/snoozeml /opt/snoozeml/snoozeml

ENV SNOOZE_PYTHON="/usr/bin/python3" \
    PYTHONPATH="/opt/snoozeml" \
    PYTHONUNBUFFERED="1" \
    PIP_BREAK_SYSTEM_PACKAGES="1"

# ==============================================================================
# 4C. TARGET: CPU (Standard CPU inference with Python worker runtime)
# ==============================================================================
FROM base-python AS cpu

COPY --from=builder /app/dist/bin/snooze /usr/local/bin/snooze
COPY --from=builder /app/dist/lib/ /usr/local/lib/
RUN ldconfig

USER appuser

# ==============================================================================
# 4D. TARGET: CUDA (Default — Full features, GPU acceleration & Python worker)
# ==============================================================================
FROM nvidia/cuda:13.0.0-cudnn-runtime-ubuntu24.04 AS cuda

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    curl \
    libssl3t64 \
    libgomp1 \
    python3 \
    python3-numpy \
    python3-requests \
    && rm -rf /var/lib/apt/lists/*

# Security: Create non-privileged system user and group (UID 10001)
RUN useradd -u 10001 -m -U -s /sbin/nologin appuser && \
    mkdir -p /models /app /home/appuser/.cache && \
    chown -R appuser:appuser /models /app /home/appuser

# Bundle the pure-Python snoozeml package from source
COPY python/snoozeml /opt/snoozeml/snoozeml

ENV SNOOZE_MODELS_DIR="/models" \
    RUST_LOG="info" \
    HF_HOME="/home/appuser/.cache/huggingface" \
    SNOOZE_PYTHON="/usr/bin/python3" \
    PYTHONPATH="/opt/snoozeml" \
    PYTHONUNBUFFERED="1" \
    PIP_BREAK_SYSTEM_PACKAGES="1" \
    NVIDIA_VISIBLE_DEVICES="all" \
    NVIDIA_DRIVER_CAPABILITIES="compute,utility" \
    ORT_DYLIB_PATH="/usr/local/lib/libonnxruntime.so" \
    LD_LIBRARY_PATH="/usr/local/lib:/usr/local/cuda/lib64:/app"

WORKDIR /app

EXPOSE 8080 50051

VOLUME ["/models"]

HEALTHCHECK --interval=10s --timeout=3s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8080/health/live || exit 1

COPY --from=builder /app/dist/bin/snooze /usr/local/bin/snooze
COPY --from=builder /app/dist/lib-cuda/ /usr/local/lib/
RUN ldconfig

USER appuser

ENTRYPOINT ["snooze"]
CMD ["serve", "--host", "0.0.0.0", "--port", "8080", "--models-dir", "/models"]
