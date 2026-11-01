# syntax=docker/dockerfile:1

# ==============================================================================
# 1. BASE CHEF STAGE (Shared compilation toolchain for AMD64 & ARM64)
# ==============================================================================
FROM lukemathwalker/cargo-chef:latest-rust-1-slim-trixie AS chef
WORKDIR /app

# Install standard C-toolchain dependencies needed for compiling native crates
# (e.g. openssl, cmake for aws-lc-sys, C/C++ build scripts for tokenizers and ONNX Runtime)
RUN apt-get update && apt-get install -y --no-install-recommends \
    pkg-config \
    libssl-dev \
    ca-certificates \
    build-essential \
    cmake \
    && rm -rf /var/lib/apt/lists/*

# ==============================================================================
# 2. PLANNER STAGE (Shared dependency recipe extraction)
# ==============================================================================
FROM chef AS planner
COPY . .
RUN cargo chef prepare --recipe-path recipe.json

# ==============================================================================
# 3A. BUILDER — SLIM (Zero Python, native Rust & ONNX engines only)
# ==============================================================================
FROM chef AS builder-slim
WORKDIR /app
COPY --from=planner /app/recipe.json recipe.json
RUN --mount=type=cache,target=/usr/local/cargo/registry \
    --mount=type=cache,target=/usr/local/cargo/git \
    cargo chef cook --release \
    --no-default-features \
    --features "ort-backend,candle-backend,tree-backend,streams" \
    --bin snooze \
    --recipe-path recipe.json

COPY . .
RUN --mount=type=cache,target=/usr/local/cargo/registry \
    --mount=type=cache,target=/usr/local/cargo/git \
    cargo build --release \
    --no-default-features \
    --features "ort-backend,candle-backend,tree-backend,streams" \
    --bin snooze && \
    mkdir -p /app/dist/bin /app/dist/lib && \
    touch /app/dist/lib/.keep && \
    cp /app/target/release/snooze /app/dist/bin/snooze && \
    find /app/target/release -maxdepth 1 -name "libonnxruntime*.so*" -exec cp -L -t /app/dist/lib/ {} +

# ==============================================================================
# 3B. BUILDER — CPU (All backends + CPU Python worker + all streams)
# ==============================================================================
FROM chef AS builder-cpu
WORKDIR /app
COPY --from=planner /app/recipe.json recipe.json
RUN --mount=type=cache,target=/usr/local/cargo/registry \
    --mount=type=cache,target=/usr/local/cargo/git \
    cargo chef cook --release \
    --features "default,streams-all" \
    --bin snooze \
    --recipe-path recipe.json

COPY . .
RUN --mount=type=cache,target=/usr/local/cargo/registry \
    --mount=type=cache,target=/usr/local/cargo/git \
    cargo build --release \
    --features "default,streams-all" \
    --bin snooze && \
    mkdir -p /app/dist/bin /app/dist/lib && \
    touch /app/dist/lib/.keep && \
    cp /app/target/release/snooze /app/dist/bin/snooze && \
    find /app/target/release -maxdepth 1 -name "libonnxruntime*.so*" -exec cp -L -t /app/dist/lib/ {} +

# ==============================================================================
# 3C. BUILDER — CUDA (All backends + CUDA/TensorRT GPU + Candle CUDA + all streams)
# ==============================================================================
FROM nvidia/cuda:13.0.0-cudnn-devel-ubuntu24.04 AS builder-cuda
WORKDIR /app

ENV DEBIAN_FRONTEND=noninteractive \
    CUDA_HOME="/usr/local/cuda" \
    CUDA_PATH="/usr/local/cuda" \
    CUDA_COMPUTE_CAP="75" \
    PATH="/usr/local/cuda/bin:${PATH}"

# Install Cargo / Rust toolchain and compilation prerequisites
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    ca-certificates \
    pkg-config \
    libssl-dev \
    build-essential \
    cmake \
    && curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --default-toolchain stable --profile minimal \
    && rm -rf /var/lib/apt/lists/*

ENV PATH="/root/.cargo/bin:${PATH}"
RUN cargo install cargo-chef --locked

COPY --from=planner /app/recipe.json recipe.json
RUN --mount=type=cache,target=/usr/local/cargo/registry \
    --mount=type=cache,target=/usr/local/cargo/git \
    cargo chef cook --release \
    --features "default,cuda,candle-cuda,tensorrt,streams-all" \
    --bin snooze \
    --recipe-path recipe.json

COPY . .
RUN --mount=type=cache,target=/usr/local/cargo/registry \
    --mount=type=cache,target=/usr/local/cargo/git \
    cargo build --release \
    --features "default,cuda,candle-cuda,tensorrt,streams-all" \
    --bin snooze && \
    mkdir -p /app/dist/bin /app/dist/lib && \
    touch /app/dist/lib/.keep && \
    cp /app/target/release/snooze /app/dist/bin/snooze && \
    find /app/target/release -maxdepth 1 -name "libonnxruntime*.so*" -exec cp -L -t /app/dist/lib/ {} +

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

COPY --from=builder-slim /app/dist/bin/snooze /usr/local/bin/snooze
COPY --from=builder-slim /app/dist/lib/ /usr/local/lib/
RUN rm -f /usr/local/lib/.keep && ldconfig

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

COPY --from=builder-cpu /app/dist/bin/snooze /usr/local/bin/snooze
COPY --from=builder-cpu /app/dist/lib/ /usr/local/lib/
RUN rm -f /usr/local/lib/.keep && ldconfig

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
    LD_LIBRARY_PATH="/usr/local/lib:/usr/local/cuda/lib64:/app"

WORKDIR /app

EXPOSE 8080 50051

VOLUME ["/models"]

HEALTHCHECK --interval=10s --timeout=3s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8080/health/live || exit 1

COPY --from=builder-cuda /app/dist/bin/snooze /usr/local/bin/snooze
COPY --from=builder-cuda /app/dist/lib/ /usr/local/lib/
RUN rm -f /usr/local/lib/.keep && ldconfig

USER appuser

ENTRYPOINT ["snooze"]
CMD ["serve", "--host", "0.0.0.0", "--port", "8080", "--models-dir", "/models"]
