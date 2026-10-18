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
RUN cargo chef cook --release \
    --no-default-features \
    --features "ort-backend,candle-backend,tree-backend,streams" \
    --bin snooze \
    --recipe-path recipe.json

COPY . .
RUN cargo build --release \
    --no-default-features \
    --features "ort-backend,candle-backend,tree-backend,streams" \
    --bin snooze
RUN mkdir -p /app/dist/bin /app/dist/lib && \
    touch /app/dist/lib/.keep && \
    cp /app/target/release/snooze /app/dist/bin/snooze && \
    strip --strip-all /app/dist/bin/snooze && \
    find /app/target/release -maxdepth 1 -name "libonnxruntime*.so*" -exec cp -L -t /app/dist/lib/ {} +

# ==============================================================================
# 3B. BUILDER — CPU (All backends + CPU Python worker + all streams)
# ==============================================================================
FROM chef AS builder-cpu
WORKDIR /app
COPY --from=planner /app/recipe.json recipe.json
RUN cargo chef cook --release \
    --features "default,streams-all" \
    --bin snooze \
    --recipe-path recipe.json

COPY . .
RUN cargo build --release \
    --features "default,streams-all" \
    --bin snooze
RUN mkdir -p /app/dist/bin /app/dist/lib && \
    touch /app/dist/lib/.keep && \
    cp /app/target/release/snooze /app/dist/bin/snooze && \
    strip --strip-all /app/dist/bin/snooze && \
    find /app/target/release -maxdepth 1 -name "libonnxruntime*.so*" -exec cp -L -t /app/dist/lib/ {} +

# ==============================================================================
# 3C. BUILDER — CUDA (All backends + CUDA GPU + Python worker + all streams)
# ==============================================================================
FROM chef AS builder-cuda
WORKDIR /app
COPY --from=planner /app/recipe.json recipe.json
RUN cargo chef cook --release \
    --features "default,cuda,streams-all" \
    --bin snooze \
    --recipe-path recipe.json

COPY . .
RUN cargo build --release \
    --features "default,cuda,streams-all" \
    --bin snooze
RUN mkdir -p /app/dist/bin /app/dist/lib && \
    touch /app/dist/lib/.keep && \
    cp /app/target/release/snooze /app/dist/bin/snooze && \
    strip --strip-all /app/dist/bin/snooze && \
    find /app/target/release -maxdepth 1 -name "libonnxruntime*.so*" -exec cp -L -t /app/dist/lib/ {} +

# ==============================================================================
# 4A. TARGET: SLIM (Smallest footprint, zero Python)
# ==============================================================================
FROM debian:trixie-slim AS slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    curl \
    libssl3t64 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Security: Create non-privileged system user and group (UID 10001)
RUN useradd -u 10001 -m -U -s /bin/bash appuser && \
    mkdir -p /models /app /home/appuser/.cache && \
    chown -R appuser:appuser /models /app /home/appuser

# Copy compiled standalone binary and dynamic libraries into system search path
COPY --from=builder-slim /app/dist/bin/snooze /usr/local/bin/snooze
COPY --from=builder-slim /app/dist/lib/ /usr/local/lib/
RUN rm -f /usr/local/lib/.keep && ldconfig

# Application environment configuration
ENV SNOOZE_MODELS_DIR="/models" \
    RUST_LOG="info" \
    HF_HOME="/home/appuser/.cache/huggingface" \
    LD_LIBRARY_PATH="/usr/local/lib:/app"

WORKDIR /app
USER appuser

EXPOSE 8080
EXPOSE 50051

VOLUME ["/models"]

HEALTHCHECK --interval=10s --timeout=3s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8080/health/live || exit 1

ENTRYPOINT ["snooze"]
CMD ["serve", "--host", "0.0.0.0", "--port", "8080", "--models-dir", "/models"]

# ==============================================================================
# 4B. TARGET: CPU (Standard CPU inference with lean Python worker runtime)
# ==============================================================================
FROM debian:trixie-slim AS cpu

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    curl \
    libssl3t64 \
    libgomp1 \
    python3 \
    python3-venv \
    && rm -rf /var/lib/apt/lists/*

# Isolated virtual environment with NumPy only for custom Python worker models
RUN python3 -m venv /opt/venv && \
    /opt/venv/bin/pip install --no-cache-dir --upgrade pip && \
    /opt/venv/bin/pip install --no-cache-dir numpy

# Security: Create non-privileged system user and group (UID 10001)
RUN useradd -u 10001 -m -U -s /bin/bash appuser && \
    mkdir -p /models /app /home/appuser/.cache && \
    chown -R appuser:appuser /models /app /home/appuser /opt/venv

COPY --from=builder-cpu /app/dist/bin/snooze /usr/local/bin/snooze
COPY --from=builder-cpu /app/dist/lib/ /usr/local/lib/
RUN rm -f /usr/local/lib/.keep && ldconfig

ENV SNOOZE_MODELS_DIR="/models" \
    RUST_LOG="info" \
    HF_HOME="/home/appuser/.cache/huggingface" \
    LD_LIBRARY_PATH="/usr/local/lib:/app" \
    PATH="/opt/venv/bin:${PATH}" \
    SNOOZE_PYTHON="/opt/venv/bin/python" \
    PYTHONUNBUFFERED="1"

WORKDIR /app
USER appuser

EXPOSE 8080
EXPOSE 50051

VOLUME ["/models"]

HEALTHCHECK --interval=10s --timeout=3s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8080/health/live || exit 1

ENTRYPOINT ["snooze"]
CMD ["serve", "--host", "0.0.0.0", "--port", "8080", "--models-dir", "/models"]

# ==============================================================================
# 4C. TARGET: CUDA (Full features, GPU acceleration & lean Python worker)
# ==============================================================================
FROM debian:trixie-slim AS cuda

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    curl \
    libssl3t64 \
    libgomp1 \
    python3 \
    python3-venv \
    && rm -rf /var/lib/apt/lists/*

# Isolated virtual environment with NumPy only for custom Python worker models
RUN python3 -m venv /opt/venv && \
    /opt/venv/bin/pip install --no-cache-dir --upgrade pip && \
    /opt/venv/bin/pip install --no-cache-dir numpy

# Security: Create non-privileged system user and group (UID 10001)
RUN useradd -u 10001 -m -U -s /bin/bash appuser && \
    mkdir -p /models /app /home/appuser/.cache && \
    chown -R appuser:appuser /models /app /home/appuser /opt/venv

COPY --from=builder-cuda /app/dist/bin/snooze /usr/local/bin/snooze
COPY --from=builder-cuda /app/dist/lib/ /usr/local/lib/
RUN rm -f /usr/local/lib/.keep && ldconfig

# NVIDIA Container Runtime hooks (compatible with both x86_64 and aarch64/sbsa)
ENV NVIDIA_VISIBLE_DEVICES="all" \
    NVIDIA_DRIVER_CAPABILITIES="compute,utility" \
    SNOOZE_MODELS_DIR="/models" \
    RUST_LOG="info" \
    HF_HOME="/home/appuser/.cache/huggingface" \
    LD_LIBRARY_PATH="/usr/local/lib:/usr/local/cuda/lib64:/usr/local/cuda/targets/sbsa-linux/lib:/app" \
    PATH="/opt/venv/bin:${PATH}" \
    SNOOZE_PYTHON="/opt/venv/bin/python" \
    PYTHONUNBUFFERED="1"

WORKDIR /app
USER appuser

EXPOSE 8080
EXPOSE 50051

VOLUME ["/models"]

HEALTHCHECK --interval=10s --timeout=3s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8080/health/live || exit 1

ENTRYPOINT ["snooze"]
CMD ["serve", "--host", "0.0.0.0", "--port", "8080", "--models-dir", "/models"]
