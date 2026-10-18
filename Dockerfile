# ==============================================================================
# 1. BASE CHEF STAGE (Official cargo-chef on Debian Trixie)
# ==============================================================================
FROM lukemathwalker/cargo-chef:latest-rust-1-slim-trixie AS chef
WORKDIR /app

# Install standard C-toolchain dependencies needed for compiling native crates
# (e.g. openssl, C/C++ build scripts for tokenizers and ONNX Runtime bindings)
RUN apt-get update && apt-get install -y --no-install-recommends \
    pkg-config \
    libssl-dev \
    ca-certificates \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# ==============================================================================
# 2. PLANNER STAGE (Extracts recipe.json of project dependencies)
# ==============================================================================
FROM chef AS planner
COPY . .
RUN cargo chef prepare --recipe-path recipe.json

# ==============================================================================
# 3. BUILDER STAGE (Cooks dependencies and compiles application)
# ==============================================================================
FROM chef AS builder
WORKDIR /app

# Prepare staging directories for output binary and dynamic libraries
RUN mkdir -p /app/dist/bin /app/dist/lib && \
    touch /app/dist/lib/.keep

# Cache & build crate dependencies layer using recipe.json
COPY --from=planner /app/recipe.json recipe.json
RUN cargo chef cook --release --bin snooze --recipe-path recipe.json

# Copy full repository source code and build the final binary
COPY . .
RUN cargo build --release --bin snooze && \
    cp /app/target/release/snooze /app/dist/bin/snooze && \
    strip --strip-all /app/dist/bin/snooze && \
    find /app/target -name "libonnxruntime*.so*" -exec cp -t /app/dist/lib/ {} + 2>/dev/null || true

# ==============================================================================
# 4. RUNTIME STAGE (Official minimal Debian)
# ==============================================================================
FROM debian:trixie-slim AS runtime

# Install minimal runtime shared libraries:
# - ca-certificates: required for SSL/TLS root certificates (Hugging Face downloads)
# - curl: required for Docker/Kubernetes container health checks
# - libssl3t64: OpenSSL runtime shared libraries
# - libgomp1: GNU OpenMP runtime, required for multithreaded ONNX Runtime CPU execution
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
COPY --from=builder /app/dist/bin/snooze /usr/local/bin/snooze
COPY --from=builder /app/dist/lib/ /usr/local/lib/
RUN rm -f /usr/local/lib/.keep && ldconfig

# Application environment configuration
ENV SNOOZE_MODELS_DIR="/models" \
    RUST_LOG="info" \
    HF_HOME="/home/appuser/.cache/huggingface" \
    LD_LIBRARY_PATH="/usr/local/lib:/app"

WORKDIR /app
USER appuser

# Expose default HTTP / REST API and gRPC ports
EXPOSE 8080
EXPOSE 50051

# Volume mount point for external model weights and configurations
VOLUME ["/models"]

# Container health probe checking SnoozeML liveness endpoint
HEALTHCHECK --interval=10s --timeout=3s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8080/health/live || exit 1

ENTRYPOINT ["snooze"]
CMD ["serve", "--host", "0.0.0.0", "--port", "8080", "--models-dir", "/models"]
