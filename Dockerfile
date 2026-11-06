# Stage 1: Builder
FROM ubuntu:24.04 AS builder

ENV DEBIAN_FRONTEND=noninteractive
ENV ORT_VERSION=1.31.1

RUN apt-get update && apt-get install -y \
    curl build-essential pkg-config libssl-dev protobuf-compiler libprotobuf-dev ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Install latest stable Rust (1.92+)
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --default-toolchain stable
ENV PATH="/root/.cargo/bin:${PATH}"

# Download ONNX Runtime 1.31.1 GPU (CUDA 13) shared libraries
RUN mkdir -p /opt/ort && \
    curl -fSL "https://github.com/microsoft/onnxruntime/releases/download/v${ORT_VERSION}/onnxruntime-linux-x64-gpu_cuda13-${ORT_VERSION}.tgz" \
    | tar -xz --strip-components=1 -C /opt/ort

WORKDIR /app
COPY . .

RUN cargo build --release --locked

# Stage 2: Runtime (Works on both CPU-only hosts and NVIDIA Container Toolkit GPU hosts)
FROM ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y \
    ca-certificates curl libssl3 libstdc++6 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /app/target/release/snooze /usr/local/bin/snooze
COPY --from=builder /opt/ort/lib/libonnxruntime* /usr/local/lib/
RUN ldconfig

ENV ORT_DYLIB_PATH=/usr/local/lib/libonnxruntime.so
ENTRYPOINT ["/usr/local/bin/snooze"]
