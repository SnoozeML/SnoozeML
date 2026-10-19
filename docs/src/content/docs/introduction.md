---
title: "Introduction to SnoozeML"
description: "**SnoozeML** is a high-performance, single-binary, bare-metal inference daemon written in Rust for deep learning models. It is designed to serve multiple mac..."
---

**SnoozeML** is a high-performance, single-binary, bare-metal inference daemon written in Rust for deep learning models. It is designed to serve multiple machine learning models concurrently on resource-constrained hardware, edge devices, and cost-sensitive cloud infrastructure.

Traditional model-serving frameworks (such as Triton or TorchServe) often assume dedicated, persistent GPU memory allocations for every deployed model. In contrast, SnoozeML is built around **zero-idle resource reclamation**: models that are not actively serving traffic automatically release their weights from GPU VRAM and host RAM, allowing dozens of diverse models to coexist comfortably on a single machine.

When traffic arrives, SnoozeML lazily rehydrates model weights with sub-millisecond coordination, dynamically micro-batches concurrent incoming requests along `Axis(0)` to saturate hardware accelerators, and optionally spins up parallel replicas when traffic spikes.

---

## Why SnoozeML?

If you are running dozens of machine learning models—such as embedding models, rerankers, image classifiers, and tabular predictors—keeping every model pinned in GPU VRAM 24/7 is prohibitively expensive. In real-world workloads, many models experience bursty or intermittent traffic. 

SnoozeML solves this by bridging the gap between **serverless scale-to-zero economics** and **low-latency bare-metal performance**:

1. **Zero-Idle Scale-to-Zero**: Inactive models automatically drop their weights from VRAM/RAM after a configurable idle duration.
2. **Transparent Dynamic Micro-Batching**: Clients send single or batched inputs without coordinating with each other. SnoozeML coalesces concurrent requests within a microsecond/millisecond window into contiguous tensors, maximizing compute efficiency.
3. **Reactive Out-Of-Memory (OOM) Protection**: When physical GPU memory fills up, SnoozeML intelligently evicts the Least Recently Used (LRU) idle model to make room for new traffic, preventing Out-Of-Memory crashes.
4. **Horizontal Replica Autoscaling**: When request queues build up on a heavily loaded model, SnoozeML spins up replica instances on the fly and routes traffic using intelligent load-balancing strategies (such as `least-queue`).
5. **Multi-Format Runtime**: Native support for both **ONNX** models (`.onnx` via ONNX Runtime) and **Safetensors** models (`.safetensors` via Hugging Face Candle).
6. **Dual-Protocol Serving (HTTP REST & gRPC)**: Serve clients over standard HTTP REST (`Axum`) and ultra-low latency, binary gRPC (`Tonic`) simultaneously with unified dynamic batching and telemetry.
7. **Hugging Face Hub Integration**: Direct CLI downloads from the Hugging Face Hub with automatic directory scanning and configuration generation.
8. **Built-in Authentication & RBAC**: Secure endpoints using constant-time API keys and JSON Web Tokens (JWT) across both HTTP headers and gRPC metadata with fine-grained role-based permissions (`Admin`, `Inference`, `ReadOnly`).
9. **Production Observability**: Standard Prometheus metrics endpoint (`/metrics`), Kubernetes-ready health probes (`/health/live`, `/health/ready`, and `grpc.health.v1`), and an interactive terminal UI dashboard (`snooze monitor`).
10. **Zero-Panic Rust Reliability**: Engineered with a strict zero-panic policy (`#![deny(clippy::unwrap_used, clippy::expect_used, clippy::panic)]`), ensuring the daemon never terminates unexpectedly in production.

---

## How to Navigate This Book

This documentation is designed to get you up and running quickly while providing deep clarity into how the system behaves under load:

- **[Getting Started](/SnoozeML/getting-started/installation/)**: Install SnoozeML, run your first model in under two minutes, explore [interactive examples](/SnoozeML/getting-started/examples/), and download models from the Hugging Face Hub.
- **[Serving Models](/SnoozeML/serving/starting-the-server/)**: Learn how to launch the daemon, configure `models.toml`, enable hardware acceleration (CUDA, TensorRT, Apple Silicon), and manage dual REST and gRPC listeners.
- **[Scaling & Performance Mechanics](/SnoozeML/scaling/overview/)**: A deep dive into what to expect from the scaling system—how dynamic micro-batching works, how scale-to-zero behaves, cold-start latency vs. warm inference, replica scaling, and OOM protection.
- **[Authentication & Security](/SnoozeML/security/overview/)**: How to set up API keys, configure JWT tokens, manage role permissions, and secure operational HTTP and gRPC endpoints.
- **[Client Usage & Inference](/SnoozeML/client/inference-requests/)**: Step-by-step examples for querying models via [HTTP REST](/SnoozeML/client/inference-requests/) and binary [gRPC](/SnoozeML/client/grpc-inference/) using `curl`, `grpcurl`, Python, TypeScript, and Rust.
- **[Operations & Monitoring](/SnoozeML/operations/tui-monitor/)**: Monitor your cluster using the built-in TUI, scrape Prometheus metrics, configure HTTP and gRPC health probes, and review production deployment checklists.
- **[Reference & Architecture](/SnoozeML/reference/architecture/)**: A single-page architectural breakdown of internal components, followed by exhaustive [CLI](/SnoozeML/reference/cli-reference/), [REST API](/SnoozeML/reference/api-reference/), and [gRPC API](/SnoozeML/reference/grpc-api-reference/) references.