# Documentation Structure & Cross-Linking Reference

This document outlines the organization of SnoozeML's Astro Starlight content collections (`docs/src/content/docs/`) and the Sphinx Python documentation (`python/docs/`).

---

## 1. Astro Starlight Content Directory Map

```text
docs/src/content/docs/
├── index.mdx                       # Landing hero page
├── introduction.md                 # Project vision, features & performance
├── getting-started/
│   ├── installation.md             # Cargo, prebuilt binaries, Docker
│   ├── quickstart.md               # First model download and curl infer
│   └── downloading-models.md       # snooze download CLI & HF Hub auth
├── serving/
│   ├── starting-the-server.md      # snooze serve flags, demo mode, multi-tenant
│   ├── configuration.md            # models.toml & models.json specification
│   ├── supported-formats.md        # ONNX, Safetensors, HessBoost, Python workers
│   ├── custom-models.md            # Custom Python worker guide
│   └── hardware-acceleration.md    # CUDA, TensorRT, CoreML execution providers
├── scaling/
│   ├── overview.md                 # Concurrency and scale architecture
│   ├── dynamic-batching.md         # Axis(0) micro-batching and latency budgets
│   ├── scale-to-zero.md            # LRU sweeper, pinning, differential timeouts
│   ├── memory-and-oom.md           # Reactive OOM eviction and anti-thrashing
│   └── horizontal-autoscaling.md   # Multi-replica routing and scale-down
├── streaming/
│   ├── overview.md                 # Real-time event streaming concepts
│   ├── quickstart.md               # Kafka, RabbitMQ, and Redis examples
│   └── configuration.md            # [[streams]] table reference
├── client/
│   ├── python-sdk.md               # SnoozeClient usage and async streaming
│   ├── grpc-inference.md           # Native gRPC and protoc integration
│   ├── inference-requests.md       # REST JSON payload format
│   ├── tensors.md                  # Binary wire formatting & dtypes
│   └── errors.md                   # Canonical error mapping table
├── operations/
│   ├── production.md               # Linux systemd, Docker, Kubernetes deployment
│   ├── tui-monitor.md              # Interactive Ratatui dashboard keybindings
│   ├── prometheus-metrics.md       # Exported Prometheus counters and histograms
│   └── health-probes.md            # /health/live and /health/ready semantics
└── reference/
    ├── cli-reference.md            # Clap subcommands, options, and flags
    ├── api-reference.md            # Axum HTTP routes and RBAC roles
    ├── grpc-api-reference.md       # Tonic proto services and metadata auth
    └── architecture.md             # Subsystem state machines and thread models
```

---

## 2. Sphinx Python SDK Integration (`python/docs/`)

The Python SDK uses Sphinx to generate API documentation from docstrings in `python/snoozeml/`:
- Source directory: `python/docs/`
- Target output: `docs/dist/python/` (served alongside Starlight under the `/python/` sub-path).
- Command: `sphinx-build -b html python/docs docs/dist/python`

---

## 3. Cross-Linking Guidelines
- Use standard markdown links pointing to relative files within the documentation tree.
- Never use backslashes (`\`) in markdown URLs, even when developing on Windows. Always use forward slashes (`/`).
