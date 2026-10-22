---
title: "Installation & Build"
description: "Install SnoozeML with a single zero-sudo command on Linux, macOS, and Windows, or compile from source with Cargo and Docker."
---

SnoozeML compiles down to a single standalone binary called `snooze`. You can install prebuilt binaries with a single command (without needing `sudo` or Administrator privileges) or compile from source with custom hardware acceleration flags.

---

## 1. One-Line Install (Recommended — Zero `sudo`)

### Linux & macOS

Run the official install script to download the prebuilt `snooze` binary and bundled ONNX Runtime libraries for your architecture (`x86_64` or `aarch64` / Apple Silicon), verify its SHA-256 checksum, and set up a rootless background daemon:

```bash
curl -fsSL https://raw.githubusercontent.com/SnoozeML/SnoozeML/main/install.sh | sh
```

**What this does (without requiring `sudo`):**
- Installs the binary and shared libraries to `~/.local/share/snoozeml/` and symlinks `~/.local/bin/snooze`.
- Initializes a default models directory at `~/.local/share/snoozeml/models` with `models.toml`.
- **On Linux (`systemd`)**: Creates and starts a user-mode systemd unit at `~/.config/systemd/user/snooze.service`.
- **On macOS (`launchd`)**: Creates and loads a user LaunchAgent at `~/Library/LaunchAgents/io.snoozeml.daemon.plist`.

#### Installer Options & Uninstalling

You can customize the installation by passing flags after `sh -s --`:

```bash
# Install the lightweight CPU-only variant (~10x smaller download, no GPU provider libraries)
curl -fsSL https://raw.githubusercontent.com/SnoozeML/SnoozeML/main/install.sh | sh -s -- --cpu

# Install binary only without creating or starting the background daemon
curl -fsSL https://raw.githubusercontent.com/SnoozeML/SnoozeML/main/install.sh | sh -s -- --no-service

# Install a specific release version
curl -fsSL https://raw.githubusercontent.com/SnoozeML/SnoozeML/main/install.sh | sh -s -- --version v0.1.0

# Uninstall the background service and binary (preserves your models directory)
curl -fsSL https://raw.githubusercontent.com/SnoozeML/SnoozeML/main/install.sh | sh -s -- --uninstall
```

### Windows (PowerShell)

On Windows (`x86_64`), run the PowerShell installer (no Administrator privileges required):

```powershell
# Standard build (includes CUDA/TensorRT GPU support)
irm https://raw.githubusercontent.com/SnoozeML/SnoozeML/main/install.ps1 | iex

# Lightweight CPU-only build
$env:SNOOZE_VARIANT="cpu"; irm https://raw.githubusercontent.com/SnoozeML/SnoozeML/main/install.ps1 | iex
```

This extracts `snooze.exe` and bundled ONNX Runtime DLLs into `%LOCALAPPDATA%\Programs\SnoozeML\bin`, initializes `%LOCALAPPDATA%\Programs\SnoozeML\models`, and adds the binary directory to your user `PATH`.

---

## 2. Prerequisites for Building from Source

If you prefer to compile SnoozeML from source (for example, to enable custom CUDA or TensorRT features), ensure your system has the following development tools installed:

- **Rust & Cargo**: Version 1.75 or later. Install via [rustup.rs](https://rustup.rs/):
  ```bash
  curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
  ```
- **C/C++ Build Essentials**: Required to link native execution providers (ONNX Runtime / Candle):
  - **Linux (Ubuntu/Debian)**: `sudo apt-get install build-essential cmake pkg-config libssl-dev`
  - **macOS**: `xcode-select --install` and `brew install cmake`
  - **Windows**: Visual Studio 2022 Build Tools (with the "Desktop development with C++" workload) and CMake.
- **(Optional) Documentation Tools**: If you wish to build or preview this documentation portal locally:
  - Node.js (>= 20) and npm: `npm --prefix docs install && npm --prefix docs run dev`
  - Python (>= 3.9) and Sphinx: `pip install -e "python[dev]" && sphinx-build -b html python/docs docs/dist/python`
- **(Optional) NVIDIA CUDA / TensorRT**:
  - NVIDIA Driver (>= 525.60.13 recommended)
  - CUDA Toolkit (11.8 or 12.x) and cuDNN if compiling with GPU support.

---

## 3. Installing via Cargo

You can install SnoozeML directly into your Cargo binary path:

```bash
cargo install --path .
```

To install with CUDA acceleration enabled:

```bash
cargo install --path . --features cuda
```

The resulting `snooze` binary will be available in your `$HOME/.cargo/bin` directory (make sure this is in your `PATH`).

---

## 4. Compiling from Source

Clone the repository:

```bash
git clone https://github.com/SnoozeML/SnoozeML.git
cd SnoozeML
```

### Build Profiles & Feature Flags

SnoozeML provides modular compilation profiles tailored for different deployment targets:

| Target Environment | Cargo Build Command | Included Capabilities |
| :--- | :--- | :--- |
| **Standard (CPU & Host)** | `cargo build --release` | Default build. Includes ONNX Runtime (CPU) and Candle (Safetensors) backends. |
| **NVIDIA CUDA** | `cargo build --release --features cuda` | Enables NVIDIA CUDA execution provider for ONNX and GPU acceleration for Candle. |
| **NVIDIA TensorRT** | `cargo build --release --features tensorrt` | Enables high-performance NVIDIA TensorRT execution provider. |
| **Apple Silicon (macOS)** | `cargo build --release --features coreml` | Enables Apple CoreML backend for ONNX Runtime alongside Metal acceleration for Candle. |
| **Lightweight / Mock** | `cargo build --release --no-default-features` | Ultra-fast compile without native ORT libraries, useful for CI/CD mock validation. |

Once compilation completes, the standalone executable is placed at:
```bash
# On Linux / macOS:
./target/release/snooze

# On Windows:
.\target\release\snooze.exe
```

---

## 5. Container Image Build (Docker)

SnoozeML includes a high-performance multi-stage Docker build pipeline powered by [cargo-chef](https://github.com/LukeMathWalker/cargo-chef) on Debian Trixie (`lukemathwalker/cargo-chef:latest-rust-1-slim-trixie` and `debian:trixie-slim`). It caches all third-party crate dependencies in a dedicated layer (`cargo chef cook`), allowing near-instant rebuilds (~0.3s cached) while generating a lean, non-root runtime image (~200MB).

### Building the Image

Compile the image locally:

```bash
# On Linux / macOS (Bash / Zsh):
DOCKER_BUILDKIT=1 docker build -t snoozeml:latest .

# On Windows (PowerShell):
$env:DOCKER_BUILDKIT=1; docker build -t snoozeml:latest .
# Or simply (BuildKit is enabled by default in modern Docker Desktop):
docker build -t snoozeml:latest .
```

### Running with Docker

Serve models from a local host directory:

```bash
docker run -d \
  --name snoozeml \
  -p 8080:8080 \
  -p 50051:50051 \
  -v $(pwd)/models:/models:ro \
  snoozeml:latest
```

Run in demonstration mode with the built-in mock model (no model files required):

```bash
docker run -it --rm \
  -p 8080:8080 \
  -p 50051:50051 \
  snoozeml:latest serve --demo
```

### Using Docker Compose

A standard `docker-compose.yml` is provided at the root of the repository:

```bash
docker compose up -d
```

---

## 6. Installing the Python SDK (`snoozeml`)

To query SnoozeML, manage model lifecycles, and export ML models from Python, install the official `snoozeml` client library directly from PyPI:

```bash
# Core lightweight HTTP REST client (zero compiled C-extensions)
pip install snoozeml

# High-performance native gRPC transport + NumPy array support
pip install "snoozeml[grpc]"

# Complete toolkit: gRPC, NumPy, scikit-learn, ONNX, and XGBoost converters
pip install "snoozeml[full]"
```

### Dependency Profiles & Extras

SnoozeML provides modular extras so you only install what your application requires:

| Extra | Command | Included Packages | Purpose |
| :--- | :--- | :--- | :--- |
| **Base** | `pip install snoozeml` | `requests` | Ultra-lightweight HTTP REST client for model lifecycle management (`load`, `unload`, `pin`, `unpin`), status polling, health probes, and metrics. Zero compiled C-extensions. |
| **`[grpc]`** | `pip install "snoozeml[grpc]"` | `grpcio`, `protobuf`, `numpy` | High-throughput native binary gRPC transport with automatic NumPy array serialization. |
| **`[converters]`** | `pip install "snoozeml[converters]"` | `numpy`, `scikit-learn`, `onnx`, `skl2onnx`, `xgboost[-cpu]` | Offline model export tools (`export_sklearn`, `export_xgboost`) targeting ONNX and pure-Rust tree formats. |
| **`[full]`** | `pip install "snoozeml[full]"` | `grpc`, `converters`, `worker` | All-in-one installation for client, gRPC, custom workers, and model converters with zero CUDA bloat. |
| **`[onnx]`** | `pip install "snoozeml[onnx]"` | `numpy`, `scikit-learn`, `onnx`, `skl2onnx` | Scikit-learn estimator and pipeline export to ONNX runtime format. |
| **`[xgboost]`** | `pip install "snoozeml[xgboost]"` | `numpy`, `xgboost-cpu` (Linux) / `xgboost` (macOS/Win) | XGBoost model export to `.xgb.json` for pure-Rust `hessboost` tree execution. Uses `xgboost-cpu` on Linux to avoid pulling heavy NVIDIA CUDA packages. |
| **`[torch]`** | `pip install "snoozeml[torch]"` | `torch` | PyTorch `nn.Module` export to ONNX. *(Duck-typing works automatically if PyTorch is already present).* |
| **`[worker]`** | `pip install "snoozeml[worker]"` | `numpy` | Minimal dependency profile for process-isolated custom Python worker models (`snooze_model.py`). |

For full API documentation and code walkthroughs, see the [Python SDK Guide](/SnoozeML/client/python-sdk/).

---

## 7. Verifying the Installation

Verify that the CLI is accessible and inspect the available commands:

```bash
# Locally:
snooze --help

# Or via Docker:
docker run --rm snoozeml:latest --help
```

You should see output similar to:

```text
High-performance ONNX & deep learning micro-batch inference daemon.

Usage: snooze <COMMAND>

Commands:
  serve     Start the inference daemon server
  models    Manage models directory and configuration files
  download  Download a model repository from Hugging Face Hub [aliases: hf-download]
  init      Quick shortcut to initialize or update a model directory configuration
  status    Query the status and active models of a running SnoozeML server
  load      Explicitly trigger loading a model into memory
  unload    Explicitly trigger unloading a model from memory (scale-to-zero override)
  monitor   Launch a live updating terminal dashboard for monitoring
  help      Print this message or the help of the given subcommand(s)

Options:
  -h, --help     Print help
  -V, --version  Print version
```

---

## 8. Helpful Environment Variables

SnoozeML respects several environment variables to streamline configuration in containerized and cloud environments:

- `SNOOZE_MODELS_DIR`: Sets the default root directory for model files (defaults to `./models`, or `/models` in the official container image).
- `SNOOZE_JWT_SECRET`: Sets the HMAC shared secret for validating JWT authentication tokens.
- `HF_TOKEN`: Sets the Hugging Face authentication token when downloading gated or private repositories.
- `RUST_LOG`: Configures structured logging output levels (e.g. `RUST_LOG=info` or `RUST_LOG=snooze_ml=debug`).
