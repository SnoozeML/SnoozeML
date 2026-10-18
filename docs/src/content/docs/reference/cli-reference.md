---
title: "CLI Command Reference"
description: "This page provides an exhaustive reference of all CLI commands, arguments, environment variable overrides, and default values supported by the `snooze` binary."
---

This page provides an exhaustive reference of all CLI commands, arguments, environment variable overrides, and default values supported by the `snooze` binary.

---

## 1. Top-Level Subcommands Overview

| Command | Alias | Description |
| :--- | :--- | :--- |
| `snooze serve` | — | Starts the HTTP model inference and telemetry server. |
| `snooze download` | `hf-download` | Downloads a model repository from the Hugging Face Hub. |
| `snooze init` | — | Initializes or updates a `models.toml` file in a models directory. |
| `snooze models` | — | Subcommands for scanning, initializing, and managing model folders. |
| `snooze status` | — | Queries a running server to list models and residency states. |
| `snooze load` | — | Explicitly pre-warms a model into memory. |
| `snooze unload` | — | Explicitly deallocates a model's weights to free memory. |
| `snooze monitor` | — | Launches the live interactive terminal (TUI) dashboard. |

---

## 2. Command Details & Flags

### `snooze serve`
Starts the inference server daemon.

```bash
snooze serve [OPTIONS]
```

| Flag | Short | Default | Env Var | Description |
| :--- | :--- | :--- | :--- | :--- |
| `--host <HOST>` | — | `0.0.0.0` | — | Network interface address to bind. |
| `--port <PORT>` | `-p` | `8080` | — | TCP port for the HTTP REST server. |
| `--grpc [BOOL]` | — | `true` | `SNOOZE_GRPC`, `GRPC` | Enable or disable the gRPC server alongside HTTP. |
| `--no-grpc` | — | `false` | — | Disable gRPC server and only serve HTTP REST endpoints. |
| `--grpc-port <PORT>`| — | `50051` | `SNOOZE_GRPC_PORT`, `GRPC_PORT` | TCP port to bind the gRPC server. |
| `--models-dir <DIR>` | — | *None* | `SNOOZE_MODELS_DIR` | Directory containing models to discover and serve. |
| `--config <PATH>` | — | *None* | — | Path to an explicit `models.toml` or `models.json` file. |
| `--idle-timeout <SECS>` | — | `300` | — | Inactivity duration in seconds before scale-to-zero deallocation. |
| `--max-batch-size <N>` | — | `16` | — | Maximum micro-batch capacity. |
| `--max-delay-ms <MS>` | — | `10` | — | Maximum batch accumulation latency budget. |
| `--provider <NAME>` | — | `auto` | — | Preferred accelerator (`auto`, `gpu`, `cuda`, `mlx`, `cpu`, `tensorrt`). |
| `--max-loaded-models <N>`| — | *None* | — | Maximum concurrent loaded model instances in memory. |
| `--demo` | — | `false` | — | Enables the built-in mock testing model `demo-model`. |
| `--auth-enabled` | — | `false` | — | Explicitly enables authentication enforcement. |
| `--api-key <ROLE:KEY>` | — | *None* | `SNOOZE_API_KEY` | Adds an API key with role (`admin`, `infer`, `ro`). Omitting role defaults to `admin`. Repeatable. |
| `--jwt-secret <SECRET>` | — | *None* | `SNOOZE_JWT_SECRET` | HMAC secret key used to validate JSON Web Tokens. |
| `--protect-metrics` | — | `false` | — | Requires `Admin` role to query the `/metrics` endpoint. |

---

### `snooze download`
Downloads a repository from the Hugging Face Hub using the official `hf-hub` client.

```bash
snooze download <REPO> [OPTIONS]
```

| Argument / Flag | Short | Default | Env Var | Description |
| :--- | :--- | :--- | :--- | :--- |
| `<REPO>` | — | *Required* | — | Hugging Face repo ID (e.g. `openai-community/gpt2`). |
| `--models-dir <DIR>` | `-d` | `./models` | `SNOOZE_MODELS_DIR` | Root models directory destination. |
| `--name <NAME>` | `-n` | Repo name | — | Subfolder name inside models directory. |
| `--revision <REV>` | `-r` | `main` | — | Git commit, branch, or release tag to download. |
| `--include <GLOBS>` | `-i` | *All* | — | Comma-separated file globs to include (e.g. `"*.onnx,config.json"`). |
| `--exclude <GLOBS>` | `-e` | *None* | — | Comma-separated file globs to exclude. |
| `--token <TOKEN>` | — | *None* | `HF_TOKEN` | Hugging Face access token for private or gated models. |
| `--no-init` | — | `false` | — | Skips auto-updating `models.toml` after download. |

---

### `snooze init`
Generates or updates a `models.toml` file based on discovered weights.

```bash
snooze init [DIR] [--file <PATH>]
```

| Argument / Flag | Short | Default | Description |
| :--- | :--- | :--- | :--- |
| `[DIR]` | — | `./models` | Directory path containing model weights. |
| `--file <PATH>` | `-f` | `<DIR>/models.toml` | Target configuration file destination. |

---

### `snooze models scan`
Scans a model directory and displays discovered models, devices, and batch configurations without launching the server.

```bash
snooze models scan [DIR]
```

---

### `snooze status`
Queries a running server and prints model residency states.

```bash
snooze status [--endpoint <URL>]
```

| Flag | Default | Description |
| :--- | :--- | :--- |
| `--endpoint <URL>` | `http://127.0.0.1:8080` | Base URL of the target SnoozeML server. |

---

### `snooze load`
Pre-warms a model into memory.

```bash
snooze load --model-id <MODEL_ID> [--endpoint <URL>]
```

---

### `snooze unload`
Immediately deallocates a model from memory (scale-to-zero).

```bash
snooze unload --model-id <MODEL_ID> [--endpoint <URL>]
```

---

### `snooze monitor`
Launches the real-time terminal dashboard.

```bash
snooze monitor [--endpoint <URL>] [--interval-secs <SECS>]
```

| Flag | Default | Description |
| :--- | :--- | :--- |
| `--endpoint <URL>` | `http://127.0.0.1:8080` | Target server URL. |
| `--interval-secs <N>` | `1` | Screen refresh interval in seconds. |
