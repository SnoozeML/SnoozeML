---
title: "Production Deployment Guide"
description: "This guide covers operational best practices, deployment patterns, and OS tuning recommendations for deploying SnoozeML in mission-critical production enviro..."
---

This guide covers operational best practices, deployment patterns, and OS tuning recommendations for deploying SnoozeML in mission-critical production environments.

---

## 1. Systemd Service Deployment (Bare Metal / VM)

Running SnoozeML as a systemd service ensures automatic restarts, structured journal logging, and clean startup on boot. You can run SnoozeML either as a **rootless user service** (no `sudo` required, created automatically by `install.sh`) or as a **system-wide service**.

### 1.1 Rootless User Service (`~/.config/systemd/user/snooze.service`)

Because SnoozeML binds to unprivileged ports by default (`:8080` and `:50051`), it can run entirely under an unprivileged user account without `sudo`:

```ini
[Unit]
Description=SnoozeML High-Performance Inference Daemon (User Service)
After=network-online.target

[Service]
Type=simple
ExecStart=%h/.local/bin/snooze serve --models-dir %h/.local/share/snoozeml/models
Restart=on-failure
RestartSec=5
Environment="RUST_LOG=info"
Environment="SNOOZE_MODELS_DIR=%h/.local/share/snoozeml/models"

[Install]
WantedBy=default.target
```

Manage the user service without `sudo`:
```bash
systemctl --user daemon-reload
systemctl --user enable --now snooze.service
systemctl --user status snooze.service
journalctl --user -u snooze.service -f
```

> [!TIP]
> On headless Linux servers, if you want your rootless user service to start at boot and remain running after you log out of SSH, enable systemd user lingering once with `loginctl enable-linger $USER`.

### 1.2 System-Wide Service (`/etc/systemd/system/snoozeml.service`)

For shared multi-tenant servers managed by a dedicated system user (`snooze`):

```ini
[Unit]
Description=SnoozeML High-Performance Inference Daemon
After=network.target

[Service]
Type=simple
User=snooze
Group=snooze
WorkingDirectory=/opt/snoozeml
ExecStart=/usr/local/bin/snooze serve \
    --models-dir /opt/snoozeml/models \
    --config /opt/snoozeml/models.toml \
    --host 0.0.0.0 \
    --port 8080 \
    --grpc-port 50051 \
    --idle-timeout 300 \
    --max-batch-size 32 \
    --max-delay-ms 5
Restart=always
RestartSec=5s
LimitNOFILE=65535
Environment="RUST_LOG=info"
Environment="SNOOZE_JWT_SECRET=your-production-secret-key-32-chars-long"

[Install]
WantedBy=multi-user.target
```

Enable and start the service:
```bash
sudo systemctl daemon-reload
sudo systemctl enable --now snoozeml
sudo systemctl status snoozeml
```

---

## 2. Container Deployment (Docker & Kubernetes)

SnoozeML provides an optimized, multi-target Dockerfile supporting both **AMD64 (`x86_64`)** and **ARM64 (`aarch64`)** across three distinct deployment tiers:

### 2.1 Multi-Target Build Matrix

| Image Target | Features & Backends | Python Support | Architectures | Best For |
| :--- | :--- | :--- | :--- | :--- |
| **`slim`** | ONNX Runtime (CPU), Candle Safetensors, Tree models, Base streams | ❌ None (zero Python runtime) | `linux/amd64`<br/>`linux/arm64` | Edge gateways, microservices, minimal CVE surface (~144 MB) |
| **`cpu`** | All backends, all streaming brokers (`streams-all`), Python worker | ✅ Lean Python 3 + venv (`numpy`) | `linux/amd64`<br/>`linux/arm64` | Standard CPU deployments, custom Python workers (~344 MB) |
| **`cuda`** | All backends, ONNX CUDA (`ort?/cuda`), TensorRT, streams, Python worker | ✅ Lean Python 3 + venv (`numpy`) | `linux/amd64` | GPU servers (NVIDIA RTX / A100 / H100 / L4, ~426 MB) |

> [!NOTE]
> All targets utilize [`cargo-chef`](https://github.com/LukeMathWalker/cargo-chef) to cache compiled dependency layers independently, ensuring lightning-fast incremental builds when source code changes.

### 2.2 Building Containers

Build your desired target using Docker BuildKit:

```bash
# 1. Build the lightweight Slim target
docker build --target slim -t snooze-ml:slim .

# 2. Build the standard CPU + Python target
docker build --target cpu -t snooze-ml:cpu .

# 3. Build the CUDA GPU target
docker build --target cuda -t snooze-ml:cuda .
```

#### Declarative Multi-Arch Builds (`docker-bake.hcl`)

With `docker-bake.hcl`, you can compile cross-platform images for both AMD64 and ARM64 in parallel:

```bash
# Build multi-architecture slim images for AMD64 and ARM64
docker buildx bake slim

# Build standard CPU images for AMD64 and ARM64
docker buildx bake cpu

# Build all three targets across both architectures simultaneously
docker buildx bake all
```

### 2.3 Production Docker Run Examples

#### Slim Production Runner (CPU, read-only filesystem)
```bash
docker run -d \
  --name snoozeml-slim \
  --restart unless-stopped \
  --read-only \
  --cap-drop=ALL \
  --tmpfs /tmp:rw,noexec,nosuid,size=64m \
  -p 8080:8080 \
  -p 50051:50051 \
  -v /var/data/snoozeml/models:/models:ro \
  -e RUST_LOG=info \
  snooze-ml:slim
```

#### CUDA GPU Runner
```bash
docker run -d \
  --name snoozeml-gpu \
  --restart unless-stopped \
  --gpus all \
  -p 8080:8080 \
  -p 50051:50051 \
  -v /var/data/snoozeml/models:/models:ro \
  -e RUST_LOG=info \
  snooze-ml:cuda
```

### 2.4 Multi-Service Docker Compose (`docker-compose.yml`)

Use the repository's root [`docker-compose.yml`](docker-compose.yml) to run any variant locally:

```bash
# Run slim
docker compose up snooze-slim

# Run CPU with Python worker support
docker compose up snooze-cpu

# Run GPU-accelerated container
docker compose up snooze-cuda
```

---

## 3. TLS Encryption (Native & Reverse Proxy)

### 3.1 Native In-Process TLS
SnoozeML supports native in-process TLS termination for both HTTP REST and gRPC endpoints using Rustls (`aws-lc-rs`).

By default (if no flags or paths are provided), TLS is **disabled** (`false`), and SnoozeML logs a security warning at startup. To enable native TLS:

```bash
snooze serve \
    --tls \
    --tls-cert /etc/ssl/certs/snooze.crt \
    --tls-key /etc/ssl/private/snooze.key
```

Or via environment variables:
```bash
export SNOOZE_TLS=true
export SNOOZE_TLS_CERT=/etc/ssl/certs/snooze.crt
export SNOOZE_TLS_KEY=/etc/ssl/private/snooze.key
snooze serve
```

When enabled, HTTP requests are served over HTTPS and gRPC requests require TLS.

---

### 3.2 Reverse Proxy Configuration (Nginx)

When exposing SnoozeML to public networks, you can also place it behind a reverse proxy like Nginx or Caddy to handle TLS termination, request rate limiting, and HTTP/2 buffering. Nginx can proxy both HTTP REST and gRPC traffic:

### `/etc/nginx/conf.d/snoozeml.conf`
```nginx
upstream snoozeml_http {
    server 127.0.0.1:8080;
    keepalive 64;
}

upstream snoozeml_grpc {
    server 127.0.0.1:50051;
    keepalive 64;
}

server {
    listen 443 ssl http2;
    server_name api.snoozeml.mycompany.com;

    ssl_certificate /etc/letsencrypt/live/snoozeml/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/snoozeml/privkey.pem;

    # Critical: Set proxy timeouts higher than your batching delay + inference duration
    proxy_connect_timeout 5s;
    proxy_send_timeout    30s;
    proxy_read_timeout    30s;
    grpc_read_timeout     30s;
    grpc_send_timeout     30s;

    # Allow large tensor payloads
    client_max_body_size 64M;

    # 1. gRPC Routing (Inference & Health)
    location /snoozeml.v1. {
        grpc_pass grpc://snoozeml_grpc;
        grpc_set_header X-Real-IP $remote_addr;
        grpc_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
    }

    location /grpc.health.v1. {
        grpc_pass grpc://snoozeml_grpc;
    }

    # 2. HTTP REST Routing
    location / {
        proxy_pass http://snoozeml_http;
        proxy_http_version 1.1;
        proxy_set_header Connection "";
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}
```

---

## 4. OS & Hardware Tuning

### 1. Increase Open File Descriptor Limits
Under high concurrency, each TCP connection requires a file descriptor. Increase limits in `/etc/security/limits.conf`:
```text
snooze   soft   nofile   65535
snooze   hard   nofile   65535
```

### 2. NVIDIA GPU Persistence Mode
Enable persistence mode on NVIDIA drivers to eliminate GPU driver reinitialization latency:
```bash
sudo nvidia-smi -pm 1
```

### 3. Fast NVMe Storage for Models
Store your models on fast local NVMe SSDs rather than network-attached storage (NFS/EFS). Because SnoozeML lazily streams weights during cold starts, NVMe storage keeps cold-start latency down to tens of milliseconds.

---

## 5. Production Readiness Checklist

Before going live, verify the following configuration points:

- [ ] **Authentication Enabled**: Ensure `--auth-enabled` is set or API keys / JWT secrets are configured.
- [ ] **Metrics Protected / Scraped**: Verify Prometheus is scraping `/metrics` and alerting on error spikes.
- [ ] **Health Probes Configured**: Orchestrator (Kubernetes / ECS) has configured `/health/live` and `/health/ready`.
- [ ] **Batching Budgets Tuned**: Verify that `max_delay_ms` satisfies your SLA latency constraints.
- [ ] **Memory Caps Configured**: Set `max_loaded_models` if running on fixed-capacity GPUs to enforce deterministic LRU eviction.
- [ ] **Zero-Panic Enforcement**: Running release builds compiled with default safety checks.