---
title: "Interactive TUI Monitor"
description: "SnoozeML includes a built-in terminal dashboard monitor (`snooze monitor`), built with [Ratatui](https://github.com/ratatui/ratatui) and [Crossterm](https://..."
---

SnoozeML includes a built-in terminal dashboard monitor (`snooze monitor`), built with [Ratatui](https://github.com/ratatui/ratatui) and [Crossterm](https://github.com/crossterm-rs/crossterm). It provides a real-time visual interface for inspecting model states, active queues, and throughput without requiring external monitoring infrastructure.

---

## 1. Launching the Monitor

Run the `monitor` command, pointing it to your running SnoozeML server:

```bash
snooze monitor --endpoint http://127.0.0.1:8080
```

By default, the dashboard refreshes every **1 second**. You can adjust the refresh rate with `--interval-secs`:

```bash
# Refresh every 2 seconds:
snooze monitor --endpoint http://127.0.0.1:8080 --interval-secs 2
```

---

## 2. Dashboard Layout & Metrics

The TUI monitor displays three primary sections:

```text
┌── SnoozeML Cluster Monitor ───────────────────────────────────────────────┐
│ Server: http://127.0.0.1:8080    Status: ONLINE    Models: 3 (1 Resident) │
├───────────────────────────────────────────────────────────────────────────┤
│ Model ID          State    Device   Batch Size   Timeout   Last Accessed  │
│ ───────────────────────────────────────────────────────────────────────── │
│ all-minilm-l6-v2  READY    CUDA     16           300s      4s ago         │
│ resnet50          IDLE     CUDA     64           600s      18m ago        │
│ bge-small         IDLE     CPU      16           180s      1h ago         │
├───────────────────────────────────────────────────────────────────────────┤
│ Telemetry & Throughput                                                    │
│ Requests Processed: 1,420    Success Rate: 100.0%    Avg Latency: 3.4ms   │
│ Batches Dispatched: 118      Batch Factor: 12.0x     Active Workers: 1    │
└───────────────────────────────────────────────────────────────────────────┘
```

### Explanations of Columns:
- **`Model ID`**: The registered unique model identifier.
- **`State`**:
  - `IDLE` (Gray): Weights deallocated from memory (0 MB VRAM/RAM).
  - `LOADING` (Yellow): Model weights currently being loaded from disk into memory.
  - `READY` (Green): Memory-resident and prepared for instantaneous warm execution.
  - `BUSY` (Cyan): Forward pass currently executing on the hardware accelerator.
- **`Device`**: Active accelerator (`CUDA`, `CoreML`, `Metal`, or `CPU`).
- **`Last Accessed`**: Elapsed time since the last inference request was served. Indicates how close the model is to triggering scale-to-zero deallocation.

---

## 3. Keyboard Shortcuts

| Key | Action |
| :--- | :--- |
| `q` or `Esc` | Safely exits the monitor and restores the terminal. |
| `Ctrl+C` | Terminates the monitor process. |