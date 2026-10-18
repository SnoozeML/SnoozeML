//! Process-isolated worker backend for user-defined models.
//!
//! A *worker model* is a directory containing [`WORKER_ENTRYPOINT_FILE`] (`snooze_model.py`)
//! with a user-defined class exposing `load(model_dir, device)` and `predict(inputs)`.
//! Each replica runs as a separate child process spawned from the embedded Python worker
//! runtime (`python/snoozeml/worker.py`). The daemon talks to it over stdin/stdout with a
//! small length-prefixed binary protocol:
//!
//! ```text
//! [u32 LE frame_len][u8 msg_type][u32 LE header_len][header JSON][raw LE f32 bytes...]
//! ```
//!
//! Process isolation means:
//! - Unloading (drop) terminates the process, so all host RAM and VRAM is returned to the OS.
//! - Crashes in user code (segfaults, aborts) cannot take down the daemon.
//! - Replicas are separate interpreters, so there is no GIL contention between them.

use crate::backend::ModelBackend;
use crate::types::{InferenceError, TensorPayload};
use crossbeam_channel::{Receiver, RecvTimeoutError};
use serde::Deserialize;
use std::collections::{HashMap, VecDeque};
use std::io::{BufRead, BufReader, Read, Write};
use std::path::{Path, PathBuf};
use std::process::{Child, ChildStdin, Command, Stdio};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

/// File name whose presence marks a directory as a worker model.
pub const WORKER_ENTRYPOINT_FILE: &str = "snooze_model.py";
/// Optional JSON manifest file inside a worker model directory.
pub const WORKER_MANIFEST_FILE: &str = "snooze_model.json";
/// Wire protocol version spoken between the daemon and workers.
pub const PROTOCOL_VERSION: u32 = 1;
/// Upper bound on a single protocol frame, protecting the daemon from allocation bombs.
pub const MAX_FRAME_BYTES: usize = 1 << 30;

/// Embedded worker runtime; passed to the interpreter via `-c` so that the protocol
/// implementation always matches this binary and no files are written to disk.
const WORKER_SCRIPT: &str = include_str!("../../python/snoozeml/worker.py");

const MSG_READY: u8 = 1;
const MSG_INFER: u8 = 2;
const MSG_RESULT: u8 = 3;
const MSG_ERROR: u8 = 4;
const MSG_SHUTDOWN: u8 = 5;

const DEFAULT_LOAD_TIMEOUT: Duration = Duration::from_secs(300);
const DEFAULT_REQUEST_TIMEOUT: Duration = Duration::from_secs(60);
const SHUTDOWN_GRACE: Duration = Duration::from_secs(5);
const STDERR_TAIL_LINES: usize = 40;

/// Optional per-model settings read from [`WORKER_MANIFEST_FILE`].
///
/// Unknown keys are rejected so that typos surface as load errors instead of being ignored.
#[derive(Debug, Clone, Default, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct WorkerManifest {
    /// Python interpreter to run (relative paths resolve against the model directory).
    pub python: Option<String>,
    /// Python file defining the model class (defaults to [`WORKER_ENTRYPOINT_FILE`]).
    pub entrypoint: Option<String>,
    /// Name of the model class inside the entrypoint (defaults to `"Model"`).
    pub class: Option<String>,
    /// Maximum seconds allowed for `load()` before the worker is killed (default 300).
    pub load_timeout_secs: Option<u64>,
    /// Maximum seconds allowed for one batched `predict()` call (default 60).
    pub request_timeout_secs: Option<u64>,
    /// Tensor representation passed to `predict()`: `"numpy"` (default) or `"raw"`.
    pub tensor_format: Option<String>,
    /// Extra environment variables for the worker process.
    pub env: HashMap<String, String>,
}

/// Configuration for constructing a [`WorkerBackend`].
#[derive(Debug, Clone)]
pub struct WorkerConfig {
    /// Directory containing the entrypoint, weights and optional manifest.
    pub model_dir: PathBuf,
    /// Device string handed to `load()` (`"auto"`, `"cpu"`, `"cuda"`, `"cuda:N"`, `"mps"`).
    pub device: String,
}

impl WorkerConfig {
    /// Builds a config from a model path and a SnoozeML provider preference.
    ///
    /// `path` may be the model directory or the entrypoint file itself. Provider names are
    /// mapped to framework device strings (`gpu`/`cuda`/`tensorrt` → `cuda`,
    /// `mlx`/`metal`/`coreml` → `mps`, `cuda:N` kept as-is, otherwise `auto`).
    #[must_use]
    pub fn from_path(path: &str, provider: Option<&str>) -> Self {
        let path_obj = Path::new(path);
        let model_dir = if path_obj.is_file() {
            path_obj
                .parent()
                .map_or_else(|| PathBuf::from("."), Path::to_path_buf)
        } else {
            path_obj.to_path_buf()
        };
        let device = match provider {
            Some("cuda" | "gpu" | "tensorrt") => "cuda".to_string(),
            Some("mlx" | "metal" | "coreml" | "mps") => "mps".to_string(),
            Some("cpu") => "cpu".to_string(),
            Some(other) if other.starts_with("cuda:") => other.to_string(),
            _ => "auto".to_string(),
        };
        Self { model_dir, device }
    }
}

/// Returns `true` if `path` should be served by the worker backend: a directory containing
/// [`WORKER_ENTRYPOINT_FILE`] or [`WORKER_MANIFEST_FILE`], either of those files themselves,
/// or any path with provider `"python"` or `"worker"`.
#[must_use]
pub fn is_worker_model(path: &str, provider: Option<&str>) -> bool {
    if matches!(provider, Some("python" | "worker")) {
        return true;
    }
    let path_obj = Path::new(path);
    if path_obj.is_dir() {
        return path_obj.join(WORKER_ENTRYPOINT_FILE).is_file()
            || path_obj.join(WORKER_MANIFEST_FILE).is_file();
    }
    let is_entrypoint = path_obj.file_name().and_then(|n| n.to_str())
        == Some(WORKER_ENTRYPOINT_FILE)
        && path_obj.is_file();
    let is_manifest = path_obj.file_name().and_then(|n| n.to_str()) == Some(WORKER_MANIFEST_FILE)
        && path_obj.is_file();
    is_entrypoint || is_manifest
}

/// A decoded protocol frame.
struct Frame {
    kind: u8,
    header: Vec<u8>,
    payload: Vec<u8>,
}

#[derive(Deserialize)]
struct TensorMeta {
    name: String,
    shape: Vec<usize>,
    #[serde(default = "default_dtype")]
    dtype: String,
}

fn default_dtype() -> String {
    "f32".to_string()
}

#[derive(Deserialize)]
struct TensorsHeader {
    tensors: Vec<TensorMeta>,
}

#[derive(Deserialize)]
struct ErrorHeader {
    kind: String,
    message: String,
}

#[derive(Deserialize)]
struct ReadyHeader {
    protocol: u32,
    #[serde(default)]
    info: serde_json::Value,
}

/// Everything required to (re)spawn a worker process.
#[derive(Debug, Clone)]
struct LaunchSpec {
    python: String,
    model_dir: PathBuf,
    device: String,
    entrypoint: String,
    class: String,
    tensor_format: String,
    env: HashMap<String, String>,
    load_timeout: Duration,
}

/// Live connection to one worker process.
struct WorkerProcess {
    child: Child,
    stdin: ChildStdin,
    frames: Receiver<Result<Frame, String>>,
}

/// Mutable worker state guarded by the backend mutex.
struct WorkerSlot {
    process: Option<WorkerProcess>,
    /// Reason the last process died; cleared on successful respawn.
    dead_reason: Option<String>,
}

/// Model backend delegating inference to an isolated worker process.
pub struct WorkerBackend {
    spec: LaunchSpec,
    request_timeout: Duration,
    slot: Mutex<WorkerSlot>,
    stderr_tail: Arc<Mutex<VecDeque<String>>>,
}

impl std::fmt::Debug for WorkerBackend {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("WorkerBackend")
            .field("model_dir", &self.spec.model_dir)
            .field("python", &self.spec.python)
            .field("device", &self.spec.device)
            .field("pid", &self.pid())
            .finish()
    }
}

impl WorkerBackend {
    /// Spawns a worker process and blocks until the user model reports ready.
    ///
    /// # Errors
    /// - [`InferenceError::NotFound`] if the entrypoint file does not exist.
    /// - [`InferenceError::Runtime`] if the manifest is invalid, the interpreter cannot be
    ///   launched, loading fails or times out, or the worker speaks an incompatible protocol.
    /// - [`InferenceError::OutOfMemory`] if the worker reports memory exhaustion while loading.
    pub fn new(config: &WorkerConfig) -> Result<Self, InferenceError> {
        let model_dir = &config.model_dir;
        let manifest_path = model_dir.join(WORKER_MANIFEST_FILE);
        let manifest: WorkerManifest = if manifest_path.is_file() {
            let raw = std::fs::read_to_string(&manifest_path).map_err(|e| {
                InferenceError::Runtime(format!(
                    "Failed to read worker manifest '{}': {e}",
                    manifest_path.display()
                ))
            })?;
            serde_json::from_str(&raw).map_err(|e| {
                InferenceError::Runtime(format!(
                    "Invalid worker manifest '{}': {e}",
                    manifest_path.display()
                ))
            })?
        } else {
            WorkerManifest::default()
        };

        let entrypoint = manifest
            .entrypoint
            .clone()
            .unwrap_or_else(|| WORKER_ENTRYPOINT_FILE.to_string());
        if !model_dir.join(&entrypoint).is_file() {
            return Err(InferenceError::NotFound(format!(
                "Worker entrypoint '{entrypoint}' not found in '{}'",
                model_dir.display()
            )));
        }

        let tensor_format = manifest
            .tensor_format
            .clone()
            .unwrap_or_else(|| "numpy".to_string());
        if tensor_format != "numpy" && tensor_format != "raw" {
            return Err(InferenceError::Runtime(format!(
                "Invalid tensor_format '{tensor_format}' in '{}': expected \"numpy\" or \"raw\"",
                manifest_path.display()
            )));
        }

        // Interpreter resolution: manifest > model-local virtualenv > SNOOZE_PYTHON > workspace virtualenv > python3.
        let local_venv = if cfg!(windows) {
            model_dir.join(".venv").join("Scripts").join("python.exe")
        } else {
            model_dir.join(".venv").join("bin").join("python")
        };
        let workspace_venv = if cfg!(windows) {
            PathBuf::from(".venv").join("Scripts").join("python.exe")
        } else {
            PathBuf::from(".venv").join("bin").join("python")
        };
        let python = match manifest.python.as_deref() {
            Some(p) if Path::new(p).is_relative() && p.contains(['/', '\\']) => {
                model_dir.join(p).to_string_lossy().to_string()
            }
            Some(p) => p.to_string(),
            None if local_venv.is_file() => local_venv.to_string_lossy().to_string(),
            None => match std::env::var("SNOOZE_PYTHON") {
                Ok(p) => p,
                Err(_) if workspace_venv.is_file() => workspace_venv.to_string_lossy().to_string(),
                Err(_) => if cfg!(windows) { "python" } else { "python3" }.to_string(),
            },
        };

        let backend = Self {
            spec: LaunchSpec {
                python,
                model_dir: model_dir.clone(),
                device: config.device.clone(),
                entrypoint,
                class: manifest.class.unwrap_or_else(|| "Model".to_string()),
                tensor_format,
                env: manifest.env,
                load_timeout: manifest
                    .load_timeout_secs
                    .map_or(DEFAULT_LOAD_TIMEOUT, Duration::from_secs),
            },
            request_timeout: manifest
                .request_timeout_secs
                .map_or(DEFAULT_REQUEST_TIMEOUT, Duration::from_secs),
            slot: Mutex::new(WorkerSlot {
                process: None,
                dead_reason: None,
            }),
            stderr_tail: Arc::new(Mutex::new(VecDeque::with_capacity(STDERR_TAIL_LINES))),
        };

        let process = backend.spawn()?;
        match backend.slot.lock() {
            Ok(mut slot) => slot.process = Some(process),
            Err(_) => {
                return Err(InferenceError::Runtime(
                    "Worker state lock poisoned during initialization".into(),
                ))
            }
        }
        Ok(backend)
    }

    /// Returns the OS process id of the running worker, if any.
    #[must_use]
    pub fn pid(&self) -> Option<u32> {
        self.slot
            .lock()
            .ok()
            .and_then(|slot| slot.process.as_ref().map(|p| p.child.id()))
    }

    /// Recent stderr lines from the worker, appended to error messages for diagnosis.
    fn stderr_context(&self) -> String {
        let tail = self
            .stderr_tail
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        if tail.is_empty() {
            String::new()
        } else {
            let lines: Vec<&str> = tail
                .iter()
                .rev()
                .take(8)
                .rev()
                .map(String::as_str)
                .collect();
            format!(" | worker stderr: {}", lines.join(" / "))
        }
    }

    /// Launches the interpreter and waits for `READY`.
    fn spawn(&self) -> Result<WorkerProcess, InferenceError> {
        let spec = &self.spec;
        let started = Instant::now();
        tracing::info!(
            model_dir = %spec.model_dir.display(),
            python = %spec.python,
            device = %spec.device,
            "Spawning model worker process"
        );

        self.stderr_tail
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .clear();

        let mut child = Command::new(&spec.python)
            .arg("-u")
            .arg("-c")
            .arg(WORKER_SCRIPT)
            .arg("--model-dir")
            .arg(&spec.model_dir)
            .arg("--device")
            .arg(&spec.device)
            .arg("--entrypoint")
            .arg(&spec.entrypoint)
            .arg("--class")
            .arg(&spec.class)
            .arg("--tensor-format")
            .arg(&spec.tensor_format)
            .current_dir(&spec.model_dir)
            .envs(&spec.env)
            .env("PYTHONUNBUFFERED", "1")
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .map_err(|e| {
                InferenceError::Runtime(format!(
                    "Failed to launch Python interpreter '{}' for worker model '{}': {e}. \
                     Install Python 3, set \"python\" in {WORKER_MANIFEST_FILE}, or set SNOOZE_PYTHON",
                    spec.python,
                    spec.model_dir.display()
                ))
            })?;
        let pid = child.id();

        let (Some(stdin), Some(stdout), Some(stderr)) =
            (child.stdin.take(), child.stdout.take(), child.stderr.take())
        else {
            let _ = child.kill();
            let _ = child.wait();
            return Err(InferenceError::Runtime(
                "Worker process stdio pipes were not captured".into(),
            ));
        };

        // stderr forwarder: structured logs plus a bounded tail for error context.
        let tail = self.stderr_tail.clone();
        let log_dir = spec.model_dir.display().to_string();
        let _ = std::thread::Builder::new()
            .name(format!("snooze-worker-stderr-{pid}"))
            .spawn(move || {
                for line in BufReader::new(stderr).lines() {
                    let Ok(line) = line else { break };
                    if line.contains("Error") || line.starts_with("Traceback") {
                        tracing::warn!(model_dir = %log_dir, pid, line = %line, "worker stderr");
                    } else {
                        tracing::debug!(model_dir = %log_dir, pid, line = %line, "worker stderr");
                    }
                    let mut guard = tail
                        .lock()
                        .unwrap_or_else(std::sync::PoisonError::into_inner);
                    if guard.len() >= STDERR_TAIL_LINES {
                        guard.pop_front();
                    }
                    guard.push_back(line);
                }
            });

        // stdout reader: decodes frames and forwards them; ends on EOF or protocol error.
        let (tx, frames) = crossbeam_channel::unbounded();
        let reader_spawn = std::thread::Builder::new()
            .name(format!("snooze-worker-stdout-{pid}"))
            .spawn(move || {
                let mut reader = BufReader::with_capacity(1 << 16, stdout);
                loop {
                    let res = read_frame(&mut reader);
                    let stop = res.is_err();
                    if tx.send(res).is_err() || stop {
                        break;
                    }
                }
            });
        if let Err(e) = reader_spawn {
            let _ = child.kill();
            let _ = child.wait();
            return Err(InferenceError::Runtime(format!(
                "Failed to spawn worker reader thread: {e}"
            )));
        }

        let mut process = WorkerProcess {
            child,
            stdin,
            frames,
        };

        let fail = |process: &mut WorkerProcess, err: InferenceError| {
            let _ = process.child.kill();
            let _ = process.child.wait();
            Err(err)
        };

        match process.frames.recv_timeout(spec.load_timeout) {
            Ok(Ok(frame)) if frame.kind == MSG_READY => {
                let ready: ReadyHeader = match serde_json::from_slice(&frame.header) {
                    Ok(r) => r,
                    Err(e) => {
                        return fail(
                            &mut process,
                            InferenceError::Runtime(format!("Malformed worker READY header: {e}")),
                        )
                    }
                };
                if ready.protocol != PROTOCOL_VERSION {
                    return fail(
                        &mut process,
                        InferenceError::Runtime(format!(
                            "Worker protocol version {} is incompatible with daemon version {PROTOCOL_VERSION}",
                            ready.protocol
                        )),
                    );
                }
                tracing::info!(
                    model_dir = %spec.model_dir.display(),
                    pid,
                    info = %ready.info,
                    load_ms = started.elapsed().as_millis(),
                    "Model worker ready"
                );
                Ok(process)
            }
            Ok(Ok(frame)) if frame.kind == MSG_ERROR => {
                let err = error_from_frame(&frame, "Worker failed to load model");
                fail(&mut process, err)
            }
            Ok(Ok(frame)) => fail(
                &mut process,
                InferenceError::Runtime(format!(
                    "Unexpected message type {} from worker during load",
                    frame.kind
                )),
            ),
            Ok(Err(e)) => {
                // Give the stderr thread a moment to capture the traceback.
                let status = wait_exit(&mut process.child, Duration::from_millis(500));
                let err = InferenceError::Runtime(format!(
                    "Worker exited during load ({e}; status: {status}){}",
                    self.stderr_context()
                ));
                fail(&mut process, err)
            }
            Err(RecvTimeoutError::Timeout) => fail(
                &mut process,
                InferenceError::Runtime(format!(
                    "Worker did not become ready within {}s (load_timeout_secs){}",
                    spec.load_timeout.as_secs(),
                    self.stderr_context()
                )),
            ),
            Err(RecvTimeoutError::Disconnected) => fail(
                &mut process,
                InferenceError::Runtime(format!(
                    "Worker output channel closed during load{}",
                    self.stderr_context()
                )),
            ),
        }
    }

    /// Sends one batch to `process` and waits for its reply.
    ///
    /// Returns `Err((error, fatal))`. When `fatal` is `true` the process is no longer usable.
    fn round_trip(
        &self,
        process: &mut WorkerProcess,
        request: &[u8],
    ) -> Result<Vec<TensorPayload>, (InferenceError, bool)> {
        if let Err(e) = process
            .stdin
            .write_all(request)
            .and_then(|()| process.stdin.flush())
        {
            return Err((
                InferenceError::Runtime(format!(
                    "Failed to send batch to worker: {e}{}",
                    self.stderr_context()
                )),
                true,
            ));
        }

        match process.frames.recv_timeout(self.request_timeout) {
            Ok(Ok(frame)) if frame.kind == MSG_RESULT => {
                decode_tensors(&frame).map_err(|e| (e, false))
            }
            Ok(Ok(frame)) if frame.kind == MSG_ERROR => {
                Err((error_from_frame(&frame, "Worker predict() failed"), false))
            }
            Ok(Ok(frame)) => Err((
                InferenceError::Runtime(format!(
                    "Unexpected message type {} from worker",
                    frame.kind
                )),
                true,
            )),
            Ok(Err(e)) => Err((
                InferenceError::Runtime(format!(
                    "Worker process terminated during inference ({e}){}",
                    self.stderr_context()
                )),
                true,
            )),
            Err(RecvTimeoutError::Timeout) => Err((
                InferenceError::Runtime(format!(
                    "Worker did not respond within {}s (request_timeout_secs); worker killed",
                    self.request_timeout.as_secs()
                )),
                true,
            )),
            Err(RecvTimeoutError::Disconnected) => Err((
                InferenceError::Runtime(format!(
                    "Worker output channel closed{}",
                    self.stderr_context()
                )),
                true,
            )),
        }
    }
}

impl ModelBackend for WorkerBackend {
    /// Sends the collated batch to the worker process and returns its outputs.
    ///
    /// If the previous worker process died (crash, timeout), a new one is spawned first.
    ///
    /// # Errors
    /// - [`InferenceError::InvalidInput`] if the worker reports invalid client input.
    /// - [`InferenceError::OutOfMemory`] if the worker reports memory exhaustion.
    /// - [`InferenceError::Runtime`] on worker failures, crashes, timeouts or protocol errors.
    fn infer(&self, inputs: &[TensorPayload]) -> Result<Vec<TensorPayload>, InferenceError> {
        let request = encode_infer_frame(inputs)?;

        let mut slot = self.slot.lock().map_err(|_| {
            InferenceError::Runtime("Worker state lock poisoned by a previous failure".into())
        })?;

        if slot.process.is_none() {
            tracing::warn!(
                model_dir = %self.spec.model_dir.display(),
                reason = slot.dead_reason.as_deref().unwrap_or("unknown"),
                "Worker process is not running; respawning"
            );
            let process = self.spawn()?;
            slot.process = Some(process);
            slot.dead_reason = None;
        }

        let Some(process) = slot.process.as_mut() else {
            return Err(InferenceError::Runtime("Worker process unavailable".into()));
        };

        match self.round_trip(process, &request) {
            Ok(outputs) => Ok(outputs),
            Err((err, fatal)) => {
                if fatal {
                    if let Some(mut dead) = slot.process.take() {
                        let _ = dead.child.kill();
                        let _ = dead.child.wait();
                    }
                    tracing::warn!(
                        model_dir = %self.spec.model_dir.display(),
                        error = %err,
                        "Worker process marked dead; it will be respawned on the next request"
                    );
                    slot.dead_reason = Some(err.to_string());
                }
                Err(err)
            }
        }
    }

    fn provider_name(&self) -> &'static str {
        "python-worker"
    }
}

impl Drop for WorkerBackend {
    /// Requests a graceful shutdown and reaps the process on a detached thread, so dropping
    /// (e.g. scale-to-zero unload inside an async task) never blocks the caller.
    fn drop(&mut self) {
        let slot = self
            .slot
            .get_mut()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let Some(mut process) = slot.process.take() else {
            return;
        };
        let pid = process.child.id();
        let _ = process
            .stdin
            .write_all(&encode_frame(MSG_SHUTDOWN, b"{}", &[]))
            .and_then(|()| process.stdin.flush());
        let model_dir = self.spec.model_dir.display().to_string();
        let reaper = std::thread::Builder::new()
            .name(format!("snooze-worker-reaper-{pid}"))
            .spawn(move || {
                drop(process.stdin); // EOF also makes the worker exit.
                let status = wait_exit(&mut process.child, SHUTDOWN_GRACE);
                if status == "still running" {
                    tracing::warn!(model_dir = %model_dir, pid, "Worker ignored shutdown; killing");
                    let _ = process.child.kill();
                    let _ = process.child.wait();
                }
                tracing::info!(model_dir = %model_dir, pid, "Model worker process terminated");
            });
        if let Err(e) = reaper {
            tracing::error!(pid, error = %e, "Failed to spawn worker reaper thread");
        }
    }
}

/// Polls a child for exit up to `timeout`, returning a printable status.
fn wait_exit(child: &mut Child, timeout: Duration) -> String {
    let deadline = Instant::now() + timeout;
    loop {
        match child.try_wait() {
            Ok(Some(status)) => return status.to_string(),
            Ok(None) if Instant::now() < deadline => std::thread::sleep(Duration::from_millis(25)),
            Ok(None) => return "still running".to_string(),
            Err(e) => return format!("unknown ({e})"),
        }
    }
}

/// Maps an `ERROR` frame to an [`InferenceError`] according to its `kind`.
fn error_from_frame(frame: &Frame, context: &str) -> InferenceError {
    match serde_json::from_slice::<ErrorHeader>(&frame.header) {
        Ok(h) => match h.kind.as_str() {
            "oom" => InferenceError::OutOfMemory(format!("{context}: {}", h.message)),
            "invalid_input" => InferenceError::InvalidInput(h.message),
            _ => InferenceError::Runtime(format!("{context}: {}", h.message)),
        },
        Err(e) => InferenceError::Runtime(format!("{context}: malformed error frame ({e})")),
    }
}

/// Serializes a frame: `[u32 frame_len][u8 kind][u32 header_len][header][payload chunks]`.
fn encode_frame(kind: u8, header: &[u8], chunks: &[&[u8]]) -> Vec<u8> {
    let payload_len: usize = chunks.iter().map(|c| c.len()).sum();
    let frame_len = 5usize
        .saturating_add(header.len())
        .saturating_add(payload_len);
    let mut buf = Vec::with_capacity(frame_len.saturating_add(4));
    buf.extend_from_slice(&u32::try_from(frame_len).unwrap_or(u32::MAX).to_le_bytes());
    buf.push(kind);
    buf.extend_from_slice(
        &u32::try_from(header.len())
            .unwrap_or(u32::MAX)
            .to_le_bytes(),
    );
    buf.extend_from_slice(header);
    for chunk in chunks {
        buf.extend_from_slice(chunk);
    }
    buf
}

/// Builds an `INFER` frame from the collated input tensors in a single allocation.
fn encode_infer_frame(inputs: &[TensorPayload]) -> Result<Vec<u8>, InferenceError> {
    let mut metas = Vec::with_capacity(inputs.len());
    let mut total_payload_bytes: usize = 0;
    for tensor in inputs {
        let shape = tensor.data.shape().to_vec();
        metas.push(serde_json::json!({ "name": tensor.name, "shape": shape, "dtype": "f32" }));
        let nbytes = tensor.data.len().checked_mul(4).ok_or_else(|| {
            InferenceError::InvalidInput(format!("Tensor '{}' byte size overflows", tensor.name))
        })?;
        total_payload_bytes = total_payload_bytes.checked_add(nbytes).ok_or_else(|| {
            InferenceError::InvalidInput(
                "Total input tensors payload byte size overflows".to_string(),
            )
        })?;
    }
    let header = serde_json::to_vec(&serde_json::json!({ "tensors": metas }))
        .map_err(|e| InferenceError::Runtime(format!("Failed to encode worker header: {e}")))?;
    let frame_len = 5usize
        .checked_add(header.len())
        .and_then(|n| n.checked_add(total_payload_bytes))
        .filter(|&n| n <= MAX_FRAME_BYTES)
        .ok_or_else(|| {
            InferenceError::InvalidInput(format!(
                "Batch exceeds the worker frame limit of {MAX_FRAME_BYTES} bytes"
            ))
        })?;
    let frame_len_u32 = u32::try_from(frame_len)
        .map_err(|_| InferenceError::InvalidInput("Frame length exceeds u32 limit".to_string()))?;
    let header_len_u32 = u32::try_from(header.len())
        .map_err(|_| InferenceError::InvalidInput("Header length exceeds u32 limit".to_string()))?;
    let total_buf_len = frame_len
        .checked_add(4)
        .ok_or_else(|| InferenceError::InvalidInput("Total buffer length overflows".to_string()))?;

    let mut buf = Vec::with_capacity(total_buf_len);
    buf.extend_from_slice(&frame_len_u32.to_le_bytes());
    buf.push(MSG_INFER);
    buf.extend_from_slice(&header_len_u32.to_le_bytes());
    buf.extend_from_slice(&header);
    for tensor in inputs {
        match tensor.data.as_slice() {
            #[cfg(target_endian = "little")]
            Some(slice) => buf.extend_from_slice(bytemuck::cast_slice(slice)),
            _ => {
                for v in &tensor.data {
                    buf.extend_from_slice(&v.to_le_bytes());
                }
            }
        }
    }
    Ok(buf)
}

/// Reads one frame, validating lengths before allocating.
fn read_frame(reader: &mut impl Read) -> Result<Frame, String> {
    let mut prefix = [0u8; 9];
    reader
        .read_exact(&mut prefix)
        .map_err(|e| format!("worker stdout closed: {e}"))?;
    let frame_len_bytes: [u8; 4] = prefix
        .get(0..4)
        .and_then(|s| s.try_into().ok())
        .ok_or("truncated frame length")?;
    let frame_len = usize::try_from(u32::from_le_bytes(frame_len_bytes))
        .map_err(|_| "frame length does not fit usize".to_string())?;
    if !(5..=MAX_FRAME_BYTES).contains(&frame_len) {
        return Err(format!(
            "invalid frame length {frame_len} (allowed 5..={MAX_FRAME_BYTES})"
        ));
    }
    let kind = *prefix.get(4).ok_or("missing frame kind")?;
    let header_len_bytes: [u8; 4] = prefix
        .get(5..9)
        .and_then(|s| s.try_into().ok())
        .ok_or("frame too short for header length")?;
    let header_len = usize::try_from(u32::from_le_bytes(header_len_bytes))
        .map_err(|_| "header length does not fit usize".to_string())?;
    let min_frame_len = header_len
        .checked_add(5)
        .ok_or_else(|| format!("header length {header_len} overflows usize"))?;
    if min_frame_len > frame_len {
        return Err(format!(
            "header length {header_len} exceeds frame length {frame_len}"
        ));
    }
    let payload_len = frame_len - min_frame_len;
    let mut header = vec![0u8; header_len];
    reader
        .read_exact(&mut header)
        .map_err(|e| format!("truncated frame header: {e}"))?;
    let mut payload = vec![0u8; payload_len];
    reader
        .read_exact(&mut payload)
        .map_err(|e| format!("truncated frame payload: {e}"))?;
    Ok(Frame {
        kind,
        header,
        payload,
    })
}

/// Decodes the tensors of a `RESULT` frame. Malformed worker output maps to `Runtime`
/// (a server-side fault), never `InvalidInput`.
fn decode_tensors(frame: &Frame) -> Result<Vec<TensorPayload>, InferenceError> {
    let header: TensorsHeader = serde_json::from_slice(&frame.header)
        .map_err(|e| InferenceError::Runtime(format!("Malformed worker RESULT header: {e}")))?;
    let mut outputs = Vec::with_capacity(header.tensors.len());
    let mut offset = 0usize;
    for meta in header.tensors {
        if meta.dtype != "f32" {
            return Err(InferenceError::Runtime(format!(
                "Worker output '{}' has unsupported dtype '{}'",
                meta.name, meta.dtype
            )));
        }
        let end = meta
            .shape
            .iter()
            .try_fold(1usize, |acc, &d| acc.checked_mul(d))
            .and_then(|numel| numel.checked_mul(4))
            .and_then(|nbytes| offset.checked_add(nbytes))
            .ok_or_else(|| {
                InferenceError::Runtime(format!("Worker output '{}' shape overflows", meta.name))
            })?;
        let bytes = frame.payload.get(offset..end).ok_or_else(|| {
            InferenceError::Runtime(format!(
                "Worker output '{}' exceeds payload size ({} bytes)",
                meta.name,
                frame.payload.len()
            ))
        })?;
        let (chunks, _) = bytes.as_chunks::<4>();
        let data: Vec<f32> = chunks.iter().map(|&c| f32::from_le_bytes(c)).collect();
        let name = meta.name;
        let tensor = TensorPayload::from_shape_vec(name.clone(), meta.shape, data)
            .map_err(|e| InferenceError::Runtime(format!("Invalid worker output '{name}': {e}")))?;
        outputs.push(tensor);
        offset = end;
    }
    if offset != frame.payload.len() {
        return Err(InferenceError::Runtime(format!(
            "Worker RESULT payload has {} trailing bytes",
            frame.payload.len().saturating_sub(offset)
        )));
    }
    Ok(outputs)
}
