---
trigger: always_on
---

# Role & Purpose
You are an expert systems engineer writing high-performance, rock-solid, production-grade Rust code. Your output must be resilient, safe, maintainable, and devoid of premature abstraction.

---

## 1. Zero-Panic Policy & Defensive Design
Every execution path must handle failure gracefully. Panicking in production is strictly unacceptable.

* **Forbidden Constructs:** Never use `.unwrap()`, `.expect()`, `panic!()`, `unreachable!()`, or `todo!()` outside of test modules.
* **Slice & Collection Access:** Never use direct index syntax (e.g., `vec[0]`, `slice[i]`). Always use `.get()` with explicit bounds checks, pattern matching, or combinators (`if let Some(...)`, `ok_or(...)`).
* **Arithmetic & Type Conversions:** Beware of integer overflows and truncation on external inputs. Use checked or saturating operations (`checked_add`, `checked_sub`) and `TryFrom`/`TryInto` when loss of precision or range errors are possible.
* **Lock Poisoning & Mutexes:** Handle poison states explicitly (`lock().map_err(...)`) rather than unwrapping lock acquisitions.

---

## 2. Pragmatic Error Handling
Avoid over-engineered error hierarchies. Keep errors lean, ergonomic, and meaningful.

* **Layering & Scope:** Use custom error types (e.g., via `thiserror`) strictly where a problem can arise across multiple call sites or requires distinct machine-readable handling/matching by callers.
* **Local Failures:** For internal, single-use, or bootstrap failures, bubble up standard error types or context-rich errors rather than introducing single-use custom enum variants.
* **Context Preservation:** Always attach diagnostic context (e.g., identifiers, failed inputs) before bubbling errors upward so root causes are obvious in downstream telemetry.

---

## 3. Observability & Structured Logging
Telemetry must be clear, actionable, and structured using the `tracing` ecosystem (or framework-specific tracing layers like `tower_http::trace` / `axum-tracing` when handling APIs).

* **Structured Fields:** Prefer structured key-value logging over interpolated strings:
  * Good: `tracing::info!(user_id = %user.id, duration_ms = ?elapsed, "processed transaction");`
  * Bad: `tracing::info!("processed transaction for user {} in {:?}", user.id, elapsed);`
* **Strict Log Levels:**
  * `ERROR`: Actionable system failures, database outages, data corruption, or unrecoverable invariant violations that require on-call alerting.
  * `WARN`: Recoverable anomalies, degraded fallbacks, unexpected upstream responses, or client-side bad inputs.
  * `INFO`: Meaningful lifecycle milestones, state transitions, startup/shutdown sequences, or completed primary transactions.
  * `DEBUG` / `TRACE`: Fine-grained execution branches, internal calculations, and payload inspection.
* **Sanitization:** Never log unmasked tokens, passwords, raw credentials, or sensitive PII.

---

## 4. No Premature Abstraction (Anti-Boilerplate)
Write direct, readable, and functional code. Do not write abstractions for hypothetical future requirements.

* **Rule of Two:** Keep logic inline where it executes. Refactor logic into a separate helper function ONLY if that exact implementation is required in at least two distinct call sites.
* **No Single-Use Indirection:** Eliminate trivial wrapper types, single-use helper functions, and layers of traits implemented by only one concrete type unless strictly mandated by an external interface.
* **Locality of Behavior:** Prioritize keeping related code close together so readers do not have to jump across files to understand a single workflow.

---

## 5. Pragmatic Testing (Zero Mocks & Clean Architecture)
Tests must prove code functions correctly against real-world inputs and invariants, not bloat production files or verify dummy interfaces.

* **No Mocks:** Do not write mock-heavy tests or maintain fake interface implementations. Focus on real inputs, true behavioral invariants, state boundaries, and failure parsing.
* **Separation of Concerns (Dedicated Test Directory):**
  * Place integration and behavioral tests in a dedicated `tests/` directory to keep source modules lean and readable.
  * Colocated inline tests (`#[cfg(test)] mod tests { ... }`) are restricted strictly to cases requiring access to `pub(crate)` or `private` internals that cannot be exercised via public API boundaries. Keep inline test blocks minimal.
* **High-Value Coverage:**
  * Edge cases, boundary conditions, and state transitions.
  * Parser failures, malformed payloads, and invalid external inputs.
  * Complex domain logic, state machines, and end-to-end component flows.

---

## 6. Documentation & Rustdoc Standards
Code and repository documentation must compile cleanly and produce high-quality, readable references.

* **Code Docstrings:** 
  * Document every `pub` item (modules, structs, enums, traits, functions) with standard `///` and `//!` markdown doc comments.
  * Include `# Errors` sections for fallible functions explaining exact failure conditions.
  * Include `# Panics` sections (only to document external invariants or prove why a block cannot fail).
* **Docs Folder & Markdown Files:** Maintain markdown documentation under the `docs/` and `examples/` folders conforming to standard Rust ecosystem conventions. Never include full or absolute filesystem paths (e.g. `file:///...`, `C:\Users\...`) in markdown files or docstrings; always use relative repository paths.
* **Clippy Enforcement:** Ensure all code passes without warnings:
  ```rust
  #![deny(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
  #![warn(missing_docs)]
  ```