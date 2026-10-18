---
trigger: always_on
---

# Documentation Standards & Integrity (`docs/`, `python/docs/`, `README.md`)

SnoozeML documentation is served as a unified developer portal combining Astro Starlight (`docs/`) and Sphinx documentation (`python/docs/`). All documentation contributions must adhere to the following rules.

---

## 1. No Absolute Filesystem Paths (Strict Rule)
* **Never use absolute or machine-specific paths** in markdown files, docstrings, notebooks, or commit messages:
  - **Forbidden:** `file:///C:/Users/...`, `C:\Users\...`, `/home/...`, `/tmp/...`
  - **Required:** Relative repository paths:
    - Good: `examples/01_sklearn_onnx.ipynb`, `src/types.rs`, `python/snoozeml/client.py`
    - Good doc link: `[API Reference](../reference/api-reference.md)`

---

## 2. Dual Starlight & Sphinx Parity
* **Synchronized Documentation Updates:** Whenever adding, renaming, or modifying:
  1. **CLI Commands or Flags (`src/cli/mod.rs`)**: Update `docs/src/content/docs/reference/cli-reference.md` and `docs/src/content/docs/getting-started/quickstart.md`.
  2. **HTTP REST Endpoints (`src/server/http.rs`)**: Update `docs/src/content/docs/reference/api-reference.md`.
  3. **gRPC RPC Methods (`src/server/grpc/`)**: Update `docs/src/content/docs/reference/grpc-api-reference.md`.
  4. **Configuration Options (`src/config.rs`)**: Update `docs/src/content/docs/serving/configuration.md`.
  5. **Streaming Brokers (`src/stream/`)**: Update `docs/src/content/docs/streaming/configuration.md` and `quickstart.md`.
  6. **Python SDK Methods (`python/snoozeml/`)**: Update docstrings in `python/snoozeml/` and `docs/src/content/docs/client/python-sdk.md`.

---

## 3. Formatting, Frontmatter & Alerts
* **Astro Starlight Frontmatter:** Every markdown document in `docs/src/content/docs/` must begin with valid YAML frontmatter containing:
  ```markdown
  ---
  title: Page Title
  description: Brief one-line description of the page content.
  ---
  ```
* **GitHub-Style Alerts:** Use standard GitHub alerts for emphasized notes:
  - `> [!NOTE]` for contextual background or tips.
  - `> [!IMPORTANT]` for essential requirements or constraints.
  - `> [!WARNING]` for deprecations or potential issues.
  - `> [!CAUTION]` for high-risk operations or data loss risks.
* **No Consecutive Alerts:** Avoid stacking alerts directly adjacent to one another.

---

## 4. Verification & Build Gate
Before finalizing any documentation changes, always verify both documentation builds complete without errors or broken links:

```powershell
# 1. Build Astro Starlight main portal and guides
npm --prefix docs run build

# 2. Build Sphinx Python SDK reference into Starlight dist
sphinx-build -b html python/docs docs/dist/python
```
