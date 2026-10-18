---
name: docs-and-examples
description: >-
  Runbook for authoring and maintaining SnoozeML documentation (Astro Starlight in
  docs/, Sphinx in python/docs/), interactive Jupyter notebooks (examples/), and
  preventing absolute path leaks. Use this skill when adding user guides,
  updating API references, or adding example models and tutorials.
---

# Documentation & Interactive Examples Runbook

SnoozeML provides a unified developer documentation portal using Astro Starlight (`docs/`) integrated with Sphinx Python SDK documentation (`python/docs/`), accompanied by interactive Jupyter notebooks in `examples/`.

---

## 1. Absolute Path Prohibition (Critical Rule)

Never commit absolute or machine-specific filesystem paths in markdown files, notebooks, or docstrings:
- **Forbidden:** `file:///C:/Users/...`, `C:\Users\...`, `/home/runner/...`, `/tmp/...`
- **Required:** Relative repository paths (e.g. `examples/01_sklearn_onnx.ipynb`, `src/types.rs`, `python/snoozeml/client.py`).

---

## 2. Authoring Starlight Documentation Pages (`docs/src/content/docs/`)

When documenting new features or updating existing guides:

### Step 1: Create or Update Markdown / MDX Page
Place the file in the appropriate category under `docs/src/content/docs/`:
- `getting-started/`: Installation, downloading models, quickstart.
- `serving/`: Starting server, configuration options, custom models, hardware acceleration.
- `scaling/`: Dynamic micro-batching, LRU scale-to-zero, OOM recovery, horizontal autoscaling.
- `streaming/`: Broker quickstarts, configuration tables, error handling.
- `client/`: Python SDK, gRPC client, REST requests, tensor wire format.
- `operations/`: Health probes, Prometheus metrics, TUI dashboard, production deployment.
- `reference/`: CLI commands, REST API, gRPC protobuf definitions, architecture specifications.

### Step 2: Frontmatter & Heading Conventions
Start every page with valid YAML frontmatter:
```markdown
---
title: Page Title Here
description: Brief one-line summary for SEO and navigation previews.
---
```
Use standard GitHub alerts (`> [!NOTE]`, `> [!IMPORTANT]`, `> [!WARNING]`). Avoid stacking alerts directly adjacent to one another.

### Step 3: Update Navigation Sidebar (`docs/astro.config.mjs`)
If you created a new page, add its link to the corresponding sidebar group in `docs/astro.config.mjs` under `starlight({ sidebar: [...] })`.

For full content collection details, see [references/docs-structure.md](./references/docs-structure.md).

---

## 3. Authoring Jupyter Notebooks in `examples/`

When adding interactive example notebooks (e.g. `examples/05_new_workflow.ipynb`):
1. **Clean Outputs**: Execute all cells from top to bottom so all plots and outputs are rendered.
2. **Relative Paths**: Load sample data and export model artifacts using paths relative to the notebook directory (e.g. `./my_model.onnx`).
3. **Markdown Companion**: Provide a concise `README.md` in the example subdirectory explaining requirements, commands to run the notebook, and commands to smoke test with SnoozeML.

---

## 4. Verification & Build Gate

Always verify both documentation builds complete without errors or broken links:

```powershell
# 1. Build Astro Starlight main portal and guides
npm --prefix docs run build

# 2. Build Sphinx Python SDK reference into Starlight dist
sphinx-build -b html python/docs docs/dist/python
```
