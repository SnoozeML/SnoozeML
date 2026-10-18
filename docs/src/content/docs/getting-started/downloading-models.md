---
title: "Downloading Models"
description: "SnoozeML includes a built-in Hugging Face Hub client (`snooze download`), powered by the official Rust `hf-hub` crate. It allows you to download model reposi..."
---

SnoozeML includes a built-in Hugging Face Hub client (`snooze download`), powered by the official Rust `hf-hub` crate. It allows you to download model repositories directly from Hugging Face without requiring Python, Git LFS, or third-party download scripts.

---

## 1. Quick Download

To download a public Hugging Face model repository into your default models directory (`./models`):

```bash
snooze download sentence-transformers/all-MiniLM-L6-v2
```

### What Happens Automatically:
1. SnoozeML connects to the Hugging Face Hub API.
2. It downloads the repository files into `./models/all-MiniLM-L6-v2/`.
3. It inspects the downloaded files for `.onnx` and `.safetensors` model weights.
4. It **automatically creates or updates `models.toml`** in your models directory with the newly registered model stub!

You can immediately serve the newly downloaded model:
```bash
snooze serve --models-dir ./models
```

---

## 2. Filtering Files with `--include` and `--exclude`

Hugging Face repositories often contain multiple weight formats (PyTorch `.bin`, TensorFlow `.h5`, Flax `.msgpack`, ONNX `.onnx`, and Safetensors `.safetensors`). Downloading all formats can waste tens of gigabytes of disk space and bandwidth.

Use the `-i, --include` flag with comma-separated glob patterns to only download the weights and metadata SnoozeML needs:

```bash
# Download only ONNX weights and companion configs:
snooze download sentence-transformers/all-MiniLM-L6-v2 \
  --include "*.onnx,config.json,tokenizer.json"

# Download only Safetensors weights:
snooze download BAAI/bge-small-en-v1.5 \
  --include "*.safetensors,config.json,tokenizer.json"
```

Conversely, you can exclude large unneeded files:

```bash
snooze download openai-community/gpt2 \
  --exclude "*.bin,*.h5,*.msgpack,*.ot"
```

---

## 3. Customizing Destination Folder & Directory

By default, models are placed in `./models/<repo-name>`. You can customize both the parent models directory and the subfolder name:

```bash
# Save into a specific models root directory:
snooze download sentence-transformers/all-MiniLM-L6-v2 \
  --models-dir /opt/ml/models

# Save with a custom subfolder name:
snooze download sentence-transformers/all-MiniLM-L6-v2 \
  --name minilm-embedder
```

You can also set the default models directory using the `SNOOZE_MODELS_DIR` environment variable:

```bash
export SNOOZE_MODELS_DIR="/opt/ml/models"
snooze download sentence-transformers/all-MiniLM-L6-v2
```

---

## 4. Downloading Private or Gated Models

For models that require authentication (such as gated models or private team repositories), supply your Hugging Face User Access Token:

```bash
# Via CLI flag:
snooze download meta-llama/Llama-2-7b-chat-hf \
  --token "hf_YourSecretTokenHere"

# Or via the HF_TOKEN environment variable (recommended):
export HF_TOKEN="hf_YourSecretTokenHere"
snooze download meta-llama/Llama-2-7b-chat-hf
```

---

## 5. Specific Git Revisions or Branches

If you want to pin a model to a specific Git commit SHA, branch, or release tag:

```bash
snooze download sentence-transformers/all-MiniLM-L6-v2 \
  --revision "refs/pr/5"
```

---

## 6. Disabling Automatic `models.toml` Initialization

If you are scripting automated model updates and do not want SnoozeML to modify `models.toml`, pass `--no-init`:

```bash
snooze download sentence-transformers/all-MiniLM-L6-v2 --no-init
```

You can always manually inspect or generate the configuration later using:
```bash
snooze models init ./models
snooze models scan ./models
```

---

## Recommended Checkpoints to Try

Here are several popular, lightweight checkpoints that run efficiently out-of-the-box on CPU and GPU with SnoozeML:

| Model ID | Primary Use Case | Format | Download Command |
| :--- | :--- | :--- | :--- |
| `sentence-transformers/all-MiniLM-L6-v2` | Fast 384-d semantic embeddings | ONNX / Safetensors | `snooze download sentence-transformers/all-MiniLM-L6-v2 --include "*.onnx,config.json"` |
| `BAAI/bge-small-en-v1.5` | High-accuracy retrieval embeddings | Safetensors | `snooze download BAAI/bge-small-en-v1.5 --include "*.safetensors,config.json"` |
| `distilbert/distilbert-base-uncased` | Fast text classification | Safetensors | `snooze download distilbert/distilbert-base-uncased --include "*.safetensors,config.json"` |
| `openai/clip-vit-base-patch32` | Multimodal text & image embeddings | Safetensors | `snooze download openai/clip-vit-base-patch32 --include "*.safetensors,config.json"` |