# Native Safetensors Support via Candle

SnoozeML provides direct, native loading of Hugging Face Safetensors model directories for small embedding and vision models using Hugging Face [Candle](https://github.com/huggingface/candle) and `candle-transformers`, without requiring prior ONNX conversion or intermediate runtime dependencies.

---

## 1. Supported Architectures & Hugging Face Identifiers

When loading a Safetensors model, SnoozeML discovers and parses `<model_dir>/config.json`. The loader resolves the `model_type` field (or the primary entry in `architectures[0]`) against the following supported architectures:

| Category | Hugging Face `model_type` / Identifier | `candle-transformers` Module | Target / Validated Checkpoints |
| --- | --- | --- | --- |
| **Standard BERT** | `"bert"`, `"roberta"` | `candle_transformers::models::bert` | `sentence-transformers/all-MiniLM-L6-v2`<br>`BAAI/bge-small-en-v1.5`<br>`BAAI/bge-base-en-v1.5`<br>`intfloat/e5-small-v2`<br>`intfloat/e5-base-v2` |
| **Jina BERT** | `"jina-bert"`, `"jina_bert"` | `candle_transformers::models::jina_bert` | `jinaai/jina-embeddings-v2-base-en`<br>`jinaai/jina-embeddings-v2-small-en` |
| **Nomic BERT** | `"nomic_bert"` | `candle_transformers::models::nomic_bert` | `nomic-ai/nomic-embed-text-v1`<br>`nomic-ai/nomic-embed-text-v1.5` |
| **DistilBERT** | `"distilbert"` | `candle_transformers::models::distilbert` | `distilbert/distilbert-base-uncased` |
| **CLIP (Text & Vision)** | `"clip"`, `"clip_text_model"`, `"clip_vision_model"` | `candle_transformers::models::clip` | `openai/clip-vit-base-patch32`<br>`openai/clip-vit-large-patch14` |
| **Vision Transformer (ViT)** | `"vit"` | `candle_transformers::models::vit` | `google/vit-base-patch16-224` |
| **DINOv2** | `"dinov2"` | `candle_transformers::models::dinov2` | `facebook/dinov2-small`<br>`facebook/dinov2-base` |

---

## 2. Dynamic Model Loading Workflow

Loading is managed by `snooze_ml::backend::local_loader::LocalModelLoader`:

1. **Path Resolution**: Accepts either a directory path containing `config.json` and `model.safetensors` or a direct `.safetensors` file path with companion configuration.
2. **Configuration Discovery**: Reads and parses `<dir>/config.json`.
3. **Architecture Resolution**: Evaluates `model_type` and `architectures` to verify compatibility.
4. **Weight Mapping**: Maps weights directly using memory-mapped I/O via `candle_nn::VarBuilder::from_mmaped_safetensors`.
5. **Tokenizer Discovery**: Automatically detects companion `tokenizer.json` if present for integrated text encoding.
6. **Graceful Error Handling**: If an unsupported architecture is supplied (e.g. `t5`, `llama`, `chronos-bolt`), the loader returns a dedicated error variant:
   ```text
   InferenceError::UnsupportedSafetensorsArchitecture("t5")
   ```
   with an actionable recommendation to export and serve the model in ONNX format via `optimum-cli export onnx`.

---

## 3. Text Embedding Post-Processing Pipeline

For text embedding models (BERT, RoBERTa, Jina BERT, Nomic BERT, DistilBERT), SnoozeML automatically executes standard post-processing:

### Mean Pooling
Tokens are averaged weighted by the sequence attention mask, ignoring padding tokens:

$$\text{embedding} = \frac{\sum (\text{token\_embeddings} \times \text{attention\_mask})}{\sum \text{attention\_mask}}$$

### L2 Normalization
Embedding vectors are normalized to unit Euclidean length ($\Vert\mathbf{v}\Vert_2 = 1.0$), ensuring immediate readiness for cosine similarity search and vector indexing:

$$\mathbf{v}_{\text{norm}} = \frac{\mathbf{v}}{\Vert{}\mathbf{v}\Vert{}_2}$$

---

## 4. Execution Targets

Candle models execute on:
- Host CPU (`Device::Cpu`)
- NVIDIA CUDA GPUs (`Device::new_cuda(0)`) when compiled with the `cuda` feature flag and physical hardware is present
- Apple Silicon Metal / MLX (`Device::new_metal(0)`) on macOS

---

## 5. Configuration Example

Add the local directory to `models/models.toml`:

```toml
[[models]]
id = "all-minilm"
path = "all-MiniLM-L6-v2"
description = "Sentence Transformers MiniLM 384-dim embedder"
device = "auto"
max_batch_size = 16
max_batch_delay_ms = 10
idle_timeout_secs = 300
```
