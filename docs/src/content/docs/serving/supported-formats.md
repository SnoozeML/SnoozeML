---
title: "Supported Formats & Architectures"
description: "SnoozeML features a tri-runtime backend architecture executing **Open Neural Network Exchange (ONNX)** models, native **Hugging Face Safetensors** weights, a..."
---

SnoozeML features a tri-runtime backend architecture executing **Open Neural Network Exchange (ONNX)** models, native **Hugging Face Safetensors** weights, and **Gradient Boosted Decision Trees (XGBoost / GBDT)** without requiring Python or C++ runtime dependencies.

---

## 1. ONNX Models (`.onnx`)

ONNX is the industry-standard intermediate representation for neural networks. SnoozeML executes `.onnx` models using **ONNX Runtime (`ort`)**, supporting virtually all deep learning architectures:

- **Computer Vision**: ResNet, EfficientNet, YOLO, MobileNet, Segmentation networks.
- **Natural Language Processing**: BERT, RoBERTa, T5, DeBERTa, GPT-style tokenizers.
- **Tabular & Classical ML**: XGBoost, LightGBM, Random Forests (converted via `skl2onnx` or `onnxmltools`).
- **Audio & Time-Series**: Wav2Vec, Whisper, Chronos.

### Dynamic Shapes & Batching
SnoozeML's dynamic batching engine works with any ONNX model whose leading batch dimension is dynamic (e.g. `[-1, 3, 224, 224]` or `["batch_size", 384]`). 

---

## 2. Native Safetensors via Hugging Face Candle

SnoozeML provides direct, native loading of Hugging Face Safetensors repositories using [Candle](https://github.com/huggingface/candle). You can download a Safetensors checkpoint directly from Hugging Face and serve it immediately **without prior ONNX conversion**:

```bash
snooze download BAAI/bge-small-en-v1.5 --include "*.safetensors,config.json,tokenizer.json"
```

### Supported Safetensors Architectures

When loading a Safetensors model, SnoozeML parses `<model_dir>/config.json` and automatically initializes the matching architecture:

| Architecture Family | `model_type` Identifiers | Popular Checkpoints |
| :--- | :--- | :--- |
| **Standard BERT** | `"bert"`, `"roberta"` | `sentence-transformers/all-MiniLM-L6-v2`<br>`BAAI/bge-small-en-v1.5`<br>`BAAI/bge-base-en-v1.5`<br>`intfloat/e5-small-v2` |
| **Jina BERT** | `"jina-bert"`, `"jina_bert"` | `jinaai/jina-embeddings-v2-base-en`<br>`jinaai/jina-embeddings-v2-small-en` |
| **Nomic BERT** | `"nomic_bert"` | `nomic-ai/nomic-embed-text-v1`<br>`nomic-ai/nomic-embed-text-v1.5` |
| **DistilBERT** | `"distilbert"` | `distilbert/distilbert-base-uncased` |
| **CLIP (Vision & Text)**| `"clip"`, `"clip_text_model"`, `"clip_vision_model"` | `openai/clip-vit-base-patch32`<br>`openai/clip-vit-large-patch14` |
| **Vision Transformer** | `"vit"` | `google/vit-base-patch16-224` |
| **DINOv2** | `"dinov2"` | `facebook/dinov2-small`<br>`facebook/dinov2-base` |

---

## 3. Built-in Text Embedding Pipeline

For text embedding models (BERT, RoBERTa, Jina, Nomic, DistilBERT), SnoozeML automatically executes standard vector post-processing:

### 1. Attention-Weighted Mean Pooling
Token embeddings are averaged across the sequence length, weighting tokens by the input attention mask and ignoring padding tokens:

$$\mathbf{e} = \frac{\sum_{i=1}^{L} (\mathbf{h}_i \cdot \mathbf{m}_i)}{\sum_{i=1}^{L} \mathbf{m}_i}$$

### 2. L2 Unit Normalization
The resulting vector is normalized to unit Euclidean length ($\Vert\mathbf{e}\Vert_2 = 1.0$):

$$\mathbf{e}_{\text{norm}} = \frac{\mathbf{e}}{\Vert{}\mathbf{e}\Vert{}_2}$$

This ensures the returned embeddings are immediately ready for cosine similarity calculations, vector databases (Qdrant, Milvus, pgvector), and retrieval pipelines.

---

## 4. Pure-Rust Tree Models via XGBoost JSON (`hessboost`)

SnoozeML provides native, zero-FFI execution of Gradient Boosted Decision Trees trained via XGBoost using [`hessboost`](https://crates.io/crates/hessboost) ([GitHub: brndnmtthws/hessboost](https://github.com/brndnmtthws/hessboost)). 

### Supported Tree Formats
- **XGBoost JSON**: Files ending in `.xgb.json`, `.tree.json`, or `.json` exported via Python `booster.save_model("model.json")`.
- **Hessboost Native**: Binary format `.bin` or native JSON.

### Training & Serving Tree Models
Train a tree model with the included Python tool and serve it instantly:

```bash
# Train on Iris dataset and export to models/
python tools/train_tree.py --dataset iris --name iris_model

# Run with SnoozeML:
snooze serve --models-dir ./models
```

Supported objectives include regression (`reg:squarederror`), binary classification (`binary:logistic`), and multiclass probabilities (`multi:softprob`).

---

## 5. Handling Unsupported Architectures

If you attempt to load a Safetensors architecture that is not yet natively supported in Candle (such as `t5` or large autoregressive LLMs like `llama`), SnoozeML catches this safely:

```json
{
  "error": "Unsupported Safetensors architecture: 'llama'. Export to ONNX via 'optimum-cli export onnx --model meta-llama/Llama-2-7b-chat-hf ...' to serve with SnoozeML."
}
```

The server does not crash, and other running models are completely unaffected.

---

## 6. Custom Python Worker Models (`worker-backend`)

For models that cannot be exported to ONNX or require specialized third-party Python packages (such as `chronos-forecasting` or custom PyTorch training checkpoints), SnoozeML provides a process-isolated [Custom Worker Backend](/SnoozeML/serving/custom-models/).

:::caution
**Use Custom Workers as a Last Resort**
Custom Python workers provide complete architectural freedom, but they are **inherently slower** and have **cold starts in the seconds magnitude (2s–10s+)** due to child process spawning and framework initialization.

Always prioritize native backends (`.onnx` via ONNX Runtime, Safetensors via Candle, or `.xgb.json` via Hessboost) whenever possible. Use custom workers only when native export paths are unavailable.
:::
