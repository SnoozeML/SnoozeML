---
title: "Sending Inference Requests (REST)"
description: "Inference requests are sent via HTTP `POST` to `/v1/models/{model_id}/infer`. SnoozeML uses a standard JSON payload format supporting single and multi-tensor..."
---

Inference requests are sent via HTTP `POST` to `/v1/models/{model_id}/infer`. SnoozeML uses a standard JSON payload format supporting single and multi-tensor inputs of arbitrary dimensionality.

:::tip
**Looking for ultra-low latency or binary tensor streaming?**
SnoozeML also provides a high-performance **gRPC interface** (`snoozeml.v1.InferenceService`) on port `50051`. gRPC avoids JSON serialization overhead and transfers raw byte buffers. See the [gRPC Inference Guide](/SnoozeML/client/grpc-inference/).
:::

---

## 1. Request & Response JSON Schema

### Request Payload (`POST /v1/models/{model_id}/infer`)
```json
{
  "inputs": [
    {
      "name": "input_ids",
      "shape": [1, 4],
      "data": [101.0, 2054.0, 2003.0, 102.0]
    }
  ]
}
```

- `name` (string): Input tensor identifier expected by the model.
- `shape` (array of integers): Dimensions of the tensor (e.g. `[1, 4]` or `[1, 3, 224, 224]`).
- `data` (array of numbers): Flattened 1D array of float32 values in standard row-major (C-contiguous) order. Total elements must equal $\prod \text{shape}$.

### Successful Response (`200 OK`)
```json
{
  "model_id": "bert-base",
  "outputs": [
    {
      "name": "output",
      "shape": [1, 4],
      "data": [0.12, 0.45, 0.88, 0.03]
    }
  ],
  "duration_ms": 2.45
}
```

- `model_id` (string): Confirms the target model that processed the request.
- `outputs` (array of tensor objects): Output tensors returned by the model.
- `duration_ms` (float): Wall-clock forward pass duration in milliseconds.

---

## 2. Multi-Tensor Input Example

Many transformer models (such as BERT or RoBERTa) require multiple input tensors: `input_ids` and `attention_mask`:

```json
{
  "inputs": [
    {
      "name": "input_ids",
      "shape": [1, 4],
      "data": [101.0, 7592.0, 1010.0, 102.0]
    },
    {
      "name": "attention_mask",
      "shape": [1, 4],
      "data": [1.0, 1.0, 1.0, 1.0]
    }
  ]
}
```

---

## 3. Code Examples in Different Languages

### cURL
```bash
curl -X POST http://127.0.0.1:8080/v1/models/demo-model/infer \
  -H "Content-Type: application/json" \
  -H "X-API-Key: sk-inference-key" \
  -d '{
    "inputs": [
      {
        "name": "input",
        "shape": [1, 3],
        "data": [1.0, 2.0, 3.0]
      }
    ]
  }'
```

### Python (using `requests` or `httpx`)
```python
import requests
import numpy as np

# Prepare input data as a numpy array
sample_tensor = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)

payload = {
    "inputs": [
        {
            "name": "input",
            "shape": list(sample_tensor.shape),
            "data": sample_tensor.flatten().tolist()
        }
    ]
}

headers = {
    "Content-Type": "application/json",
    "X-API-Key": "sk-inference-key"
}

response = requests.post(
    "http://127.0.0.1:8080/v1/models/demo-model/infer",
    json=payload,
    headers=headers
)
response.raise_for_status()
result = response.json()

output_tensor = result["outputs"][0]
output_array = np.array(output_tensor["data"]).reshape(output_tensor["shape"])
print(f"Inference took {result['duration_ms']} ms")
print("Output shape:", output_array.shape)
print("Output values:", output_array)
```

### JavaScript / TypeScript (`fetch`)
```typescript
async function queryModel() {
  const payload = {
    inputs: [
      {
        name: "input",
        shape: [1, 3],
        data: [1.0, 2.0, 3.0]
      }
    ]
  };

  const res = await fetch("http://127.0.0.1:8080/v1/models/demo-model/infer", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      "X-API-Key": "sk-inference-key"
    },
    body: JSON.stringify(payload)
  });

  if (!res.ok) {
    throw new Error(`Inference failed: ${res.statusText}`);
  }

  const data = await res.json();
  console.log("Duration (ms):", data.duration_ms);
  console.log("Outputs:", data.outputs);
}
```

### Rust (`reqwest`)
```rust,ignore
use reqwest::Client;
use serde_json::json;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let client = Client::new();
    let body = json!({
        "inputs": [{
            "name": "input",
            "shape": [1, 3],
            "data": [1.0, 2.0, 3.0]
        }]
    });

    let resp = client
        .post("http://127.0.0.1:8080/v1/models/demo-model/infer")
        .header("X-API-Key", "sk-inference-key")
        .json(&body)
        .send()
        .await?
        .error_for_status()?
        .json::<serde_json::Value>()
        .await?;

    println!("Inference response: {:#?}", resp);
    Ok(())
}
```