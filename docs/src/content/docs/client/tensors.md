---
title: "Input & Output Tensor Formats"
description: "SnoozeML uses standard JSON-serializable tensor payloads. This page explains how multi-dimensional arrays are represented, validated, and converted across di..."
---

SnoozeML uses standard JSON-serializable tensor payloads. This page explains how multi-dimensional arrays are represented, validated, and converted across different programming languages.

---

## 1. Tensor Structure & Memory Layout

A `TensorPayload` consists of three required fields:

```json
{
  "name": "input_ids",
  "shape": [1, 4],
  "data": [101.0, 2054.0, 2003.0, 102.0]
}
```

1. **`name`**: The node name corresponding to the input or output port of the model graph (e.g. `"images"`, `"input_ids"`, `"dense_input"`).
2. **`shape`**: An array of non-negative integers specifying tensor dimensions.
3. **`data`**: A flattened 1D array of single-precision floating-point numbers (`float32`) serialized in standard **row-major (C-contiguous)** order.

### Validation Invariant: Length Check
The total number of elements in the `data` array must strictly equal the product of the shape dimensions:

$$\text{len}(\text{data}) = \prod_{i=0}^{k-1} \text{shape}[i]$$

If this condition is violated, SnoozeML rejects the request with an HTTP `400 Bad Request` before attempting model execution.

---

## 2. Common Dimensionality Examples

### 1D Vector (Embedding or Bias)
- **Shape**: `[4]`
- **Data**: `[0.1, 0.2, 0.3, 0.4]`
- **Length**: $4 = 4$

### 2D Matrix (Batch of Token IDs or Tabular Features)
- **Shape**: `[2, 3]` (Batch size = 2, Feature length = 3)
- **Data**: `[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]`
- **Length**: $2 \times 3 = 6$
- **Layout**:
  - Sample 0: `[1.0, 2.0, 3.0]`
  - Sample 1: `[4.0, 5.0, 6.0]`

### 4D Vision Tensor (`[Batch, Channels, Height, Width]`)
- **Shape**: `[1, 3, 224, 224]` (Batch size = 1, 3 RGB channels, 224x224 pixels)
- **Data**: Array of $1 \times 3 \times 224 \times 224 = 150,528$ normalized float values.

---

## 3. Python & NumPy Conversion Helpers

Working with NumPy arrays makes encoding and decoding tensors trivial:

### Encoding: NumPy Array to JSON Payload
```python
import numpy as np

def tensor_to_payload(name: str, array: np.ndarray) -> dict:
    """Converts a NumPy array into a SnoozeML TensorPayload dict."""
    contiguous_arr = np.ascontiguousarray(array, dtype=np.float32)
    return {
        "name": name,
        "shape": list(contiguous_arr.shape),
        "data": contiguous_arr.flatten().tolist()
    }

# Example usage:
my_image = np.random.randn(1, 3, 224, 224).astype(np.float32)
payload = tensor_to_payload("input", my_image)
```

### Decoding: JSON Response to NumPy Array
```python
def payload_to_tensor(payload: dict) -> np.ndarray:
    """Reconstructs a NumPy array from a SnoozeML output payload."""
    shape = payload["shape"]
    data = payload["data"]
    return np.array(data, dtype=np.float32).reshape(shape)

# Example usage:
response_json = response.json()
output_array = payload_to_tensor(response_json["outputs"][0])
```

---

## 4. Binary gRPC Tensor Serialization (`TensorData`)

In addition to JSON arrays over REST, SnoozeML supports binary tensor serialization via gRPC using the `TensorData` message:

```protobuf
message TensorData {
  string name = 1;
  repeated int64 shape = 2;
  DataType data_type = 3;  // DATA_TYPE_FP32
  bytes raw_data = 4;      // Contiguous little-endian bytes
}
```

### Memory Layout & Invariants

- **Contiguous Buffer**: Floats are packed sequentially in little-endian IEEE-754 format.
- **Byte Length Assertion**:
  $$\text{len}(\text{raw\_data}) = \left(\prod_{i=0}^{N-1} \text{shape}[i]\right) \times 4$$
- If the byte buffer size does not match this invariant, SnoozeML immediately returns `Status::invalid_argument`.

### Python NumPy Conversion
```python
import numpy as np
from snoozeml.v1 import inference_pb2

# Encoding NumPy array to TensorData:
def numpy_to_tensor_data(name: str, arr: np.ndarray) -> inference_pb2.TensorData:
    contiguous = np.ascontiguousarray(arr, dtype=np.float32)
    return inference_pb2.TensorData(
        name=name,
        shape=list(contiguous.shape),
        data_type=inference_pb2.DATA_TYPE_FP32,
        raw_data=contiguous.tobytes() # Direct memory copy
    )

# Decoding TensorData to NumPy array:
def tensor_data_to_numpy(tensor: inference_pb2.TensorData) -> np.ndarray:
    return np.frombuffer(tensor.raw_data, dtype=np.float32).reshape(list(tensor.shape))
```

### Rust Conversion (`bytemuck`)
```rust,ignore
use snooze_ml::server::grpc::proto::{DataType, TensorData};

// Encoding floats to TensorData:
fn floats_to_tensor_data(name: String, shape: Vec<i64>, floats: &[f32]) -> TensorData {
    TensorData {
        name,
        shape,
        data_type: DataType::Fp32 as i32,
        raw_data: bytemuck::cast_slice(floats).to_vec(),
    }
}

// Decoding TensorData to slice of floats:
fn tensor_data_to_floats(tensor: &TensorData) -> &[f32] {
    bytemuck::cast_slice(&tensor.raw_data)
}
```

---

## 5. Discovering Model Node Names

To determine the exact input and output node names expected by an ONNX model:

### Using the Python `onnx` Library
```python
import onnx

model = onnx.load("models/my_model.onnx")
inputs = [node.name for node in model.graph.input]
outputs = [node.name for node in model.graph.output]

print("Input nodes:", inputs)
print("Output nodes:", outputs)
```

### Visual Inspection via Netron
You can open any `.onnx` or `.safetensors` file in [Netron](https://netron.app/) to interactively inspect node names, dynamic axis definitions, and layer shapes.