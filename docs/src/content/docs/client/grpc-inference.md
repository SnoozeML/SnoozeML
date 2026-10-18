---
title: "gRPC Inference"
description: "SnoozeML features a high-performance **gRPC serving subsystem** alongside its HTTP REST API. Built on [Tonic](https://github.com/hyperium/tonic) and [Prost](..."
---

SnoozeML features a high-performance **gRPC serving subsystem** alongside its HTTP REST API. Built on [Tonic](https://github.com/hyperium/tonic) and [Prost](https://github.com/tokio-rs/prost), the gRPC service provides ultra-low latency, binary tensor transmission, multiplexed HTTP/2 streams, and standard `grpc.health.v1` health checking.

---

## 1. Why Use gRPC for Model Inference?

While the HTTP REST API is convenient for quick tests, debugging, and web frontends, **gRPC** offers major advantages for production machine learning pipelines:

| Feature | HTTP REST (`/v1/models/:id/infer`) | gRPC (`snoozeml.v1.InferenceService/Infer`) |
| :--- | :--- | :--- |
| **Serialization** | Text JSON (`[1.0, 2.0, 3.0]`) | Binary Protocol Buffers (`bytes`) |
| **Tensor Encoding** | Decimal string representation per float | Direct IEEE-754 little-endian byte buffers |
| **Wire Overhead** | High (quotes, commas, brackets) | Minimal (compact protobuf tags) |
| **Client Serialization CPU** | High (string formatting & parsing) | Near-zero (`memcpy` / `tobytes()`) |
| **Transport** | HTTP/1.1 or HTTP/2 | Multiplexed HTTP/2 |
| **Schema Validation** | Runtime JSON validation | Compile-time Protobuf contract |
| **Health Checking** | Custom `/health/live`, `/health/ready` | Standard `grpc.health.v1.Health` |

:::tip
For large vision tensors (e.g. `[1, 3, 224, 224] = 150,528` floats) or high-frequency embedding requests, gRPC eliminates megabytes of JSON string serialization per request, cutting client and server CPU time dramatically.
:::

---

## 2. Protocol Buffer Schema Overview

The service schema is defined in `proto/snoozeml/v1/inference.proto`:

```protobuf
syntax = "proto3";

package snoozeml.v1;

// Service providing low-latency tensor inference over gRPC.
service InferenceService {
  // Executes unary inference for a registered model.
  rpc Infer(ModelInferRequest) returns (ModelInferResponse);
}

// Supported tensor element data types.
enum DataType {
  DATA_TYPE_UNSPECIFIED = 0;
  DATA_TYPE_FP32 = 1;
  DATA_TYPE_FP16 = 2;
  DATA_TYPE_BF16 = 3;
  DATA_TYPE_INT32 = 4;
  DATA_TYPE_INT64 = 5;
  DATA_TYPE_UINT8 = 6;
  DATA_TYPE_BOOL = 7;
}

// Multi-dimensional tensor payload with raw byte buffer.
message TensorData {
  string name = 1;              // Node binding name (e.g. "input_ids")
  repeated int64 shape = 2;     // Dimensions (e.g. [1, 4])
  DataType data_type = 3;       // Defaults to DATA_TYPE_FP32
  bytes raw_data = 4;           // Contiguous little-endian byte representation
}

message ModelInferRequest {
  string model_id = 1;                 // Target model identifier
  repeated TensorData inputs = 2;       // Input tensors
  optional uint64 client_timeout_ms = 3;// Optional request timeout
  map<string, string> parameters = 4;   // Routing hints or parameters
}

message ModelInferResponse {
  string model_id = 1;                 // Executed model identifier
  repeated TensorData outputs = 2;      // Resulting output tensors
  double duration_ms = 3;              // Execution latency in milliseconds
}
```

---

## 3. Binary Tensor Encoding

In `TensorData`, floating-point values are packed contiguously into `raw_data` as little-endian IEEE-754 32-bit floats (`4` bytes per element):

$$\text{raw\_data.len} = \prod \text{shape} \times 4$$

### In Rust
```rust,ignore
let floats: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
let raw_data: Vec<u8> = bytemuck::cast_slice(&floats).to_vec();
```

### In Python
```python
import numpy as np

array = np.array([[1.0, 2.0, 3.0, 4.0]], dtype=np.float32)
raw_bytes = array.tobytes()
```

---

## 4. Querying with `grpcurl`

You can use the standard [`grpcurl`](https://github.com/fullstorydev/grpcurl) CLI utility to interact with SnoozeML's gRPC endpoints:

### 1. Check Server Health
```bash
grpcurl -plaintext 127.0.0.1:50051 grpc.health.v1.Health/Check
```
Output:
```json
{
  "status": "SERVING"
}
```

### 2. Check Specific Service Health
```bash
grpcurl -plaintext \
  -d '{"service": "snoozeml.v1.InferenceService"}' \
  127.0.0.1:50051 grpc.health.v1.Health/Check
```

### 3. Send Unary Inference Request
To send a raw float tensor `[10.0, 20.0]` (Base64-encoded `AADAQAAAQEAA`):
```bash
grpcurl -plaintext \
  -proto proto/snoozeml/v1/inference.proto \
  -d '{
    "model_id": "demo-model",
    "inputs": [
      {
        "name": "input_ids",
        "shape": [1, 2],
        "data_type": "DATA_TYPE_FP32",
        "raw_data": "AADAQAAAQEAA"
      }
    ]
  }' \
  127.0.0.1:50051 snoozeml.v1.InferenceService/Infer
```

### 4. Authenticated Request via Metadata Header
```bash
grpcurl -plaintext \
  -proto proto/snoozeml/v1/inference.proto \
  -H "x-api-key: sk-infer-client-654" \
  -d '{"model_id": "demo-model", "inputs": [{"name": "input_ids", "shape": [1, 2], "raw_data": "AADAQAAAQEAA"}]}' \
  127.0.0.1:50051 snoozeml.v1.InferenceService/Infer
```

---

## 5. Client Code Examples

### Python Client (`grpcio` + NumPy)

Install dependencies:
```bash
pip install grpcio grpcio-tools numpy
```

Compile the `.proto` file to Python:
```bash
python -m grpc_tools.protoc \
  -I proto \
  --python_out=. \
  --grpc_python_out=. \
  proto/snoozeml/v1/inference.proto
```

Inference client script:
```python
import grpc
import numpy as np
from snoozeml.v1 import inference_pb2, inference_pb2_grpc

def run_grpc_inference():
    # 1. Connect to SnoozeML gRPC endpoint
    channel = grpc.insecure_channel("127.0.0.1:50051")
    client = inference_pb2_grpc.InferenceServiceStub(channel)

    # 2. Prepare input tensor using NumPy
    input_data = np.array([[10.0, 20.0, 30.0]], dtype=np.float32)
    
    tensor = inference_pb2.TensorData(
        name="input",
        shape=list(input_data.shape),
        data_type=inference_pb2.DATA_TYPE_FP32,
        raw_data=input_data.tobytes()
    )

    request = inference_pb2.ModelInferRequest(
        model_id="demo-model",
        inputs=[tensor]
    )

    # 3. Supply authentication metadata (API key or Bearer token)
    metadata = [("x-api-key", "sk-infer-client-654")]

    # 4. Execute unary call
    try:
        response = client.Infer(request, metadata=metadata)
        print(f"Model: {response.model_id} executed in {response.duration_ms:.2f} ms")

        # 5. Decode output tensors
        for out in response.outputs:
            out_shape = list(out.shape)
            out_array = np.frombuffer(out.raw_data, dtype=np.float32).reshape(out_shape)
            print(f"Output '{out.name}': shape {out_shape}, values:\n{out_array}")

    except grpc.RpcError as e:
        print(f"gRPC call failed: {e.code()} - {e.details()}")

if __name__ == "__main__":
    run_grpc_inference()
```

---

### Rust Client (`tonic` + `prost`)

Add dependencies to `Cargo.toml`:
```toml
[dependencies]
tonic = "0.12"
prost = "0.13"
tokio = { version = "1.38", features = ["full"] }
bytemuck = "1.16"
```

Inference client code:
```rust,ignore
use tonic::metadata::MetadataValue;
use tonic::transport::Channel;

pub mod snoozeml {
    tonic::include_proto!("snoozeml.v1");
}

use snoozeml::inference_service_client::InferenceServiceClient;
use snoozeml::{DataType, ModelInferRequest, TensorData};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let channel = Channel::from_static("http://127.0.0.1:50051")
        .connect()
        .await?;

    let mut client = InferenceServiceClient::new(channel);

    // Prepare input tensor: [1, 3] with floats [1.0, 2.0, 3.0]
    let input_floats: Vec<f32> = vec![1.0, 2.0, 3.0];
    let raw_bytes: Vec<u8> = bytemuck::cast_slice(&input_floats).to_vec();

    let mut request = tonic::Request::new(ModelInferRequest {
        model_id: "demo-model".to_string(),
        inputs: vec![TensorData {
            name: "input".to_string(),
            shape: vec![1, 3],
            data_type: DataType::Fp32 as i32,
            raw_data: raw_bytes,
        }],
        client_timeout_ms: Some(5000),
        parameters: Default::default(),
    });

    // Add optional authentication header
    let token = MetadataValue::from_static("sk-infer-client-654");
    request.metadata_mut().insert("x-api-key", token);

    let response = client.infer(request).await?.into_inner();
    println!("Inference took {} ms", response.duration_ms);

    for output in response.outputs {
        let floats: &[f32] = bytemuck::cast_slice(&output.raw_data);
        println!("Output '{}' shape {:?}: {:?}", output.name, output.shape, floats);
    }

    Ok(())
}
```

---

### Node.js / TypeScript Client (`@grpc/grpc-js`)

```typescript
import * as grpc from "@grpc/grpc-js";
import * as protoLoader from "@grpc/proto-loader";
import * as path from "path";

const PROTO_PATH = path.join(__dirname, "proto/snoozeml/v1/inference.proto");
const packageDefinition = protoLoader.loadSync(PROTO_PATH, {
  keepCase: true,
  longs: String,
  enums: String,
  defaults: true,
  oneofs: true,
});

const protoDescriptor = grpc.loadPackageDefinition(packageDefinition) as any;
const client = new protoDescriptor.snoozeml.v1.InferenceService(
  "127.0.0.1:50051",
  grpc.credentials.createInsecure()
);

// Prepare float buffer (4 bytes per float, little-endian)
const floats = new Float32Array([1.0, 2.0, 3.0]);
const buffer = Buffer.from(floats.buffer);

const request = {
  model_id: "demo-model",
  inputs: [
    {
      name: "input",
      shape: [1, 3],
      data_type: "DATA_TYPE_FP32",
      raw_data: buffer,
    },
  ],
};

const metadata = new grpc.Metadata();
metadata.add("x-api-key", "sk-infer-client-654");

client.Infer(request, metadata, (err: grpc.ServiceError, response: any) => {
  if (err) {
    console.error("Inference error:", err.message);
    return;
  }
  console.log(`Executed model: ${response.model_id} in ${response.duration_ms} ms`);
  
  for (const out of response.outputs) {
    const outFloats = new Float32Array(
      out.raw_data.buffer,
      out.raw_data.byteOffset,
      out.raw_data.byteLength / 4
    );
    console.log(`Output: ${out.name}, Shape: ${out.shape}, Values:`, Array.from(outFloats));
  }
});
```

---

## 6. Transparent Dynamic Batching & Scale-to-Zero

Requests submitted over gRPC follow the exact same execution pipeline as REST requests:

1. **Shared Model Registry**: Both HTTP and gRPC workers query the same `ModelRegistry`.
2. **Transparent Micro-Batching**: If multiple gRPC clients (or a mix of gRPC and HTTP clients) send concurrent requests for `demo-model`, SnoozeML's `DynamicBatcher` collates their tensors along `Axis(0)` into a single contiguous batch.
3. **Automatic Cold Rehydration**: If a requested model is currently in the `Idle` state, the gRPC handler transparently triggers asynchronous weight rehydration before executing the forward pass.
4. **Prometheus Metrics**: Successful and failed gRPC inferences increment the global `snoozeml_requests_total` counter and accumulate latency into `snoozeml_inference_latency_seconds_total`.