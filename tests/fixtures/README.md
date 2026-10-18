# SnoozeML Test Fixtures

This directory holds test fixtures used by the SnoozeML test suite.

## ONNX Model Fixture (`test_model.onnx`)

Integration tests and ONNX backend unit tests look for a test ONNX model at:
`tests/fixtures/test_model.onnx`

Alternatively, you can specify an arbitrary path using the environment variable:
`SNOOZE_TEST_ONNX_PATH=path/to/any_model.onnx`

### Behavior when absent:
If no ONNX model is found at `tests/fixtures/test_model.onnx` or `SNOOZE_TEST_ONNX_PATH`, all tests that depend on a physical ONNX file are **gracefully skipped** with an informative message.

### Providing a model:
You can drop any valid `.onnx` model (e.g., an Identity model, MNIST, MobileNet, or ResNet) here named `test_model.onnx`.

