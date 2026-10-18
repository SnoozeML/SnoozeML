class Model:
    def load(self, model_dir: str, device: str) -> None:
        self.device = device

    def predict(self, inputs):
        return {"output": inputs["input"]}
