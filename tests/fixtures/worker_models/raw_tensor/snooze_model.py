class Model:
    def load(self, model_dir: str, device: str) -> None:
        pass

    def predict(self, inputs):
        return {"output": inputs["input"]}
