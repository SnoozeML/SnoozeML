class Model:
    def load(self, model_dir: str, device: str) -> None:
        pass

    def predict(self, inputs):
        a = inputs["a"]
        b = inputs["b"]
        return {"sum": a + b, "diff": a - b}
