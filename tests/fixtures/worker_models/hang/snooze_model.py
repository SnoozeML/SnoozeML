import time

class Model:
    def load(self, model_dir: str, device: str) -> None:
        pass

    def predict(self, inputs):
        time.sleep(10)
        return {"output": inputs["input"]}
