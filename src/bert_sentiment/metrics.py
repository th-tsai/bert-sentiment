import numpy as np
from transformers import EvalPrediction


def compute_metrics(eval_pred: EvalPrediction) -> dict[str, float]:
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return {"accuracy": float(np.mean(predictions == labels))}
