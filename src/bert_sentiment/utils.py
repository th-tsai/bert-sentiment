from torch import nn


def count_trainable(model: nn.Module) -> tuple[int, int]:
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total


def last_eval_accuracy(log_history: list[dict]) -> float:
    for entry in reversed(log_history):
        if "eval_accuracy" in entry:
            return entry["eval_accuracy"]
    raise RuntimeError("No eval_accuracy entries found in training log_history")
