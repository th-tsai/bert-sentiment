import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


def plot_training_curves(history_path: str, output_path: str) -> None:
    Path(output_path).parent.mkdir(exist_ok=True)

    with open(history_path) as f:
        history: list[dict] = json.load(f)

    train_logs = [e for e in history if "loss" in e and "eval_loss" not in e]
    eval_logs = [e for e in history if "eval_accuracy" in e]

    train_steps = [e["step"] for e in train_logs]
    train_loss = [e["loss"] for e in train_logs]
    eval_steps = [e["step"] for e in eval_logs]
    eval_loss = [e["eval_loss"] for e in eval_logs]
    eval_acc = [e["eval_accuracy"] * 100 for e in eval_logs]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))

    ax1.plot(train_steps, train_loss, label="Train", color="#4C72B0")
    ax1.plot(eval_steps, eval_loss, "o-", label="Val", color="#DD8452")
    ax1.set_xlabel("Step")
    ax1.set_ylabel("Loss")
    ax1.set_title("Loss")
    ax1.legend()
    ax1.grid(alpha=0.3)

    ax2.plot(eval_steps, eval_acc, "o-", color="#55A868")
    ax2.set_xlabel("Step")
    ax2.set_ylabel("Accuracy (%)")
    ax2.set_title("Validation Accuracy")
    ax2.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.1f"))
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
