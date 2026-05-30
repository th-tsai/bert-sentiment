"""Full fine-tuning of BERT on SST-2."""

import json
import time
from pathlib import Path

from .config import OUTPUT_DIR, PROJECT_ROOT
from .data import load_and_tokenize
from .model import get_model
from .plot import plot_training_curves
from .train import get_trainer
from .utils import count_trainable, last_eval_accuracy

METHOD = "full fine-tuning"


def run_full_ft() -> None:
    print(f"Method: {METHOD}")
    print("Loading and tokenizing SST-2...")
    tokenized, tokenizer = load_and_tokenize()

    print("Loading model...")
    model = get_model()
    trainable, total = count_trainable(model)
    print(f"Trainable params: {trainable:,} / {total:,} ({100 * trainable / total:.2f}%)")

    trainer = get_trainer(
        model,
        tokenizer,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"],
    )

    print("Training...")
    t0 = time.time()
    trainer.train()
    elapsed = time.time() - t0
    minutes, seconds = divmod(int(elapsed), 60)
    print(f"Training complete in {minutes}m {seconds}s")

    out = Path(OUTPUT_DIR)
    history_path = out / "training_history.json"
    with open(history_path, "w") as f:
        json.dump(trainer.state.log_history, f, indent=2)

    val_accuracy = last_eval_accuracy(trainer.state.log_history)
    results = {
        "method": METHOD,
        "trainable_params": trainable,
        "total_params": total,
        "val_accuracy": val_accuracy,
        "training_time_seconds": round(elapsed, 1),
    }
    with open(out / "results.json", "w") as f:
        json.dump(results, f, indent=2)

    trainer.save_model(str(out))
    figures_dir = PROJECT_ROOT / "figures"
    figures_dir.mkdir(exist_ok=True)
    plot_training_curves(str(history_path), str(figures_dir / "training_curves.png"))

    print(f"Val accuracy: {val_accuracy:.4f}")
    print(f"Best model saved to {out}/")
