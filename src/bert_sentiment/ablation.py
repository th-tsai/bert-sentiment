"""LoRA rank ablation on SST-2.

Sweeps r in [4, 8, 16] with alpha = 2r held fixed, saves per-run results,
and produces a comparison plot of val accuracy vs rank.
"""

import json
import time
from pathlib import Path

import matplotlib.pyplot as plt

from .config import OUTPUT_DIR
from .data import load_and_tokenize
from .model import get_model
from .train import get_trainer
from .utils import count_trainable, last_eval_accuracy

RANKS = [4, 8, 16]


def _run_one(tokenized, tokenizer, r: int, alpha: int, output_dir: str) -> dict:
    model = get_model(use_lora=True, lora_r=r, lora_alpha=alpha)
    trainable, _ = count_trainable(model)
    trainer = get_trainer(
        model,
        tokenizer,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"],
        use_lora=True,
        output_dir=output_dir,
    )
    t0 = time.time()
    trainer.train()
    elapsed = time.time() - t0
    trainer.save_model(output_dir)
    return {
        "lora_r": r,
        "lora_alpha": alpha,
        "trainable_params": trainable,
        "val_accuracy": last_eval_accuracy(trainer.state.log_history),
        "training_time_seconds": round(elapsed, 1),
    }


def _plot(results: list[dict], output_path: Path) -> None:
    ranks = [r["lora_r"] for r in results]
    accs = [r["val_accuracy"] * 100 for r in results]
    params = [r["trainable_params"] for r in results]

    fig, ax1 = plt.subplots(figsize=(7, 4))
    ax1.plot(ranks, accs, "o-", color="#4C72B0", label="Val accuracy")
    ax1.set_xlabel("LoRA rank (r)")
    ax1.set_ylabel("Val accuracy (%)", color="#4C72B0")
    ax1.tick_params(axis="y", labelcolor="#4C72B0")
    ax1.set_xticks(ranks)
    ax1.grid(alpha=0.3)

    ax2 = ax1.twinx()
    ax2.plot(ranks, params, "s--", color="#DD8452", label="Trainable params")
    ax2.set_ylabel("Trainable params", color="#DD8452")
    ax2.tick_params(axis="y", labelcolor="#DD8452")

    plt.title("LoRA rank ablation on SST-2 (alpha = 2r)")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")


def run_ablation() -> None:
    print("Loading and tokenizing SST-2 (once)...")
    tokenized, tokenizer = load_and_tokenize()

    root = Path(OUTPUT_DIR) / "ablation_lora_rank"
    root.mkdir(parents=True, exist_ok=True)

    results = []
    for r in RANKS:
        alpha = 2 * r
        print(f"\n=== LoRA r={r}, alpha={alpha} ===")
        run_dir = root / f"r{r}"
        res = _run_one(tokenized, tokenizer, r, alpha, str(run_dir))
        results.append(res)
        with open(run_dir / "results.json", "w") as f:
            json.dump(res, f, indent=2)

    with open(root / "summary.json", "w") as f:
        json.dump(results, f, indent=2)

    plot_path = root / "ablation.png"
    _plot(results, plot_path)
    print(f"\nSaved ablation plot to {plot_path}")

    print("\nSummary:")
    for r in results:
        print(
            f"  r={r['lora_r']:>2}  alpha={r['lora_alpha']:>2}  "
            f"trainable={r['trainable_params']:>7,}  val_acc={r['val_accuracy']:.4f}"
        )
