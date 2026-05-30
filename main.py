import json
import time
from pathlib import Path

from src.bert_sentiment.data import load_and_tokenize
from src.bert_sentiment.model import get_model
from src.bert_sentiment.train import get_trainer
from config import OUTPUT_DIR


def main() -> None:
    print("Loading and tokenizing SST-2...")
    tokenized, tokenizer = load_and_tokenize()

    train_dataset = tokenized["train"]
    eval_dataset = tokenized["validation"]

    print("Loading model...")
    model = get_model()

    trainer = get_trainer(model, tokenizer, train_dataset, eval_dataset)

    print("Training...")
    t0 = time.time()
    trainer.train()
    elapsed = time.time() - t0

    minutes, seconds = divmod(int(elapsed), 60)
    print(f"Training complete in {minutes}m {seconds}s")

    out = Path(OUTPUT_DIR)
    out.mkdir(exist_ok=True)

    history_path = out / "training_history.json"
    with open(history_path, "w") as f:
        json.dump(trainer.state.log_history, f, indent=2)

    final_eval = next(
        (e for e in reversed(trainer.state.log_history) if "eval_accuracy" in e), {}
    )
    results = {
        "val_accuracy": final_eval.get("eval_accuracy"),
        "training_time_seconds": round(elapsed, 1),
        "training_time_human": f"{minutes}m {seconds}s",
    }
    with open(out / "results.json", "w") as f:
        json.dump(results, f, indent=2)

    trainer.save_model(OUTPUT_DIR)

    print(f"Val accuracy: {results['val_accuracy']:.4f}")
    print(f"Results saved to {out}/results.json")

    from src.bert_sentiment.plot import plot_training_curves
    plot_training_curves(str(history_path))

    print(f"Done. Best model saved to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
