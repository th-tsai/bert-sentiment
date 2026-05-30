"""Baseline accuracy on SST-2 before any fine-tuning.

Two reference points:
- Untrained classifier head on pretrained BERT (random head, seed=42).
- Majority-class predictor.

Both expose how much accuracy fine-tuning actually contributes.
"""

import json
import tempfile
from pathlib import Path

import numpy as np
import torch
from transformers import DataCollatorWithPadding, Trainer, TrainingArguments

from .config import BATCH_SIZE, MODEL_NAME, OUTPUT_DIR
from .data import load_and_tokenize
from .metrics import compute_metrics
from .model import get_model


def run_baseline() -> None:
    torch.manual_seed(42)

    print("Loading and tokenizing SST-2...")
    tokenized, tokenizer = load_and_tokenize()
    val = tokenized["validation"]

    print(f"Loading {MODEL_NAME} with random classifier head...")
    model = get_model()

    with tempfile.TemporaryDirectory(prefix="baseline_") as tmpdir:
        args = TrainingArguments(
            output_dir=tmpdir,
            per_device_eval_batch_size=BATCH_SIZE,
            bf16=True,
            report_to="none",
        )
        trainer = Trainer(
            model=model,
            args=args,
            eval_dataset=val,
            processing_class=tokenizer,
            data_collator=DataCollatorWithPadding(tokenizer),
            compute_metrics=compute_metrics,
        )
        untrained_acc = trainer.evaluate()["eval_accuracy"]

    labels = np.array(val["label"])
    majority_acc = float(max(np.mean(labels == 0), np.mean(labels == 1)))

    print(f"\nUntrained classifier head val accuracy: {untrained_acc:.4f}")
    print(f"Majority class baseline:                {majority_acc:.4f}")

    out = Path(OUTPUT_DIR)
    out.mkdir(exist_ok=True)
    with open(out / "baseline.json", "w") as f:
        json.dump(
            {
                "model": MODEL_NAME,
                "untrained_head_accuracy": untrained_acc,
                "majority_class_accuracy": majority_acc,
                "seed": 42,
            },
            f,
            indent=2,
        )
    print(f"Saved to {out / 'baseline.json'}")
