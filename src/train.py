import math
import numpy as np
from datasets import Dataset
from transformers import (
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
    EvalPrediction,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)
from config import BATCH_SIZE, GRADIENT_ACCUMULATION_STEPS, LEARNING_RATE, NUM_EPOCHS, OUTPUT_DIR


def compute_metrics(eval_pred: EvalPrediction) -> dict[str, float]:
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return {"accuracy": float(np.mean(predictions == labels))}


def get_trainer(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    train_dataset: Dataset,
    eval_dataset: Dataset,
) -> Trainer:
    total_steps = math.ceil(len(train_dataset) / (BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS)) * NUM_EPOCHS
    warmup_steps = int(0.1 * total_steps)
    args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        learning_rate=LEARNING_RATE,
        weight_decay=0.01,
        warmup_steps=warmup_steps,
        eval_strategy="steps",
        eval_steps=1000,
        save_strategy="steps",
        save_steps=1000,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        bf16=True,
        logging_steps=100,
        report_to="none",
    )
    return Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer),
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
    )
