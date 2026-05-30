import math

from datasets import Dataset
from transformers import (
    DataCollatorWithPadding,
    EarlyStoppingCallback,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    Trainer,
    TrainingArguments,
)

from .config import (
    BATCH_SIZE,
    GRADIENT_ACCUMULATION_STEPS,
    LEARNING_RATE,
    LORA_LEARNING_RATE,
    NUM_EPOCHS,
    OUTPUT_DIR,
)
from .metrics import compute_metrics


def get_trainer(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    train_dataset: Dataset,
    eval_dataset: Dataset,
    use_lora: bool = False,
    output_dir: str | None = None,
) -> Trainer:
    steps_per_epoch = math.ceil(len(train_dataset) / (BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS))
    total_steps = steps_per_epoch * NUM_EPOCHS
    warmup_steps = int(0.1 * total_steps)
    lr = LORA_LEARNING_RATE if use_lora else LEARNING_RATE
    args = TrainingArguments(
        output_dir=output_dir or OUTPUT_DIR,
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        learning_rate=lr,
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
