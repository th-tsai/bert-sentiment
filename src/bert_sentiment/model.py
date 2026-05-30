from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoModelForSequenceClassification, PreTrainedModel

from .config import (
    LORA_ALPHA,
    LORA_DROPOUT,
    LORA_R,
    LORA_TARGET_MODULES,
    MODEL_NAME,
)
from .data import LABELS


def get_model(
    use_lora: bool = False,
    lora_r: int | None = None,
    lora_alpha: int | None = None,
) -> PreTrainedModel:
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(LABELS),
        id2label=LABELS,
        label2id={v: k for k, v in LABELS.items()},
    )
    if use_lora:
        lora_config = LoraConfig(
            task_type=TaskType.SEQ_CLS,
            r=lora_r if lora_r is not None else LORA_R,
            lora_alpha=lora_alpha if lora_alpha is not None else LORA_ALPHA,
            lora_dropout=LORA_DROPOUT,
            target_modules=LORA_TARGET_MODULES,
            bias="none",
        )
        model = get_peft_model(model, lora_config)
    return model
