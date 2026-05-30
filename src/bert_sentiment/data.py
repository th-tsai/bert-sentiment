from datasets import DatasetDict, load_dataset
from transformers import AutoTokenizer, PreTrainedTokenizerBase

from .config import MAX_LENGTH, MODEL_NAME

LABELS = {0: "NEGATIVE", 1: "POSITIVE"}


def load_and_tokenize() -> tuple[DatasetDict, PreTrainedTokenizerBase]:
    dataset = load_dataset("glue", "sst2")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    def tokenize(batch: dict) -> dict:
        return tokenizer(batch["sentence"], truncation=True, max_length=MAX_LENGTH)

    return dataset.map(tokenize, batched=True), tokenizer
