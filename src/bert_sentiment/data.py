from datasets import DatasetDict
from transformers import AutoTokenizer, PreTrainedTokenizerBase
from config import MODEL_NAME, MAX_LENGTH


def get_tokenizer() -> PreTrainedTokenizerBase:
    return AutoTokenizer.from_pretrained(MODEL_NAME)


def load_and_tokenize() -> tuple[DatasetDict, PreTrainedTokenizerBase]:
    from datasets import load_dataset

    dataset = load_dataset("glue", "sst2")
    tokenizer = get_tokenizer()

    def tokenize(batch: dict) -> dict:
        return tokenizer(batch["sentence"], truncation=True, max_length=MAX_LENGTH)

    tokenized = dataset.map(tokenize, batched=True)
    return tokenized, tokenizer
