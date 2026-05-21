import logging

from transformers import AutoModelForSequenceClassification, PreTrainedModel
from config import MODEL_NAME


def get_model() -> PreTrainedModel:
    logging.getLogger("transformers").setLevel(logging.ERROR)
    return AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)
