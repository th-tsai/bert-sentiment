import argparse
import sys
from pathlib import Path

# Package import is first on purpose: its __init__.py sets HF env vars that must
# take effect before huggingface_hub/transformers initialize below.
from src.bert_sentiment.config import MAX_LENGTH, OUTPUT_DIR
from src.bert_sentiment.data import LABELS

import torch  # noqa: E402
from transformers import (  # noqa: E402
    AutoModelForSequenceClassification,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)


def load_model() -> tuple[PreTrainedModel, PreTrainedTokenizerBase]:
    path = Path(OUTPUT_DIR)
    if not path.exists():
        print(
            f"Error: checkpoint not found at '{OUTPUT_DIR}'. "
            "Run `uv run python main.py` first.",
            file=sys.stderr,
        )
        sys.exit(1)

    model = AutoModelForSequenceClassification.from_pretrained(path)
    tokenizer = AutoTokenizer.from_pretrained(path)

    # Backfill semantic labels for older checkpoints saved without id2label/label2id.
    if all(str(v).startswith("LABEL_") for v in model.config.id2label.values()):
        model.config.id2label = LABELS
        model.config.label2id = {v: k for k, v in LABELS.items()}

    if torch.cuda.is_available():
        model.to("cuda")
    model.eval()
    return model, tokenizer


def predict(
    text: str, model: PreTrainedModel, tokenizer: PreTrainedTokenizerBase
) -> dict[str, object]:
    inputs = tokenizer(text, truncation=True, max_length=MAX_LENGTH, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    with torch.no_grad():
        logits = model(**inputs).logits
    probs = torch.softmax(logits, dim=-1).squeeze()
    label_id = int(probs.argmax().item())
    return {"label": model.config.id2label[label_id], "score": float(probs[label_id].item())}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Predict sentiment of text using the fine-tuned BERT model"
    )
    parser.add_argument("text", nargs="?", help="Text to classify (omit for interactive mode)")
    args = parser.parse_args()

    model, tokenizer = load_model()

    if args.text:
        result = predict(args.text, model, tokenizer)
        print(f"{result['label']} ({result['score']:.1%})")
        return

    print("Interactive mode — enter text or 'quit' to exit.\n")
    while True:
        try:
            text = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not text:
            continue
        if text.lower() in ("quit", "exit", "q"):
            break
        result = predict(text, model, tokenizer)
        print(f"  {result['label']} ({result['score']:.1%})\n")


if __name__ == "__main__":
    main()
