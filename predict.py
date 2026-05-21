import argparse
import sys
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, PreTrainedModel, PreTrainedTokenizerBase

from config import MAX_LENGTH, OUTPUT_DIR

LABELS = {0: "NEGATIVE", 1: "POSITIVE"}


def load_model(checkpoint_dir: str = OUTPUT_DIR) -> tuple[PreTrainedModel, PreTrainedTokenizerBase]:
    path = Path(checkpoint_dir)
    if not path.exists():
        print(f"Error: checkpoint not found at '{checkpoint_dir}'. Run training first.", file=sys.stderr)
        sys.exit(1)
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir)
    model = AutoModelForSequenceClassification.from_pretrained(checkpoint_dir)
    model.eval()
    return model, tokenizer


def predict(text: str, model: PreTrainedModel, tokenizer: PreTrainedTokenizerBase) -> dict[str, object]:
    inputs = tokenizer(text, truncation=True, max_length=MAX_LENGTH, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    with torch.no_grad():
        logits = model(**inputs).logits
    probs = torch.softmax(logits, dim=-1).squeeze()
    label_id = int(probs.argmax().item())
    return {"label": LABELS[label_id], "score": float(probs[label_id].item())}


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict sentiment of text using a fine-tuned BERT model")
    parser.add_argument("text", nargs="?", help="Text to classify (omit for interactive mode)")
    parser.add_argument("--checkpoint", default=OUTPUT_DIR, metavar="DIR", help="Path to model checkpoint")
    args = parser.parse_args()

    model, tokenizer = load_model(args.checkpoint)

    if args.text:
        result = predict(args.text, model, tokenizer)
        print(f"{result['label']} ({result['score']:.1%})")
    else:
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
