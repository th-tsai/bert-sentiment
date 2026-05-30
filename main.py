import argparse

from src.bert_sentiment.ablation import run_ablation
from src.bert_sentiment.baseline import run_baseline
from src.bert_sentiment.full_ft import run_full_ft


def main() -> None:
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--lora",
        action="store_true",
        help="Run LoRA rank ablation (r in {4, 8, 16}, alpha = 2r)",
    )
    mode.add_argument(
        "--baseline",
        action="store_true",
        help="Evaluate untrained classifier head + majority-class baseline (no training)",
    )
    args = parser.parse_args()

    if args.lora:
        run_ablation()
    elif args.baseline:
        run_baseline()
    else:
        run_full_ft()


if __name__ == "__main__":
    main()
