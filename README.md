# BERT Sentiment Fine-tuning Practice

Fine-tuning `bert-base-uncased` on SST-2 for binary sentiment classification. Reaches **93.3% validation accuracy** — 0.2 points below the original paper — in **5 minutes** on a consumer GPU.

![Python](https://img.shields.io/badge/python-3.13-3776AB?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.11-EE4C2C?logo=pytorch&logoColor=white)
![Accuracy](https://img.shields.io/badge/SST--2%20accuracy-93.3%25-brightgreen)
[![CI](https://github.com/USERNAME/bert-sentiment/actions/workflows/ci.yml/badge.svg)](https://github.com/USERNAME/bert-sentiment/actions/workflows/ci.yml)

## Results

| Model                                                                       | Val Accuracy | Training Time | Hardware           |
| --------------------------------------------------------------------------- | ------------ | ------------- | ------------------ |
| BERT-base-uncased (ours)                                                    | **93.3%**    | 5m 13s        | RTX 5070 Ti Laptop |
| BERT-base-uncased ([Devlin et al., 2019](https://arxiv.org/abs/1810.04805)) | 93.5%        | —             | TPU v3             |

Accuracy peaked at epoch 3 (93.35%), with the Trainer automatically retaining the best checkpoint via `load_best_model_at_end`.

![Training curves](figures/training_curves.png)

## Demo

```bash
uv run python predict.py "This movie was absolutely fantastic!"
# POSITIVE (99.9%)

uv run python predict.py "Boring, predictable, and way too long."
# NEGATIVE (100.0%)
```

Interactive mode:

```bash
uv run python predict.py
# > A masterpiece of modern cinema.
#   POSITIVE (100.0%)
# > I've seen better films at a school play.
#   NEGATIVE (99.2%)
```

## Setup

```bash
git clone https://github.com/USERNAME/bert-sentiment
cd bert-sentiment
uv sync
```

Requires Python 3.13, CUDA 12.8, and [uv](https://docs.astral.sh/uv/).

## Training

```bash
uv run python main.py   # or: make train
```

Trains for 3 epochs, saves the best checkpoint to `output/`, and automatically writes `output/results.json` and `assets/training_curves.png`.

## Inference

```bash
# Single sentence
uv run python predict.py "Your text here"

# Interactive REPL
uv run python predict.py

# Custom checkpoint
uv run python predict.py --checkpoint output/checkpoint-4210 "Your text"
```

## Configuration

All hyperparameters live in `config.py`:

| Parameter                     | Default             | Description                                           |
| ----------------------------- | ------------------- | ----------------------------------------------------- |
| `MODEL_NAME`                  | `bert-base-uncased` | Pretrained model                                      |
| `MAX_LENGTH`                  | `128`               | Token sequence length                                 |
| `BATCH_SIZE`                  | `32`                | Per-device batch size                                 |
| `GRADIENT_ACCUMULATION_STEPS` | `1`                 | Raise to simulate larger batches when VRAM is limited |
| `LEARNING_RATE`               | `2e-5`              | AdamW learning rate                                   |
| `NUM_EPOCHS`                  | `3`                 | Training epochs                                       |
| `OUTPUT_DIR`                  | `output`            | Checkpoint directory                                  |

## Project Structure

```
bert-sentiment/
├── config.py        # hyperparameters
├── main.py          # training entry point
├── predict.py       # inference on raw text
└── src/
    ├── data.py      # SST-2 loading and tokenization
    ├── model.py     # model setup
    ├── train.py     # Trainer config and metrics
    └── plot.py      # training curve generation
```

## Dataset

[SST-2](https://huggingface.co/datasets/nyu-mll/glue/viewer/sst2) (Stanford Sentiment Treebank) is a binary sentence-level sentiment benchmark from the GLUE suite, loaded automatically from the Hugging Face Hub.

- Train: 67,349 examples
- Validation: 872 examples
- Labels: `0` = negative, `1` = positive
