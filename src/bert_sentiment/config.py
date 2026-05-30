from pathlib import Path

import yaml

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
with open(_PROJECT_ROOT / "config" / "training_config.yaml") as _f:
    _cfg = yaml.safe_load(_f)

MODEL_NAME = _cfg["model_name"]
MAX_LENGTH = _cfg["max_length"]
BATCH_SIZE = _cfg["batch_size"]
GRADIENT_ACCUMULATION_STEPS = _cfg["gradient_accumulation_steps"]
LEARNING_RATE = _cfg["learning_rate"]
NUM_EPOCHS = _cfg["num_epochs"]
OUTPUT_DIR = _cfg["output_dir"]

LORA_LEARNING_RATE = _cfg["lora_learning_rate"]
LORA_R = _cfg["lora_r"]
LORA_ALPHA = _cfg["lora_alpha"]
LORA_DROPOUT = _cfg["lora_dropout"]
LORA_TARGET_MODULES = _cfg["lora_target_modules"]
