import os

# Must be set before huggingface_hub is first imported.
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")

from huggingface_hub.utils import disable_progress_bars
from huggingface_hub.utils import logging as _hub_logging
from transformers import logging as _hf_logging

disable_progress_bars()
_hf_logging.set_verbosity_error()
_hub_logging.set_verbosity_error()
