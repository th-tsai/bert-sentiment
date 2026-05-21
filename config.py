MODEL_NAME = "bert-base-uncased"
MAX_LENGTH = 128
BATCH_SIZE = 32
GRADIENT_ACCUMULATION_STEPS = 1  # increase if GPU memory is tight (effective batch = BATCH_SIZE * steps)
LEARNING_RATE = 2e-5
NUM_EPOCHS = 3
OUTPUT_DIR = "output"
