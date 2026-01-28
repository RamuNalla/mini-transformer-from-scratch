import torch
from pathlib import Path

ROOT_DIR = Path(__file__).parent
DATA_DIR = ROOT_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODEL_DIR = ROOT_DIR / "saved_models"
CHECKPOINT_DIR = ROOT_DIR / "checkpoints"
LOGS_DIR = ROOT_DIR / "logs"
VISUALIZATION_DIR = ROOT_DIR / "visualizations"

# Create directories
for dir_path in [DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR, MODEL_DIR,
                 CHECKPOINT_DIR, LOGS_DIR, VISUALIZATION_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# Dataset source
DATASET_NAME = "conll2003"
DATASET_URL = "https://raw.githubusercontent.com/davidsbatista/NER-datasets/master/CONLL2003/"

# NER labels (BIO tagging)
NER_TAGS = [
    'O',           # Outside
    'B-PER',       # Begin Person
    'I-PER',       # Inside Person
    'B-ORG',       # Begin Organization
    'I-ORG',       # Inside Organization
    'B-LOC',       # Begin Location
    'I-LOC',       # Inside Location
    'B-MISC',      # Begin Miscellaneous
    'I-MISC'       # Inside Miscellaneous
]

TAG2IDX = {tag: idx for idx, tag in enumerate(NER_TAGS)}
IDX2TAG = {idx: tag for idx, tag in enumerate(NER_TAGS)}
NUM_LABELS = len(NER_TAGS)

# Special tokens
PAD_TOKEN = "[PAD]"
UNK_TOKEN = "[UNK]"
PAD_IDX = 0


# Model Hyperparameters

# Embedding dimensions
D_MODEL = 128              # Model dimension (embedding size)
VOCAB_SIZE = 10000         # Vocabulary size

# Transformer architecture
NUM_LAYERS = 2             # Number of Transformer encoder layers
NUM_HEADS = 4              # Number of attention heads
D_FF = 512                # Dimension of feed-forward network
DROPOUT = 0.1              # Dropout rate

# Positional encoding
MAX_SEQ_LENGTH = 128       # Maximum sequence length

# Attention
D_K = D_MODEL // NUM_HEADS  # Dimension per head (64 for 256/8)
D_V = D_MODEL // NUM_HEADS  # Value dimension per head




# Training configureation

# Training parameters
BATCH_SIZE = 16
LEARNING_RATE = 1e-4
NUM_EPOCHS = 30
WARMUP_STEPS = 500
MAX_GRAD_NORM = 1.0        # Gradient clipping

# Optimizer
WEIGHT_DECAY = 0.01
BETAS = (0.9, 0.999)
EPS = 1e-8

# Label smoothing
LABEL_SMOOTHING = 0.1

# Learning rate schedule
LR_SCHEDULE = "warmup_linear"  # Options: warmup_linear, constant

# Early stopping
EARLY_STOPPING_PATIENCE = 10



# Device COnfiguration

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_WORKERS = 0 if torch.cuda.is_available() else 0


# Logging and Checkpointing

LOG_INTERVAL = 25          # Log every N steps
EVAL_INTERVAL = 200        # Evaluate every N steps
SAVE_INTERVAL = 1          # Save checkpoint every N epochs

# Evaluation configuration

# Metrics to compute
COMPUTE_METRICS = ['precision', 'recall', 'f1']


# MODEL SAVE PATHS
MODEL_SAVE_PATH = MODEL_DIR / "transformer_ner.pth"
VOCAB_SAVE_PATH = PROCESSED_DATA_DIR / "vocab.pkl"
CONFIG_SAVE_PATH = MODEL_DIR / "config.json"


def print_config():
    """Print configuration"""
    print("=" * 80)
    print("TRANSFORMER NER CONFIGURATION")
    print("=" * 80)
    
    print(f"\n📊 DATASET:")
    print(f"  Name: {DATASET_NAME}")
    print(f"  Labels: {NUM_LABELS} ({', '.join(NER_TAGS)})")
    
    print(f"\n🏗️  MODEL:")
    print(f"  d_model: {D_MODEL}")
    print(f"  Layers: {NUM_LAYERS}")
    print(f"  Heads: {NUM_HEADS}")
    print(f"  d_k/d_v: {D_K}")
    print(f"  d_ff: {D_FF}")
    print(f"  Dropout: {DROPOUT}")
    
    print(f"\n🎯 TRAINING:")
    print(f"  Batch size: {BATCH_SIZE}")
    print(f"  Learning rate: {LEARNING_RATE}")
    print(f"  Epochs: {NUM_EPOCHS}")
    print(f"  Device: {DEVICE}")
    print(f"  Warmup steps: {WARMUP_STEPS}")
    
    print("=" * 80)


def get_config_dict():
    """Get configuration as dictionary"""
    return {
        "d_model": D_MODEL,
        "num_layers": NUM_LAYERS,
        "num_heads": NUM_HEADS,
        "d_ff": D_FF,
        "dropout": DROPOUT,
        "max_seq_length": MAX_SEQ_LENGTH,
        "vocab_size": VOCAB_SIZE,
        "num_labels": NUM_LABELS,
        "ner_tags": NER_TAGS,
    }


if __name__ == "__main__":
    print_config()