"""Configuration, reproducibility, and JSON persistence."""
import json
import random
from pathlib import Path
import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]

def load_config(path=None):
    """Load configuration independent of the shell's working directory."""
    with open(path or ROOT / 'configs/config.yaml') as f:
        return yaml.safe_load(f)

def seed_everything(seed=42):
    """Seed Python, NumPy, CPU/GPU tensors; request deterministic operations."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.set_num_threads(8)

def save_json(path, value):
    """Atomically persist measured metadata."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temp.replace(path)
