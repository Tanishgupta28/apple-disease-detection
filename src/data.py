"""Shared preprocessing and efficient disk-backed PyTorch data loaders."""
import random
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageOps
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T
from .utils import ROOT

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

def transform(image_size=224, training=False):
    """Resize without cropping symptoms; augment only training; normalize RGB."""
    steps = [T.Resize((image_size, image_size), antialias=True)]
    if training:
        steps += [T.RandomHorizontalFlip(),
                  T.RandomAffine(degrees=12, translate=(0.05, 0.05), scale=(0.95, 1.05),
                                 interpolation=T.InterpolationMode.BILINEAR),
                  T.ColorJitter(brightness=0.12, contrast=0.12)]
    return T.Compose(steps + [T.ToTensor(), T.Normalize(MEAN, STD)])

def preprocess(image, image_size=224):
    """Apply the exact validation/test transformation for uploaded images."""
    return transform(image_size)(ImageOps.exif_transpose(image).convert('RGB'))

class LeafDataset(Dataset):
    """Load one image on demand using a persisted split manifest."""
    def __init__(self, rows, image_size=224, training=False):
        self.rows = rows.reset_index(drop=True)
        self.preprocessing = transform(image_size, training)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows.iloc[index]
        with Image.open(ROOT / row.path) as image:
            image = ImageOps.exif_transpose(image).convert('RGB')
            tensor = self.preprocessing(image)
        return tensor, int(row.label)

def worker_seed(worker_id):
    """Seed NumPy and Python consistently inside each loader worker."""
    seed = torch.initial_seed() % 2**32
    random.seed(seed)
    np.random.seed(seed)

def loaders(config, splits=('train', 'validation')):
    """Build only requested splits, keeping the test loader out of training."""
    rows = pd.read_csv(ROOT / config['processed_dir'] / 'manifest.csv')
    result = {}
    for split in splits:
        dataset = LeafDataset(rows[rows.split == split], config['image_size'], split == 'train')
        workers = config['num_workers']
        result[split] = DataLoader(dataset, batch_size=config['batch_size'],
            shuffle=split == 'train', num_workers=workers, pin_memory=torch.cuda.is_available(),
            persistent_workers=workers > 0, worker_init_fn=worker_seed,
            generator=torch.Generator().manual_seed(config['seed']))
    return result

def class_weights(config):
    """Derive imbalance weights from training labels alone."""
    rows = pd.read_csv(ROOT / config['processed_dir'] / 'manifest.csv')
    counts = rows[rows.split == 'train'].label.value_counts().sort_index().reindex(range(6), fill_value=0)
    if (counts == 0).any():
        raise ValueError('Every class must have training examples')
    use_weights = counts.max() / counts.min() >= config['class_weight_ratio_threshold']
    weights = len(rows[rows.split == 'train']) / (6 * counts.to_numpy()) if use_weights else np.ones(6)
    return torch.tensor(weights, dtype=torch.float32)
