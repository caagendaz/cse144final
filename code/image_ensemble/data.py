"""Dataset discovery and model-aware image transformations."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from PIL import Image
from timm.data import create_transform, resolve_model_data_config
from torch.utils.data import Dataset, WeightedRandomSampler
from torchvision import transforms
from torchvision.transforms import InterpolationMode

from .config import VariantConfig

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


def discover_train_items(train_dir: Path) -> list[tuple[Path, int]]:
    """Read class folders such as train/0, train/1, and so on."""

    items: list[tuple[Path, int]] = []
    class_dirs = [
        path
        for path in train_dir.iterdir()
        if path.is_dir() and path.name.isdigit()
    ]
    for class_dir in sorted(class_dirs, key=lambda path: int(path.name)):
        label = int(class_dir.name)
        for path in sorted(class_dir.iterdir()):
            if path.suffix.lower() in IMAGE_EXTENSIONS:
                items.append((path, label))
    return items


def discover_test_items(test_dir: Path) -> list[tuple[Path, str]]:
    def sort_key(path: Path) -> tuple[int, int | str]:
        return (0, int(path.stem)) if path.stem.isdigit() else (1, path.name)

    paths = [
        path
        for path in test_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    ]
    return [(path, path.name) for path in sorted(paths, key=sort_key)]


class ImageDataset(Dataset):
    def __init__(self, items: Sequence[tuple[Path, object]], transform=None):
        self.items = items
        self.transform = transform

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int):
        path, label_or_id = self.items[index]
        with Image.open(path) as image:
            image = image.convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, label_or_id


def summarize_class_counts(labels: Sequence[int]) -> dict[int, int]:
    return dict(sorted(Counter(int(label) for label in labels).items()))


def make_balanced_sampler(
    labels: Sequence[int],
    automatically_balance: bool,
    imbalance_ratio_threshold: float,
) -> tuple[WeightedRandomSampler | None, float]:
    """Create a sampler only when class imbalance is substantial."""

    counts = Counter(int(label) for label in labels)
    imbalance_ratio = max(counts.values()) / min(counts.values())
    if not automatically_balance or imbalance_ratio < imbalance_ratio_threshold:
        return None, imbalance_ratio

    sample_weights = torch.tensor(
        [1.0 / counts[int(label)] for label in labels],
        dtype=torch.double,
    )
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True,
    )
    return sampler, imbalance_ratio


def _interpolation_mode(name: str) -> InterpolationMode:
    modes = {
        "bicubic": InterpolationMode.BICUBIC,
        "bilinear": InterpolationMode.BILINEAR,
        "nearest": InterpolationMode.NEAREST,
    }
    return modes.get(name, InterpolationMode.BICUBIC)


def make_transforms(model, variant: VariantConfig):
    """Use each pretrained backbone's normalization and interpolation settings."""

    data_config = resolve_model_data_config(model)
    interpolation = _interpolation_mode(data_config.get("interpolation", "bicubic"))
    mean = data_config["mean"]
    std = data_config["std"]

    train_transform = transforms.Compose(
        [
            transforms.RandomResizedCrop(
                variant.image_size,
                scale=(variant.crop_minimum, 1.0),
                interpolation=interpolation,
            ),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(variant.rotation_degrees),
            transforms.ColorJitter(
                brightness=variant.color_jitter_strength,
                contrast=variant.color_jitter_strength,
                saturation=variant.color_jitter_strength,
                hue=0.03,
            ),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
            transforms.RandomErasing(
                p=variant.random_erasing_probability,
                scale=(0.02, 0.18),
                ratio=(0.3, 3.3),
            ),
        ]
    )

    evaluation_config = dict(data_config)
    evaluation_config["input_size"] = (3, variant.image_size, variant.image_size)
    evaluation_transform = create_transform(**evaluation_config, is_training=False)

    return train_transform, evaluation_transform
