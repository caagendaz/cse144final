"""Configuration for training and blending the image models."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class VariantConfig:
    """One independently trained model variant."""

    name: str
    family: str
    model_patterns: tuple[str, ...]
    image_size: int
    batch_size: int
    accum_steps: int
    learning_rate: float
    weight_decay: float
    drop_rate: float
    drop_path_rate: float
    mixup_alpha: float
    cutmix_alpha: float
    mixup_probability: float
    label_smoothing: float
    random_erasing_probability: float
    rotation_degrees: int
    color_jitter_strength: float
    crop_minimum: float
    warmup_epochs: int
    patience: int
    seed_offset: int = 0
    backbone_lr_multiplier: float = 0.35
    gradient_checkpointing: bool = True
    use_ema: bool = False
    ema_decay: float = 0.9998
    enabled: bool = True


@dataclass(frozen=True)
class ProjectConfig:
    """Settings shared across the full experiment."""

    train_dir: Path
    test_dir: Path
    output_dir: Path
    num_classes: int = 100
    num_folds: int = 4
    final_epochs: int = 120
    prediction_batch_size: int = 8
    num_workers: int = 2
    pin_memory: bool = False
    persistent_workers: bool = False
    prefetch_factor: int = 1
    seed: int = 42
    use_amp: bool = True
    automatically_balance_classes: bool = True
    imbalance_ratio_threshold: float = 1.5
    fold_weight_temperature: float = 0.08
    blend_objective: str = "log_loss"
    blend_search_iterations: int = 4000
    tta_candidates: tuple[str, ...] = ("none", "horizontal_flip")
    variants: tuple[VariantConfig, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        for key in ("train_dir", "test_dir", "output_dir"):
            data[key] = str(data[key])
        return data


def default_colab_config(
    kaggle_dir: str | Path = "/content/drive/MyDrive/Kaggle",
) -> ProjectConfig:
    """Return the recommended full H100/A100 experiment configuration."""

    kaggle_dir = Path(kaggle_dir)
    return ProjectConfig(
        train_dir=kaggle_dir / "train",
        test_dir=kaggle_dir / "test",
        output_dir=kaggle_dir / "improved_oof_ensemble",
        variants=(
            VariantConfig(
                name="convnext_large",
                family="convnext",
                model_patterns=(
                    "convnext_large.fb_in22k_ft_in1k",
                    "convnext_large",
                ),
                image_size=384,
                batch_size=16,
                accum_steps=4,
                learning_rate=1.73e-4,
                weight_decay=0.089,
                drop_rate=0.141,
                drop_path_rate=0.186,
                mixup_alpha=0.15,
                cutmix_alpha=0.50,
                mixup_probability=0.68,
                label_smoothing=0.08,
                random_erasing_probability=0.25,
                rotation_degrees=8,
                color_jitter_strength=0.23,
                crop_minimum=0.59,
                warmup_epochs=5,
                patience=12,
            ),
            VariantConfig(
                name="swin_base",
                family="swin",
                model_patterns=(
                    "swin_base_patch4_window12_384.ms_in22k_ft_in1k",
                    "swin_base_patch4_window12_384",
                    "swin_base",
                ),
                image_size=384,
                batch_size=12,
                accum_steps=4,
                learning_rate=5.93e-5,
                weight_decay=0.072,
                drop_rate=0.155,
                drop_path_rate=0.165,
                mixup_alpha=0.12,
                cutmix_alpha=0.70,
                mixup_probability=0.65,
                label_smoothing=0.08,
                random_erasing_probability=0.21,
                rotation_degrees=6,
                color_jitter_strength=0.28,
                crop_minimum=0.66,
                warmup_epochs=6,
                patience=12,
            ),
            VariantConfig(
                name="eva02_large_seed_a",
                family="eva02",
                model_patterns=(
                    "eva02_large_patch14_448.mim_m38m_ft_in22k_in1k",
                    "eva02_large_patch14_448",
                    "eva02_large",
                ),
                image_size=448,
                batch_size=6,
                accum_steps=16,
                learning_rate=9.13e-5,
                weight_decay=0.075,
                drop_rate=0.166,
                drop_path_rate=0.175,
                mixup_alpha=0.03,
                cutmix_alpha=0.50,
                mixup_probability=0.545,
                label_smoothing=0.04,
                random_erasing_probability=0.30,
                rotation_degrees=6,
                color_jitter_strength=0.17,
                crop_minimum=0.59,
                warmup_epochs=4,
                patience=20,
                seed_offset=0,
            ),
            VariantConfig(
                name="eva02_large_seed_b",
                family="eva02",
                model_patterns=(
                    "eva02_large_patch14_448.mim_m38m_ft_in22k_in1k",
                    "eva02_large_patch14_448",
                    "eva02_large",
                ),
                image_size=448,
                batch_size=6,
                accum_steps=16,
                learning_rate=8.50e-5,
                weight_decay=0.075,
                drop_rate=0.166,
                drop_path_rate=0.175,
                mixup_alpha=0.03,
                cutmix_alpha=0.50,
                mixup_probability=0.545,
                label_smoothing=0.04,
                random_erasing_probability=0.30,
                rotation_degrees=6,
                color_jitter_strength=0.17,
                crop_minimum=0.62,
                warmup_epochs=4,
                patience=20,
                seed_offset=10_000,
            ),
        ),
    )
