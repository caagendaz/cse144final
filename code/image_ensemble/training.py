"""Training and inference for one model variant at a time."""

from __future__ import annotations

import gc
import json
import math
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import StratifiedKFold
from timm.data import Mixup
from timm.loss import SoftTargetCrossEntropy
from timm.utils import ModelEmaV2
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader

from .config import ProjectConfig, VariantConfig
from .data import ImageDataset, make_balanced_sampler, make_transforms
from .ensemble import (
    classification_accuracy,
    fold_weights_from_losses,
    multiclass_log_loss,
    normalize_probabilities,
)
from .models import create_model, make_optimizer, resolve_model_name


@dataclass
class VariantResult:
    name: str
    family: str
    model_name: str
    tta_mode: str
    oof_probabilities: np.ndarray
    test_probabilities: np.ndarray
    test_ids: list[str]
    fold_weights: np.ndarray
    fold_metrics: list[dict[str, object]]
    checkpoint_paths: list[str]

    def metadata(self) -> dict[str, object]:
        return {
            "name": self.name,
            "family": self.family,
            "model_name": self.model_name,
            "tta_mode": self.tta_mode,
            "oof_accuracy": classification_accuracy(self.oof_probabilities, self._labels),
            "oof_log_loss": multiclass_log_loss(self.oof_probabilities, self._labels),
            "fold_weights": self.fold_weights.tolist(),
            "fold_metrics": self.fold_metrics,
            "checkpoint_paths": self.checkpoint_paths,
        }

    # Labels are attached after construction to avoid storing them in saved metadata.
    _labels: np.ndarray | None = None


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def cleanup(device: torch.device) -> None:
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()


def cuda_memory_summary(device: torch.device) -> str:
    if device.type != "cuda":
        return ""
    allocated = torch.cuda.memory_allocated() / 1024**3
    reserved = torch.cuda.memory_reserved() / 1024**3
    peak = torch.cuda.max_memory_allocated() / 1024**3
    return f" | VRAM {allocated:.2f} GB allocated, {reserved:.2f} GB reserved, {peak:.2f} GB peak"


def make_scheduler(
    optimizer: torch.optim.Optimizer,
    total_steps: int,
    warmup_steps: int,
) -> torch.optim.lr_scheduler.LambdaLR:
    def multiplier(step: int) -> float:
        if step < warmup_steps:
            return float(step + 1) / float(max(1, warmup_steps))
        progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, multiplier)


def _loader(
    dataset: ImageDataset,
    batch_size: int,
    num_workers: int,
    pin_memory: bool,
    persistent_workers: bool,
    prefetch_factor: int,
    shuffle: bool = False,
    sampler=None,
) -> DataLoader:
    worker_options = {}
    if num_workers > 0:
        worker_options = {
            "persistent_workers": persistent_workers,
            "prefetch_factor": prefetch_factor,
        }
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle if sampler is None else False,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
        **worker_options,
    )


def _train_one_epoch(
    model,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler,
    scaler: GradScaler,
    criterion,
    mixup: Mixup,
    accum_steps: int,
    ema: ModelEmaV2 | None,
    device: torch.device,
    use_amp: bool,
) -> tuple[float, float]:
    model.train()
    optimizer.zero_grad(set_to_none=True)
    total_loss = 0.0
    total_matches = 0
    total_samples = 0

    for step, (images, labels) in enumerate(loader):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        original_labels = labels.clone()
        images, mixed_labels = mixup(images, labels)

        with autocast(device_type=device.type, enabled=use_amp):
            logits = model(images)
            loss = criterion(logits, mixed_labels) / accum_steps

        scaler.scale(loss).backward()
        should_update = (step + 1) % accum_steps == 0 or (step + 1) == len(loader)
        if should_update:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()
            if ema is not None:
                ema.update(model)

        batch_size = images.size(0)
        total_loss += loss.item() * accum_steps * batch_size
        total_matches += (logits.detach().argmax(dim=1) == original_labels).sum().item()
        total_samples += batch_size

    # This accuracy is only a rough diagnostic because Mixup/CutMix alter the inputs.
    return total_loss / total_samples, total_matches / total_samples


@torch.inference_mode()
def _predict_probabilities(
    model,
    loader: DataLoader,
    device: torch.device,
    use_amp: bool,
    tta_mode: str,
) -> tuple[np.ndarray, list[object]]:
    model.eval()
    all_probabilities: list[np.ndarray] = []
    all_values: list[object] = []

    for images, values in loader:
        images = images.to(device, non_blocking=True)
        with autocast(device_type=device.type, enabled=use_amp):
            logits = model(images)
            if tta_mode == "horizontal_flip":
                flipped_logits = model(torch.flip(images, dims=[3]))
                logits = (logits + flipped_logits) / 2.0
            elif tta_mode != "none":
                raise ValueError(f"Unsupported TTA mode: {tta_mode}")

        probabilities = torch.softmax(logits, dim=1).cpu().numpy()
        all_probabilities.append(probabilities)
        if isinstance(values, torch.Tensor):
            all_values.extend(values.cpu().numpy().tolist())
        else:
            all_values.extend(list(values))

    return normalize_probabilities(np.concatenate(all_probabilities)), all_values


def _evaluation_metrics(probabilities: np.ndarray, labels: np.ndarray) -> dict[str, float]:
    return {
        "accuracy": classification_accuracy(probabilities, labels),
        "log_loss": multiclass_log_loss(probabilities, labels),
    }


def _save_checkpoint(
    path: Path,
    model,
    variant: VariantConfig,
    model_name: str,
    fold: int,
    epoch: int,
    metrics: dict[str, float],
) -> None:
    torch.save(
        {
            "variant": asdict(variant),
            "model_name": model_name,
            "fold": fold,
            "epoch": epoch,
            "metrics": metrics,
            "model_state_dict": model.state_dict(),
        },
        path,
    )


def _load_checkpoint_model(
    path: Path,
    model_name: str,
    variant: VariantConfig,
    config: ProjectConfig,
    device: torch.device,
):
    model = create_model(
        model_name=model_name,
        variant=variant,
        num_classes=config.num_classes,
        pretrained=False,
    )
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    del checkpoint
    return model.to(device)


def _choose_tta_mode(
    candidates: Sequence[str],
    oof_by_tta: dict[str, np.ndarray],
    labels: np.ndarray,
) -> tuple[str, dict[str, dict[str, float]]]:
    metrics = {
        mode: _evaluation_metrics(oof_by_tta[mode], labels)
        for mode in candidates
    }
    selected = min(
        candidates,
        key=lambda mode: (metrics[mode]["log_loss"], -metrics[mode]["accuracy"]),
    )
    return selected, metrics


def train_variant(
    variant: VariantConfig,
    config: ProjectConfig,
    train_items: Sequence[tuple[Path, int]],
    test_items: Sequence[tuple[Path, str]],
    labels: np.ndarray,
    device: torch.device,
) -> VariantResult:
    """Train folds, select TTA with OOF predictions, and infer one variant."""

    output_dir = config.output_dir / "variants" / variant.name
    output_dir.mkdir(parents=True, exist_ok=True)
    model_name = resolve_model_name(variant.model_patterns)
    print(f"\n{'=' * 88}\nTraining {variant.name}: {model_name}\n{'=' * 88}")

    template_model = create_model(model_name, variant, config.num_classes, pretrained=False)
    train_transform, evaluation_transform = make_transforms(template_model, variant)
    del template_model

    splitter = StratifiedKFold(
        n_splits=config.num_folds,
        shuffle=True,
        random_state=config.seed + variant.seed_offset,
    )
    oof_by_tta = {
        mode: np.zeros((len(train_items), config.num_classes), dtype=np.float32)
        for mode in config.tta_candidates
    }
    checkpoint_paths: list[str] = []
    fold_metrics: list[dict[str, object]] = []

    for fold, (train_indices, validation_indices) in enumerate(
        splitter.split(np.zeros(len(labels)), labels),
        start=1,
    ):
        fold_seed = config.seed + variant.seed_offset + fold
        seed_everything(fold_seed)
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()
        print(f"\n{variant.name} | fold {fold}/{config.num_folds} | seed {fold_seed}")

        fold_train_items = [train_items[index] for index in train_indices]
        fold_validation_items = [train_items[index] for index in validation_indices]
        fold_train_labels = labels[train_indices]
        fold_validation_labels = labels[validation_indices]
        sampler, imbalance_ratio = make_balanced_sampler(
            fold_train_labels,
            automatically_balance=config.automatically_balance_classes,
            imbalance_ratio_threshold=config.imbalance_ratio_threshold,
        )

        train_loader = _loader(
            ImageDataset(fold_train_items, train_transform),
            batch_size=variant.batch_size,
            num_workers=config.num_workers,
            pin_memory=config.pin_memory,
            persistent_workers=config.persistent_workers,
            prefetch_factor=config.prefetch_factor,
            shuffle=True,
            sampler=sampler,
        )
        validation_loader = _loader(
            ImageDataset(fold_validation_items, evaluation_transform),
            batch_size=config.prediction_batch_size,
            num_workers=config.num_workers,
            pin_memory=config.pin_memory,
            persistent_workers=config.persistent_workers,
            prefetch_factor=config.prefetch_factor,
        )

        model = create_model(model_name, variant, config.num_classes, pretrained=True).to(device)
        ema = ModelEmaV2(model, decay=variant.ema_decay) if variant.use_ema else None
        optimizer = make_optimizer(model, variant)
        updates_per_epoch = math.ceil(len(train_loader) / variant.accum_steps)
        scheduler = make_scheduler(
            optimizer,
            total_steps=updates_per_epoch * config.final_epochs,
            warmup_steps=updates_per_epoch * variant.warmup_epochs,
        )
        scaler = GradScaler(device.type, enabled=config.use_amp and device.type == "cuda")
        mixup = Mixup(
            mixup_alpha=variant.mixup_alpha,
            cutmix_alpha=variant.cutmix_alpha,
            prob=variant.mixup_probability,
            switch_prob=0.5,
            mode="batch",
            label_smoothing=variant.label_smoothing,
            num_classes=config.num_classes,
        )
        criterion = SoftTargetCrossEntropy()

        checkpoint_path = output_dir / f"{variant.name}_fold{fold}_best.pt"
        best_metrics = {"accuracy": -1.0, "log_loss": float("inf")}
        best_epoch = -1
        stale_epochs = 0

        for epoch in range(1, config.final_epochs + 1):
            started = time.time()
            train_loss, mixed_input_accuracy = _train_one_epoch(
                model=model,
                loader=train_loader,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                criterion=criterion,
                mixup=mixup,
                accum_steps=variant.accum_steps,
                ema=ema,
                device=device,
                use_amp=config.use_amp and device.type == "cuda",
            )
            validation_model = ema.module if ema is not None else model
            validation_probabilities, _ = _predict_probabilities(
                validation_model,
                validation_loader,
                device,
                config.use_amp and device.type == "cuda",
                tta_mode="none",
            )
            validation_metrics = _evaluation_metrics(
                validation_probabilities,
                fold_validation_labels,
            )
            improved = (
                validation_metrics["accuracy"] > best_metrics["accuracy"]
                or (
                    validation_metrics["accuracy"] == best_metrics["accuracy"]
                    and validation_metrics["log_loss"] < best_metrics["log_loss"]
                )
            )
            if improved:
                best_metrics = validation_metrics
                best_epoch = epoch
                stale_epochs = 0
                _save_checkpoint(
                    checkpoint_path,
                    validation_model,
                    variant,
                    model_name,
                    fold,
                    epoch,
                    validation_metrics,
                )
            else:
                stale_epochs += 1

            print(
                f"{variant.name} | fold {fold} | epoch {epoch}/{config.final_epochs} | "
                f"train loss {train_loss:.4f} | mixed-input acc {mixed_input_accuracy:.4f} | "
                f"val loss {validation_metrics['log_loss']:.4f} | "
                f"val acc {validation_metrics['accuracy']:.4f} | "
                f"best {best_metrics['accuracy']:.4f} @ {best_epoch} | "
                f"stale {stale_epochs}/{variant.patience} | {time.time() - started:.1f}s"
                f"{cuda_memory_summary(device)}"
            )
            if stale_epochs >= variant.patience:
                print("Early stopping.")
                break

        del model, ema, optimizer, scheduler, scaler, train_loader
        cleanup(device)

        selected_model = _load_checkpoint_model(
            checkpoint_path,
            model_name,
            variant,
            config,
            device,
        )
        fold_tta_metrics: dict[str, dict[str, float]] = {}
        for tta_mode in config.tta_candidates:
            probabilities, _ = _predict_probabilities(
                selected_model,
                validation_loader,
                device,
                config.use_amp and device.type == "cuda",
                tta_mode=tta_mode,
            )
            oof_by_tta[tta_mode][validation_indices] = probabilities
            fold_tta_metrics[tta_mode] = _evaluation_metrics(
                probabilities,
                fold_validation_labels,
            )

        checkpoint_paths.append(str(checkpoint_path))
        fold_metrics.append(
            {
                "fold": fold,
                "seed": fold_seed,
                "checkpoint": str(checkpoint_path),
                "best_epoch": best_epoch,
                "checkpoint_metrics": best_metrics,
                "tta_metrics": fold_tta_metrics,
                "class_imbalance_ratio": imbalance_ratio,
                "used_balanced_sampler": sampler is not None,
            }
        )
        del selected_model, validation_loader
        cleanup(device)

    selected_tta, tta_metrics = _choose_tta_mode(config.tta_candidates, oof_by_tta, labels)
    selected_oof = oof_by_tta[selected_tta]
    selected_fold_losses = [
        float(metrics["tta_metrics"][selected_tta]["log_loss"])
        for metrics in fold_metrics
    ]
    fold_weights = fold_weights_from_losses(
        selected_fold_losses,
        temperature=config.fold_weight_temperature,
    )
    print(f"\n{variant.name} selected TTA: {selected_tta}")
    print(f"{variant.name} OOF TTA metrics: {tta_metrics}")
    print(f"{variant.name} fold weights: {fold_weights.tolist()}")

    test_loader = _loader(
        ImageDataset(test_items, evaluation_transform),
        batch_size=config.prediction_batch_size,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        persistent_workers=config.persistent_workers,
        prefetch_factor=config.prefetch_factor,
    )
    weighted_test_probabilities = np.zeros(
        (len(test_items), config.num_classes),
        dtype=np.float32,
    )
    test_ids: list[str] = []
    for checkpoint_path, fold_weight in zip(checkpoint_paths, fold_weights):
        selected_model = _load_checkpoint_model(
            Path(checkpoint_path),
            model_name,
            variant,
            config,
            device,
        )
        probabilities, fold_test_ids = _predict_probabilities(
            selected_model,
            test_loader,
            device,
            config.use_amp and device.type == "cuda",
            selected_tta,
        )
        if not test_ids:
            test_ids = [str(value) for value in fold_test_ids]
        elif test_ids != [str(value) for value in fold_test_ids]:
            raise ValueError("Test item order changed between fold predictions.")
        weighted_test_probabilities += probabilities * float(fold_weight)
        del selected_model
        cleanup(device)

    weighted_test_probabilities = normalize_probabilities(weighted_test_probabilities)
    np.save(output_dir / "oof_probabilities.npy", selected_oof)
    np.save(output_dir / "test_probabilities.npy", weighted_test_probabilities)
    with (output_dir / "variant_info.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "name": variant.name,
                "family": variant.family,
                "model_name": model_name,
                "variant": asdict(variant),
                "selected_tta": selected_tta,
                "oof_tta_metrics": tta_metrics,
                "fold_weights": fold_weights.tolist(),
                "fold_metrics": fold_metrics,
                "checkpoint_paths": checkpoint_paths,
            },
            handle,
            indent=2,
        )

    result = VariantResult(
        name=variant.name,
        family=variant.family,
        model_name=model_name,
        tta_mode=selected_tta,
        oof_probabilities=selected_oof,
        test_probabilities=weighted_test_probabilities,
        test_ids=test_ids,
        fold_weights=fold_weights,
        fold_metrics=fold_metrics,
        checkpoint_paths=checkpoint_paths,
    )
    result._labels = labels
    del test_loader
    cleanup(device)
    return result
