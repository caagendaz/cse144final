"""Run all enabled model variants and create the final submission."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .config import ProjectConfig
from .data import discover_test_items, discover_train_items, summarize_class_counts
from .ensemble import blend_predictions, optimize_blend
from .training import train_variant


def run_experiment(config: ProjectConfig) -> Path:
    """Train variants, optimize the OOF blend, and write a Kaggle submission."""

    config.output_dir.mkdir(parents=True, exist_ok=True)
    train_items = discover_train_items(config.train_dir)
    test_items = discover_test_items(config.test_dir)
    labels = np.asarray([label for _, label in train_items], dtype=np.int64)
    class_counts = summarize_class_counts(labels)

    if not train_items:
        raise ValueError(f"No training images found under {config.train_dir}")
    if not test_items:
        raise ValueError(f"No test images found under {config.test_dir}")
    if len(class_counts) != config.num_classes:
        raise ValueError(
            f"Expected {config.num_classes} classes but discovered {len(class_counts)}."
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        total_vram = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"GPU memory: {total_vram:.2f} GB")
    print(f"Training images: {len(train_items)}")
    print(f"Test images: {len(test_items)}")
    print(f"Smallest class: {min(class_counts.values())}")
    print(f"Largest class: {max(class_counts.values())}")

    results = []
    for variant in config.variants:
        if variant.enabled:
            results.append(
                train_variant(
                    variant=variant,
                    config=config,
                    train_items=train_items,
                    test_items=test_items,
                    labels=labels,
                    device=device,
                )
            )
    if not results:
        raise ValueError("No enabled model variants were configured.")

    oof_by_name = {result.name: result.oof_probabilities for result in results}
    test_by_name = {result.name: result.test_probabilities for result in results}
    blend = optimize_blend(
        oof_by_name,
        labels,
        objective=config.blend_objective,
        iterations=config.blend_search_iterations,
        seed=config.seed,
    )
    final_probabilities = blend_predictions(
        [test_by_name[name] for name in blend.names],
        blend.weights,
        blend.mode,
    )
    final_predictions = final_probabilities.argmax(axis=1).astype(int)
    test_ids = results[0].test_ids
    for result in results[1:]:
        if result.test_ids != test_ids:
            raise ValueError(f"Test ID mismatch for variant {result.name}.")

    submission_path = config.output_dir / "improved_oof_ensemble_ID_target.csv"
    probabilities_path = config.output_dir / "improved_oof_ensemble_probs.npy"
    info_path = config.output_dir / "improved_oof_ensemble_info.json"
    pd.DataFrame({"ID": test_ids, "target": final_predictions}).to_csv(
        submission_path,
        index=False,
    )
    np.save(probabilities_path, final_probabilities)
    with info_path.open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "submission": str(submission_path),
                "probabilities": str(probabilities_path),
                "config": config.to_dict(),
                "class_counts": class_counts,
                "blend": blend.to_dict(),
                "variants": [result.metadata() for result in results],
            },
            handle,
            indent=2,
        )

    print(f"\nSelected blend: {blend.to_dict()}")
    print(f"Saved submission: {submission_path}")
    print(f"Saved probabilities: {probabilities_path}")
    print(f"Saved experiment metadata: {info_path}")
    return submission_path
