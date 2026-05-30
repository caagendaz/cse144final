"""Model resolution and optimizer construction."""

from __future__ import annotations

from typing import Iterable

import timm
import torch

from .config import VariantConfig


def resolve_model_name(patterns: Iterable[str]) -> str:
    """Find the first installed timm pretrained model matching a preferred name."""

    available_models = set(timm.list_models(pretrained=True))
    for pattern in patterns:
        if pattern in available_models:
            return pattern
        matches = sorted(name for name in available_models if pattern in name)
        if matches:
            return matches[0]
    raise ValueError(f"No timm model matched: {tuple(patterns)}")


def create_model(
    model_name: str,
    variant: VariantConfig,
    num_classes: int,
    pretrained: bool,
):
    model = timm.create_model(
        model_name,
        pretrained=pretrained,
        num_classes=num_classes,
        drop_rate=variant.drop_rate,
        drop_path_rate=variant.drop_path_rate,
    )
    if pretrained and variant.gradient_checkpointing:
        set_grad_checkpointing = getattr(model, "set_grad_checkpointing", None)
        if set_grad_checkpointing is None:
            print(f"Gradient checkpointing is unavailable for {model_name}.")
        else:
            set_grad_checkpointing(enable=True)
    return model


def make_optimizer(model, variant: VariantConfig) -> torch.optim.Optimizer:
    """Fine-tune the backbone conservatively while adapting the classifier faster."""

    classifier = model.get_classifier()
    classifier_parameter_ids = {id(parameter) for parameter in classifier.parameters()}
    backbone_parameters = [
        parameter
        for parameter in model.parameters()
        if id(parameter) not in classifier_parameter_ids
    ]
    classifier_parameters = list(classifier.parameters())

    return torch.optim.AdamW(
        [
            {
                "params": backbone_parameters,
                "lr": variant.learning_rate * variant.backbone_lr_multiplier,
            },
            {
                "params": classifier_parameters,
                "lr": variant.learning_rate,
            },
        ],
        weight_decay=variant.weight_decay,
        foreach=False,
    )
