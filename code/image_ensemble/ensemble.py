"""Small, dependency-light utilities for validation-driven ensembling."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Mapping, Sequence

import numpy as np

EPSILON = 1e-12


@dataclass(frozen=True)
class BlendResult:
    """The selected blend method and its out-of-fold measurements."""

    names: tuple[str, ...]
    weights: tuple[float, ...]
    mode: str
    accuracy: float
    log_loss: float
    objective: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def normalize_probabilities(probabilities: np.ndarray) -> np.ndarray:
    """Clip numerical noise and make each row sum to one."""

    probabilities = np.asarray(probabilities)
    if probabilities.ndim != 2:
        raise ValueError("Expected a two-dimensional probability matrix.")
    dtype = np.float32 if probabilities.dtype == np.float32 else np.float64
    probabilities = probabilities.astype(dtype, copy=False)
    probabilities = np.clip(probabilities, EPSILON, None)
    return probabilities / probabilities.sum(axis=1, keepdims=True)


def classification_accuracy(probabilities: np.ndarray, labels: np.ndarray) -> float:
    probabilities = normalize_probabilities(probabilities)
    labels = np.asarray(labels)
    return float(np.mean(probabilities.argmax(axis=1) == labels))


def multiclass_log_loss(probabilities: np.ndarray, labels: np.ndarray) -> float:
    probabilities = normalize_probabilities(probabilities)
    labels = np.asarray(labels, dtype=np.int64)
    if probabilities.shape[0] != labels.shape[0]:
        raise ValueError("Prediction and label counts do not match.")
    chosen = probabilities[np.arange(labels.shape[0]), labels]
    return float(-np.mean(np.log(np.clip(chosen, EPSILON, 1.0))))


def blend_predictions(
    prediction_sets: Sequence[np.ndarray],
    weights: Sequence[float],
    mode: str = "probabilities",
) -> np.ndarray:
    """Blend probabilities arithmetically or as log-probability scores."""

    if not prediction_sets:
        raise ValueError("At least one prediction matrix is required.")

    weights_array = np.asarray(weights, dtype=np.float64)
    if weights_array.shape != (len(prediction_sets),):
        raise ValueError("There must be one weight per prediction matrix.")
    if np.any(weights_array < 0) or weights_array.sum() <= 0:
        raise ValueError("Blend weights must be non-negative with a positive sum.")
    weights_array = weights_array / weights_array.sum()

    matrices = [normalize_probabilities(matrix) for matrix in prediction_sets]
    if len({matrix.shape for matrix in matrices}) != 1:
        raise ValueError("All prediction matrices must have the same shape.")

    stacked = np.stack(matrices, axis=0)
    if mode == "probabilities":
        blended = np.tensordot(weights_array, stacked, axes=(0, 0))
    elif mode == "logits":
        log_scores = np.tensordot(weights_array, np.log(stacked), axes=(0, 0))
        log_scores -= log_scores.max(axis=1, keepdims=True)
        blended = np.exp(log_scores)
    else:
        raise ValueError(f"Unknown blend mode: {mode}")

    return normalize_probabilities(blended)


def fold_weights_from_losses(
    losses: Sequence[float],
    temperature: float = 0.08,
) -> np.ndarray:
    """Give lower-loss folds more influence while retaining fold diversity."""

    losses_array = np.asarray(losses, dtype=np.float64)
    if losses_array.ndim != 1 or losses_array.size == 0:
        raise ValueError("At least one fold loss is required.")
    if temperature <= 0:
        raise ValueError("Temperature must be positive.")

    scores = -(losses_array - losses_array.min()) / temperature
    scores -= scores.max()
    weights = np.exp(scores)
    return weights / weights.sum()


def _score_candidate(
    probabilities: np.ndarray,
    labels: np.ndarray,
    objective: str,
) -> tuple[float, float]:
    accuracy = classification_accuracy(probabilities, labels)
    log_loss = multiclass_log_loss(probabilities, labels)
    if objective == "log_loss":
        return log_loss, -accuracy
    if objective == "accuracy":
        return -accuracy, log_loss
    raise ValueError(f"Unsupported blend objective: {objective}")


def _candidate_weights(num_models: int, iterations: int, seed: int) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    candidates = [np.full(num_models, 1.0 / num_models)]
    candidates.extend(np.eye(num_models))
    candidates.extend(rng.dirichlet(np.ones(num_models), size=iterations))
    return candidates


def _refine_weights(
    matrices: Sequence[np.ndarray],
    labels: np.ndarray,
    weights: np.ndarray,
    mode: str,
    objective: str,
) -> np.ndarray:
    best_weights = weights.copy()
    best_score = _score_candidate(blend_predictions(matrices, best_weights, mode), labels, objective)

    for step_size in (0.10, 0.05, 0.02, 0.01, 0.005):
        improved = True
        while improved:
            improved = False
            for source in range(len(best_weights)):
                for target in range(len(best_weights)):
                    if source == target or best_weights[source] < step_size:
                        continue
                    candidate = best_weights.copy()
                    candidate[source] -= step_size
                    candidate[target] += step_size
                    score = _score_candidate(
                        blend_predictions(matrices, candidate, mode),
                        labels,
                        objective,
                    )
                    if score < best_score:
                        best_weights = candidate
                        best_score = score
                        improved = True
    return best_weights


def optimize_blend(
    predictions_by_name: Mapping[str, np.ndarray],
    labels: np.ndarray,
    objective: str = "log_loss",
    iterations: int = 4000,
    seed: int = 42,
) -> BlendResult:
    """Select weights and an averaging method from out-of-fold predictions."""

    if not predictions_by_name:
        raise ValueError("At least one model prediction set is required.")

    names = tuple(predictions_by_name)
    matrices = [normalize_probabilities(predictions_by_name[name]) for name in names]
    labels = np.asarray(labels, dtype=np.int64)

    best_score: tuple[float, float] | None = None
    best_weights: np.ndarray | None = None
    best_mode = ""

    for mode in ("probabilities", "logits"):
        for weights in _candidate_weights(len(names), iterations, seed):
            probabilities = blend_predictions(matrices, weights, mode)
            score = _score_candidate(probabilities, labels, objective)
            if best_score is None or score < best_score:
                best_score = score
                best_weights = np.asarray(weights, dtype=np.float64)
                best_mode = mode

    assert best_weights is not None
    best_weights = _refine_weights(matrices, labels, best_weights, best_mode, objective)
    final = blend_predictions(matrices, best_weights, best_mode)

    return BlendResult(
        names=names,
        weights=tuple(float(weight) for weight in best_weights / best_weights.sum()),
        mode=best_mode,
        accuracy=classification_accuracy(final, labels),
        log_loss=multiclass_log_loss(final, labels),
        objective=objective,
    )
