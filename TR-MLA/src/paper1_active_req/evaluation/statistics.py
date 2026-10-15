from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from scipy.stats import rankdata


def holm_adjust(p_values: Sequence[float]) -> list[float]:
    if any(value < 0.0 or value > 1.0 for value in p_values):
        raise ValueError("p-values must be between 0 and 1")
    order = np.argsort(p_values)
    adjusted = np.empty(len(p_values), dtype=float)
    running = 0.0
    for rank, index in enumerate(order):
        value = min(1.0, (len(p_values) - rank) * p_values[index])
        running = max(running, value)
        adjusted[index] = running
    return adjusted.tolist()


def paired_rank_biserial(differences: np.ndarray) -> float:
    """Return the signed-rank effect size for paired observations."""
    nonzero = np.asarray(differences, dtype=float)
    nonzero = nonzero[~np.isclose(nonzero, 0.0)]
    if len(nonzero) == 0:
        return 0.0
    ranks = rankdata(np.abs(nonzero), method="average")
    positive = float(ranks[nonzero > 0.0].sum())
    negative = float(ranks[nonzero < 0.0].sum())
    return (positive - negative) / (positive + negative)


def paired_bootstrap_mean_ci(
    differences: np.ndarray,
    *,
    seed: int,
    resamples: int = 10_000,
) -> tuple[float, float]:
    """Return a fixed-seed percentile interval for a paired mean difference."""
    values = np.asarray(differences, dtype=float)
    if len(values) == 0:
        return (0.0, 0.0)
    if resamples < 1:
        raise ValueError("resamples must be positive")
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(values), size=(resamples, len(values)))
    bootstrap_means = values[indices].mean(axis=1)
    low, high = np.quantile(bootstrap_means, [0.025, 0.975])
    return (float(low), float(high))
