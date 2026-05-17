"""
importance.py
=============

Importance sampling utilities for ``mcflow``.

This module implements:

* Standard (normalized) importance sampling.
* Self-normalized importance sampling, useful when densities are known
  only up to a constant.
* Importance-weight computation and normalization.
* Effective sample size (ESS).
* Variance estimation for the IS estimator.
* A Monte Carlo estimate of the KL divergence between target and proposal.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
from tqdm.auto import tqdm

from . import estimators


@dataclass
class ImportanceResult:
    """Container for importance-sampling output."""
    estimate: float
    standard_error: float
    weights: np.ndarray
    samples: np.ndarray
    effective_sample_size: float

    def __repr__(self) -> str:
        return (
            f"ImportanceResult(estimate={self.estimate:.6g}, "
            f"se={self.standard_error:.3g}, ess={self.effective_sample_size:.1f})"
        )


# ---------------------------------------------------------------------------
# Weights
# ---------------------------------------------------------------------------
def importance_weights(
    samples: np.ndarray,
    target_pdf: Callable[[np.ndarray], np.ndarray],
    proposal_pdf: Callable[[np.ndarray], np.ndarray],
) -> np.ndarray:
    """Raw importance weights w_i = p(x_i) / q(x_i)."""
    samples = np.asarray(samples, dtype=float)
    p = np.asarray(target_pdf(samples), dtype=float)
    q = np.asarray(proposal_pdf(samples), dtype=float)
    return p / q


def normalize_weights(weights: np.ndarray) -> np.ndarray:
    """Return weights normalized to sum to 1."""
    weights = np.asarray(weights, dtype=float)
    total = float(np.sum(weights))
    if total == 0.0:
        raise ValueError("All importance weights are zero; cannot normalize.")
    return weights / total


def effective_sample_size(weights: np.ndarray) -> float:
    """Effective sample size: (sum w)^2 / sum(w^2)."""
    weights = np.asarray(weights, dtype=float)
    num = float(np.sum(weights)) ** 2
    den = float(np.sum(weights**2))
    if den == 0.0:
        return 0.0
    return num / den


# ---------------------------------------------------------------------------
# Importance-sampling estimators
# ---------------------------------------------------------------------------
def importance_sampling(
    func: Callable[[np.ndarray], np.ndarray],
    target_pdf: Callable[[np.ndarray], np.ndarray],
    proposal_pdf: Callable[[np.ndarray], np.ndarray],
    proposal_sampler: Callable[[int], np.ndarray],
    n_samples: int,
    progress: bool = True,
) -> ImportanceResult:
    """
    Standard importance-sampling estimator:

        I_hat = (1/N) * sum_i f(X_i) * p(X_i) / q(X_i),  X_i ~ q
    """
    x = np.asarray(proposal_sampler(n_samples), dtype=float)

    iterator = range(n_samples) if not progress else tqdm(
        range(n_samples), desc="importance sampling"
    )
    try:
        fx = np.asarray(func(x), dtype=float)
        w = importance_weights(x, target_pdf, proposal_pdf)
        if progress:
            for _ in iterator:
                pass
    except Exception:
        fx = np.empty(n_samples)
        w = np.empty(n_samples)
        for i in iterator:
            xi = np.array([x[i]])
            fx[i] = float(func(xi))
            w[i] = float(importance_weights(xi, target_pdf, proposal_pdf))

    values = fx * w
    estimate = float(np.mean(values))
    se = float(estimators.standard_error(values))
    ess = effective_sample_size(w)
    return ImportanceResult(estimate, se, w, x, ess)


def self_normalized_importance_sampling(
    func: Callable[[np.ndarray], np.ndarray],
    target_pdf: Callable[[np.ndarray], np.ndarray],
    proposal_pdf: Callable[[np.ndarray], np.ndarray],
    proposal_sampler: Callable[[int], np.ndarray],
    n_samples: int,
    progress: bool = True,
) -> ImportanceResult:
    """
    Self-normalized importance sampling:

        I_hat = sum_i f(X_i) w_i / sum_i w_i,  X_i ~ q

    Useful when ``p`` and/or ``q`` are known only up to a normalizing
    constant.
    """
    x = np.asarray(proposal_sampler(n_samples), dtype=float)

    iterator = range(n_samples) if not progress else tqdm(
        range(n_samples), desc="SNIS"
    )
    try:
        fx = np.asarray(func(x), dtype=float)
        w = importance_weights(x, target_pdf, proposal_pdf)
        if progress:
            for _ in iterator:
                pass
    except Exception:
        fx = np.empty(n_samples)
        w = np.empty(n_samples)
        for i in iterator:
            xi = np.array([x[i]])
            fx[i] = float(func(xi))
            w[i] = float(importance_weights(xi, target_pdf, proposal_pdf))

    w_sum = float(np.sum(w))
    estimate = float(np.sum(fx * w) / w_sum)

    # Delta-method standard error for the ratio estimator.
    w_norm = w / w_sum
    se = float(np.sqrt(np.sum(w_norm**2 * (fx - estimate) ** 2)))
    ess = effective_sample_size(w)
    return ImportanceResult(estimate, se, w, x, ess)


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------
def variance_of_is_estimator(
    samples: np.ndarray,
    weights: np.ndarray,
) -> float:
    """
    Empirical variance of the importance-sampling estimator using the
    pre-computed weighted samples.
    """
    samples = np.asarray(samples, dtype=float)
    weights = np.asarray(weights, dtype=float)
    values = samples * weights
    n = values.size
    if n < 2:
        raise ValueError("Need at least 2 samples to estimate variance.")
    return float(np.var(values, ddof=1) / n)


def kl_divergence(
    target_pdf: Callable[[np.ndarray], np.ndarray],
    proposal_pdf: Callable[[np.ndarray], np.ndarray],
    samples: np.ndarray,
) -> float:
    """
    Monte Carlo estimate of D_KL(p || q) using samples drawn from ``p``.

    D_KL(p || q) ~= (1/N) * sum_i log( p(X_i) / q(X_i) ),  X_i ~ p
    """
    samples = np.asarray(samples, dtype=float)
    p = np.asarray(target_pdf(samples), dtype=float)
    q = np.asarray(proposal_pdf(samples), dtype=float)
    # guard against zero or negative values
    mask = (p > 0) & (q > 0)
    if not np.any(mask):
        return float("inf")
    return float(np.mean(np.log(p[mask] / q[mask])))
