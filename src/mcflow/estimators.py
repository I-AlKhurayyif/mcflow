"""
estimators.py
=============

Statistical primitives for ``mcflow``.

This module provides point estimators (mean, weighted mean), dispersion
measures (variance, standard deviation), uncertainty quantification
(standard error, confidence and percentile intervals), running diagnostics,
error metrics, and higher-order moments (skewness, kurtosis).

Every other module ultimately depends on these primitives.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
from scipy.special import erfinv
from scipy.stats import t as student_t
from tqdm.auto import tqdm


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _z_critical(level: float) -> float:
    """Two-sided z critical value for the given confidence ``level``."""
    alpha = 1.0 - level
    return float(np.sqrt(2.0) * erfinv(1.0 - alpha))


# ---------------------------------------------------------------------------
# Point estimators
# ---------------------------------------------------------------------------
def mc_mean(samples) -> float:
    """Monte Carlo mean estimator: arithmetic average of the samples."""
    samples = np.asarray(samples, dtype=float)
    return float(np.mean(samples))


def weighted_mean(samples, weights) -> float:
    """Weighted Monte Carlo mean."""
    samples = np.asarray(samples, dtype=float)
    weights = np.asarray(weights, dtype=float)
    return float(np.sum(weights * samples) / np.sum(weights))


# ---------------------------------------------------------------------------
# Dispersion
# ---------------------------------------------------------------------------
def sample_variance(samples) -> float:
    """Unbiased sample variance (Bessel's correction, ddof=1)."""
    samples = np.asarray(samples, dtype=float)
    return float(np.var(samples, ddof=1))


def population_variance(samples) -> float:
    """Biased / population variance (ddof=0)."""
    samples = np.asarray(samples, dtype=float)
    return float(np.var(samples, ddof=0))


def weighted_variance(samples, weights) -> float:
    """Weighted variance using normalized weights."""
    samples = np.asarray(samples, dtype=float)
    weights = np.asarray(weights, dtype=float)
    mu_w = weighted_mean(samples, weights)
    return float(np.sum(weights * (samples - mu_w) ** 2) / np.sum(weights))


def standard_deviation(samples) -> float:
    """Sample standard deviation (square root of the unbiased variance)."""
    return float(np.sqrt(sample_variance(samples)))


# ---------------------------------------------------------------------------
# Uncertainty
# ---------------------------------------------------------------------------
def standard_error(samples) -> float:
    """Standard error of the mean: S / sqrt(N)."""
    samples = np.asarray(samples, dtype=float)
    n = samples.size
    return float(standard_deviation(samples) / np.sqrt(n))


def weighted_standard_error(samples, weights) -> float:
    """Standard error for a weighted-mean estimator."""
    samples = np.asarray(samples, dtype=float)
    weights = np.asarray(weights, dtype=float)
    mu_w = weighted_mean(samples, weights)
    num = np.sum(weights**2 * (samples - mu_w) ** 2)
    den = np.sum(weights) ** 2
    return float(np.sqrt(num / den))


def confidence_interval(samples, level: float = 0.95) -> Tuple[float, float]:
    """Normal-approximation confidence interval for the mean."""
    mu = mc_mean(samples)
    se = standard_error(samples)
    z = _z_critical(level)
    return (mu - z * se, mu + z * se)


def t_confidence_interval(samples, level: float = 0.95) -> Tuple[float, float]:
    """Student-t confidence interval for the mean (preferred when N is small)."""
    samples = np.asarray(samples, dtype=float)
    n = samples.size
    if n < 2:
        raise ValueError("Need at least 2 samples for a t-based CI.")
    mu = mc_mean(samples)
    se = standard_error(samples)
    alpha = 1.0 - level
    t_crit = float(student_t.ppf(1.0 - alpha / 2.0, df=n - 1))
    return (mu - t_crit * se, mu + t_crit * se)


def percentile_interval(
    samples, lower: float = 2.5, upper: float = 97.5
) -> Tuple[float, float]:
    """Nonparametric interval defined by empirical quantiles."""
    samples = np.asarray(samples, dtype=float)
    lo, hi = np.percentile(samples, [lower, upper])
    return float(lo), float(hi)


# ---------------------------------------------------------------------------
# Running diagnostics
# ---------------------------------------------------------------------------
def running_mean(samples, progress: bool = False) -> np.ndarray:
    """Running average mu_n for n = 1, ..., N. Vectorized."""
    samples = np.asarray(samples, dtype=float)
    if progress:
        # mostly cosmetic: a single sweep is already linear and fast
        for _ in tqdm(range(1), desc="running mean", disable=not progress):
            out = np.cumsum(samples) / np.arange(1, samples.size + 1)
        return out
    return np.cumsum(samples) / np.arange(1, samples.size + 1)


def running_variance(samples, progress: bool = False) -> np.ndarray:
    """
    Running unbiased variance using Welford's online algorithm.

    Returns an array of length N where element n is the sample variance
    using the first ``n + 1`` samples (with NaN for n = 0).
    """
    samples = np.asarray(samples, dtype=float)
    n = samples.size
    out = np.empty(n, dtype=float)
    out[0] = np.nan

    mean = samples[0]
    M2 = 0.0
    iterator = range(1, n)
    if progress:
        iterator = tqdm(iterator, desc="running variance")
    for i in iterator:
        x = samples[i]
        delta = x - mean
        mean += delta / (i + 1)
        M2 += delta * (x - mean)
        out[i] = M2 / i  # divide by (i+1) - 1 = i
    return out


def running_standard_error(samples, progress: bool = False) -> np.ndarray:
    """Running standard error: S_n / sqrt(n)."""
    var = running_variance(samples, progress=progress)
    n = np.arange(1, var.size + 1, dtype=float)
    return np.sqrt(var / n)


# ---------------------------------------------------------------------------
# Error metrics
# ---------------------------------------------------------------------------
def bias(estimate: float, true_value: float) -> float:
    """Estimator bias = estimate - true_value."""
    return float(estimate) - float(true_value)


def mean_squared_error(estimate, true_value: float) -> float:
    """Mean squared error of an estimator (or array of estimates)."""
    estimate = np.asarray(estimate, dtype=float)
    return float(np.mean((estimate - true_value) ** 2))


def root_mean_squared_error(estimate, true_value: float) -> float:
    """Root mean squared error."""
    return float(np.sqrt(mean_squared_error(estimate, true_value)))


def coefficient_of_variation(samples) -> float:
    """Coefficient of variation: standard deviation divided by |mean|."""
    mu = mc_mean(samples)
    if mu == 0.0:
        return float("inf")
    return standard_deviation(samples) / abs(mu)


# ---------------------------------------------------------------------------
# Higher moments and association
# ---------------------------------------------------------------------------
def skewness(samples) -> float:
    """Sample skewness (third standardized moment)."""
    samples = np.asarray(samples, dtype=float)
    mu = np.mean(samples)
    s = np.sqrt(np.mean((samples - mu) ** 2))
    if s == 0.0:
        return 0.0
    return float(np.mean((samples - mu) ** 3) / s**3)


def kurtosis(samples) -> float:
    """Excess kurtosis (fourth standardized moment minus 3)."""
    samples = np.asarray(samples, dtype=float)
    mu = np.mean(samples)
    s = np.sqrt(np.mean((samples - mu) ** 2))
    if s == 0.0:
        return 0.0
    return float(np.mean((samples - mu) ** 4) / s**4 - 3.0)


def covariance(samples_x, samples_y) -> float:
    """Unbiased sample covariance between two equal-length samples."""
    x = np.asarray(samples_x, dtype=float)
    y = np.asarray(samples_y, dtype=float)
    if x.size != y.size:
        raise ValueError("samples_x and samples_y must have the same length.")
    return float(np.cov(x, y, ddof=1)[0, 1])


def correlation(samples_x, samples_y) -> float:
    """Pearson correlation coefficient between two equal-length samples."""
    x = np.asarray(samples_x, dtype=float)
    y = np.asarray(samples_y, dtype=float)
    if x.size != y.size:
        raise ValueError("samples_x and samples_y must have the same length.")
    return float(np.corrcoef(x, y)[0, 1])
