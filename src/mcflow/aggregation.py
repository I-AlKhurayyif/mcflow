"""
aggregation.py
==============

Result aggregation and convergence diagnostics for ``mcflow``.

This module collects results from one or many simulation runs and produces
summary statistics, convergence indicators, and autocorrelation
diagnostics.
"""

from __future__ import annotations

from typing import Dict, List, Sequence

import numpy as np
from tqdm.auto import tqdm

from . import estimators


# ---------------------------------------------------------------------------
# Summaries
# ---------------------------------------------------------------------------
def summarize(samples, level: float = 0.95) -> Dict[str, float]:
    """
    Compute a standard summary dictionary for a 1-D sample array.

    Returns mean, standard deviation, standard error, the confidence
    interval at ``level``, the 95% percentile interval, the minimum and
    maximum, and the sample size.
    """
    samples = np.asarray(samples, dtype=float)
    lo_ci, hi_ci = estimators.confidence_interval(samples, level=level)
    lo_pi, hi_pi = estimators.percentile_interval(samples, 2.5, 97.5)
    return {
        "n": int(samples.size),
        "mean": estimators.mc_mean(samples),
        "std": estimators.standard_deviation(samples),
        "se": estimators.standard_error(samples),
        "ci_lower": lo_ci,
        "ci_upper": hi_ci,
        "pi_lower": lo_pi,
        "pi_upper": hi_pi,
        "min": float(np.min(samples)),
        "max": float(np.max(samples)),
    }


def aggregate_replications(
    results: Sequence[Sequence[float]], level: float = 0.95
) -> Dict[str, float]:
    """
    Aggregate the means of multiple replicated simulation runs.

    Each entry of ``results`` is a sample array from one replication. This
    function returns a summary over the per-replication means, which gives
    a clean Monte Carlo error estimate across independent runs.
    """
    means = np.array([np.mean(np.asarray(r, dtype=float)) for r in results])
    return summarize(means, level=level)


# ---------------------------------------------------------------------------
# Convergence diagnostics
# ---------------------------------------------------------------------------
def relative_error(estimate: float, true_value: float) -> float:
    """Relative error |estimate - true_value| / |true_value|."""
    if true_value == 0.0:
        return float("inf")
    return float(abs(estimate - true_value) / abs(true_value))


def convergence_diagnostic(samples) -> np.ndarray:
    """
    Running relative standard error R_n = SE_n / |mu_n|.

    A common stopping rule is to keep sampling until R_n falls below a
    chosen tolerance (e.g., 0.01).
    """
    samples = np.asarray(samples, dtype=float)
    mu_n = estimators.running_mean(samples)
    se_n = estimators.running_standard_error(samples)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(np.abs(mu_n) > 0, se_n / np.abs(mu_n), np.inf)
    return ratio


def gelman_rubin_diagnostic(chains) -> float:
    """
    Gelman-Rubin potential scale reduction factor (R-hat).

    ``chains`` is a 2-D array of shape (n_chains, n_samples). A value close
    to 1 indicates that the chains have converged.
    """
    chains = np.asarray(chains, dtype=float)
    if chains.ndim != 2:
        raise ValueError("chains must be a 2-D array (n_chains, n_samples).")
    m, n = chains.shape
    if m < 2:
        raise ValueError("Need at least 2 chains to compute R-hat.")

    chain_means = np.mean(chains, axis=1)
    chain_vars = np.var(chains, axis=1, ddof=1)
    grand_mean = np.mean(chain_means)

    B = n * np.sum((chain_means - grand_mean) ** 2) / (m - 1)
    W = np.mean(chain_vars)
    var_hat = ((n - 1) / n) * W + (1.0 / n) * B
    return float(np.sqrt(var_hat / W))


# ---------------------------------------------------------------------------
# Autocorrelation and effective sample size
# ---------------------------------------------------------------------------
def autocorrelation(samples, lag: int) -> float:
    """Sample autocorrelation rho_k at the specified ``lag``."""
    samples = np.asarray(samples, dtype=float)
    n = samples.size
    if lag < 0 or lag >= n:
        raise ValueError("lag must satisfy 0 <= lag < N.")
    mu = np.mean(samples)
    num = np.sum((samples[: n - lag] - mu) * (samples[lag:] - mu))
    den = np.sum((samples - mu) ** 2)
    if den == 0.0:
        return 0.0
    return float(num / den)


def autocorrelation_function(
    samples, max_lag: int | None = None, progress: bool = False
) -> np.ndarray:
    """Autocorrelation values for lags 0, 1, ..., ``max_lag``."""
    samples = np.asarray(samples, dtype=float)
    n = samples.size
    if max_lag is None:
        max_lag = min(n - 1, 50)
    lags = range(max_lag + 1)
    if progress:
        lags = tqdm(lags, desc="autocorrelation")
    return np.array([autocorrelation(samples, k) for k in lags])


def integrated_autocorrelation_time(
    samples, max_lag: int | None = None
) -> float:
    """
    Integrated autocorrelation time tau_int = 1 + 2 * sum_k rho_k.

    The sum is truncated either at ``max_lag`` or at the first lag where the
    autocorrelation becomes non-positive (Sokal's automatic windowing).
    """
    samples = np.asarray(samples, dtype=float)
    n = samples.size
    if max_lag is None:
        max_lag = min(n - 1, 1000)

    tau = 1.0
    for k in range(1, max_lag + 1):
        rho = autocorrelation(samples, k)
        if rho <= 0.0:
            break
        tau += 2.0 * rho
    return float(tau)


def effective_sample_size_correlated(samples) -> float:
    """Effective sample size for a correlated sequence: N / tau_int."""
    samples = np.asarray(samples, dtype=float)
    tau = integrated_autocorrelation_time(samples)
    return float(samples.size / tau)
