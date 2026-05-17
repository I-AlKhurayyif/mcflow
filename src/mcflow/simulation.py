"""
simulation.py
=============

High-level simulation orchestration for ``mcflow``.

This module wires the lower-level samplers and estimators together into
common Monte Carlo workflows:

* Run a stochastic model many times and collect outputs.
* Estimate expectations, probabilities, and variances by simulation.
* Compute the batch-means estimator for correlated samples.
* Bootstrap-resample a set of samples and compute bootstrap confidence
  intervals for arbitrary statistics.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List

import numpy as np
from tqdm.auto import tqdm

from . import estimators, sampling


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------
@dataclass
class SimulationResult:
    """Container for a single simulation run."""
    samples: np.ndarray
    mean: float
    standard_error: float
    n_samples: int

    def __repr__(self) -> str:
        return (
            f"SimulationResult(mean={self.mean:.6g}, "
            f"se={self.standard_error:.3g}, n={self.n_samples})"
        )


# ---------------------------------------------------------------------------
# Simulation runners
# ---------------------------------------------------------------------------
def run_simulation(
    model: Callable[[], float],
    n_samples: int,
    seed: int | None = None,
    progress: bool = True,
) -> SimulationResult:
    """
    Run a stochastic ``model`` ``n_samples`` times and collect the outputs.

    ``model`` is a callable with no arguments that returns one scalar per
    call. If you need per-iteration randomness, manage the seed inside
    ``model`` or call ``mcflow.set_seed`` beforehand.
    """
    if seed is not None:
        sampling.set_seed(seed)

    out = np.empty(n_samples, dtype=float)
    iterator = range(n_samples)
    if progress:
        iterator = tqdm(iterator, desc="simulation")
    for i in iterator:
        out[i] = float(model())
    return SimulationResult(
        samples=out,
        mean=estimators.mc_mean(out),
        standard_error=estimators.standard_error(out),
        n_samples=n_samples,
    )


def run_replicated_simulation(
    model: Callable[[], float],
    n_samples: int,
    n_replications: int,
    progress: bool = True,
) -> List[SimulationResult]:
    """Run ``run_simulation`` ``n_replications`` times and return all results."""
    results: List[SimulationResult] = []
    iterator = range(n_replications)
    if progress:
        iterator = tqdm(iterator, desc="replications")
    for _ in iterator:
        results.append(run_simulation(model, n_samples, progress=False))
    return results


# ---------------------------------------------------------------------------
# Expectation / probability / variance via simulation
# ---------------------------------------------------------------------------
def simulate_expectation(
    func: Callable[[np.ndarray], np.ndarray],
    sampler: Callable[[int], np.ndarray],
    n_samples: int,
    progress: bool = True,
) -> SimulationResult:
    """
    Estimate E[g(X)] by Monte Carlo.

    Samples ``X_i`` from ``sampler(n_samples)`` and averages ``func(X_i)``.
    """
    x = np.asarray(sampler(n_samples), dtype=float)
    iterator = range(n_samples) if not progress else tqdm(
        range(n_samples), desc="expectation"
    )
    try:
        g = np.asarray(func(x), dtype=float)
        if progress:
            for _ in iterator:
                pass
    except Exception:
        g = np.empty(n_samples)
        for i in iterator:
            g[i] = float(func(x[i]))

    return SimulationResult(
        samples=g,
        mean=estimators.mc_mean(g),
        standard_error=estimators.standard_error(g),
        n_samples=n_samples,
    )


def simulate_probability(
    event_func: Callable[[np.ndarray], np.ndarray],
    sampler: Callable[[int], np.ndarray],
    n_samples: int,
    progress: bool = True,
) -> SimulationResult:
    """
    Estimate P(X in A) by Monte Carlo, where ``event_func`` is the indicator
    of the event of interest.

    ``event_func`` should return boolean or {0, 1} values.
    """
    x = np.asarray(sampler(n_samples), dtype=float)
    iterator = range(n_samples) if not progress else tqdm(
        range(n_samples), desc="probability"
    )
    try:
        ind = np.asarray(event_func(x), dtype=float)
        if progress:
            for _ in iterator:
                pass
    except Exception:
        ind = np.empty(n_samples)
        for i in iterator:
            ind[i] = float(event_func(x[i]))

    return SimulationResult(
        samples=ind,
        mean=estimators.mc_mean(ind),
        standard_error=estimators.standard_error(ind),
        n_samples=n_samples,
    )


def simulate_variance(
    func: Callable[[np.ndarray], np.ndarray],
    sampler: Callable[[int], np.ndarray],
    n_samples: int,
    progress: bool = True,
) -> float:
    """Estimate Var[g(X)] by Monte Carlo."""
    result = simulate_expectation(func, sampler, n_samples, progress=progress)
    return estimators.sample_variance(result.samples)


# ---------------------------------------------------------------------------
# Batch means
# ---------------------------------------------------------------------------
def batch_means(samples: np.ndarray, n_batches: int) -> SimulationResult:
    """
    Batch-means estimator.

    Splits ``samples`` into ``n_batches`` consecutive batches of equal size,
    averages within each batch, and returns the mean and standard error of
    the batch means. ``len(samples)`` must be divisible by ``n_batches``.
    """
    samples = np.asarray(samples, dtype=float)
    n = samples.size
    if n % n_batches != 0:
        raise ValueError("len(samples) must be divisible by n_batches.")
    m = n // n_batches
    batched = samples[: n_batches * m].reshape(n_batches, m)
    batch_avg = batched.mean(axis=1)
    return SimulationResult(
        samples=batch_avg,
        mean=estimators.mc_mean(batch_avg),
        standard_error=estimators.standard_error(batch_avg),
        n_samples=n_batches,
    )


# ---------------------------------------------------------------------------
# Bootstrap
# ---------------------------------------------------------------------------
def bootstrap_resample(
    samples: np.ndarray,
    n_resamples: int,
    seed: int | None = None,
    progress: bool = True,
) -> np.ndarray:
    """
    Generate ``n_resamples`` bootstrap resamples (with replacement) of
    ``samples``.

    Returns a 2-D array of shape ``(n_resamples, N)``.
    """
    samples = np.asarray(samples, dtype=float)
    n = samples.size
    rng = sampling._rng(seed)
    out = np.empty((n_resamples, n), dtype=float)
    iterator = range(n_resamples)
    if progress:
        iterator = tqdm(iterator, desc="bootstrap resample")
    for b in iterator:
        idx = rng.integers(0, n, n)
        out[b] = samples[idx]
    return out


def bootstrap_confidence_interval(
    samples: np.ndarray,
    statistic: Callable[[np.ndarray], float] = np.mean,
    n_resamples: int = 1000,
    level: float = 0.95,
    seed: int | None = None,
    progress: bool = True,
):
    """
    Bootstrap percentile confidence interval for an arbitrary statistic.

    Returns ``(point_estimate, ci_lower, ci_upper, bootstrap_statistics)``.
    """
    samples = np.asarray(samples, dtype=float)
    n = samples.size
    rng = sampling._rng(seed)
    boot_stats = np.empty(n_resamples, dtype=float)

    iterator = range(n_resamples)
    if progress:
        iterator = tqdm(iterator, desc="bootstrap CI")
    for b in iterator:
        idx = rng.integers(0, n, n)
        boot_stats[b] = float(statistic(samples[idx]))

    alpha = 1.0 - level
    lo, hi = np.percentile(boot_stats, [100.0 * alpha / 2.0, 100.0 * (1.0 - alpha / 2.0)])
    point = float(statistic(samples))
    return point, float(lo), float(hi), boot_stats
