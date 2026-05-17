"""
integration.py
==============

Monte Carlo integration for ``mcflow``.

Supports:

* One-dimensional and multi-dimensional uniform Monte Carlo integration.
* Importance-sampling integration.
* Antithetic-variate integration.
* Stratified-sample integration.
* Control-variate integration.

Each integrator returns the integral estimate, its standard error, and a
running history of partial estimates for convergence inspection.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence, Tuple

import numpy as np
from tqdm.auto import tqdm

from . import estimators, sampling


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------
@dataclass
class IntegrationResult:
    """Container for Monte Carlo integration output."""
    estimate: float
    standard_error: float
    n_samples: int
    running_estimate: np.ndarray
    samples: np.ndarray  # the integrand evaluations

    def __repr__(self) -> str:
        return (
            f"IntegrationResult(estimate={self.estimate:.6g}, "
            f"se={self.standard_error:.3g}, n={self.n_samples})"
        )


# ---------------------------------------------------------------------------
# One-dimensional uniform Monte Carlo
# ---------------------------------------------------------------------------
def mc_integrate_1d(
    func: Callable[[np.ndarray], np.ndarray],
    low: float,
    high: float,
    n_samples: int,
    seed: int | None = None,
    progress: bool = True,
) -> IntegrationResult:
    """
    Estimate the integral of ``func`` over ``[low, high]`` by uniform sampling.

    Returns an ``IntegrationResult`` with estimate, standard error, the
    running integral history, and the array of scaled integrand values.
    """
    volume = high - low
    x = sampling.sample_uniform(low, high, n_samples, seed=seed)

    if progress:
        # vectorize when possible; otherwise loop with a progress bar
        try:
            fx = func(x)
        except Exception:
            fx = np.empty(n_samples)
            for i in tqdm(range(n_samples), desc="mc_integrate_1d"):
                fx[i] = func(x[i])
    else:
        fx = func(x)

    fx = np.asarray(fx, dtype=float)
    values = volume * fx
    estimate = float(np.mean(values))
    se = float(volume * estimators.standard_error(fx))
    running = estimators.running_mean(values)
    return IntegrationResult(estimate, se, n_samples, running, values)


# ---------------------------------------------------------------------------
# Multi-dimensional uniform Monte Carlo
# ---------------------------------------------------------------------------
def mc_integrate_nd(
    func: Callable[[np.ndarray], np.ndarray],
    bounds: Sequence[Tuple[float, float]],
    n_samples: int,
    seed: int | None = None,
    progress: bool = True,
) -> IntegrationResult:
    """
    Estimate a multi-dimensional integral over an axis-aligned box.

    ``bounds`` is a sequence of ``(low_j, high_j)`` pairs, one per
    dimension. ``func`` must accept a 2-D array of shape ``(n_samples,
    n_dim)`` and return a 1-D array of length ``n_samples``.
    """
    bounds = np.asarray(bounds, dtype=float)
    if bounds.ndim != 2 or bounds.shape[1] != 2:
        raise ValueError("bounds must have shape (n_dim, 2).")
    lows = bounds[:, 0]
    highs = bounds[:, 1]
    n_dim = bounds.shape[0]
    volume = float(np.prod(highs - lows))

    rng = sampling._rng(seed)
    u = rng.uniform(size=(n_samples, n_dim))
    x = lows + u * (highs - lows)

    if progress:
        try:
            fx = func(x)
        except Exception:
            fx = np.empty(n_samples)
            for i in tqdm(range(n_samples), desc="mc_integrate_nd"):
                fx[i] = func(x[i])
    else:
        fx = func(x)

    fx = np.asarray(fx, dtype=float)
    values = volume * fx
    estimate = float(np.mean(values))
    se = float(volume * estimators.standard_error(fx))
    running = estimators.running_mean(values)
    return IntegrationResult(estimate, se, n_samples, running, values)


# ---------------------------------------------------------------------------
# Error reporting
# ---------------------------------------------------------------------------
def integration_error(samples, volume: float) -> float:
    """Standard error of a Monte Carlo integral, volume * S_f / sqrt(N)."""
    return float(volume * estimators.standard_error(samples))


# ---------------------------------------------------------------------------
# Importance-sampling integration
# ---------------------------------------------------------------------------
def mc_integrate_importance(
    func: Callable[[np.ndarray], np.ndarray],
    proposal_sampler: Callable[[int], np.ndarray],
    proposal_pdf: Callable[[np.ndarray], np.ndarray],
    n_samples: int,
    progress: bool = True,
) -> IntegrationResult:
    """
    Estimate the integral of ``func`` using importance sampling.

    Samples are drawn from a proposal distribution ``q`` with sampler
    ``proposal_sampler`` and density ``proposal_pdf``. The estimator is

        I_hat = (1/N) * sum_i f(X_i) / q(X_i),  X_i ~ q
    """
    x = np.asarray(proposal_sampler(n_samples), dtype=float)

    iterator = range(n_samples) if not progress else tqdm(
        range(n_samples), desc="importance integration"
    )
    f_over_q = np.empty(n_samples, dtype=float)
    # vectorize if possible
    try:
        fx = np.asarray(func(x), dtype=float)
        qx = np.asarray(proposal_pdf(x), dtype=float)
        f_over_q = fx / qx
        if progress:
            for _ in iterator:  # consume the bar
                pass
    except Exception:
        for i in iterator:
            f_over_q[i] = float(func(x[i])) / float(proposal_pdf(x[i]))

    estimate = float(np.mean(f_over_q))
    se = float(estimators.standard_error(f_over_q))
    running = estimators.running_mean(f_over_q)
    return IntegrationResult(estimate, se, n_samples, running, f_over_q)


# ---------------------------------------------------------------------------
# Antithetic-variate integration
# ---------------------------------------------------------------------------
def mc_integrate_antithetic(
    func: Callable[[np.ndarray], np.ndarray],
    low: float,
    high: float,
    n_samples: int,
    seed: int | None = None,
    progress: bool = True,
) -> IntegrationResult:
    """
    Estimate the integral of ``func`` over ``[low, high]`` using antithetic
    uniform pairs.

    ``n_samples`` is the total number of integrand evaluations and must be
    even; it is split into ``n_samples / 2`` antithetic pairs.
    """
    if n_samples % 2 != 0:
        raise ValueError("n_samples must be even for antithetic integration.")
    half = n_samples // 2

    u, u_anti = sampling.antithetic_sample(half, seed=seed)
    volume = high - low
    x = low + u * volume
    x_anti = low + u_anti * volume

    iterator = range(half) if not progress else tqdm(
        range(half), desc="antithetic integration"
    )
    try:
        fx = np.asarray(func(x), dtype=float)
        fx_anti = np.asarray(func(x_anti), dtype=float)
        if progress:
            for _ in iterator:
                pass
    except Exception:
        fx = np.empty(half)
        fx_anti = np.empty(half)
        for i in iterator:
            fx[i] = float(func(x[i]))
            fx_anti[i] = float(func(x_anti[i]))

    pair_means = 0.5 * (fx + fx_anti)
    values = volume * pair_means
    estimate = float(np.mean(values))
    se = float(volume * estimators.standard_error(pair_means))
    running = estimators.running_mean(values)
    return IntegrationResult(estimate, se, n_samples, running, values)


# ---------------------------------------------------------------------------
# Stratified integration
# ---------------------------------------------------------------------------
def mc_integrate_stratified(
    func: Callable[[np.ndarray], np.ndarray],
    low: float,
    high: float,
    n_samples: int,
    n_strata: int,
    seed: int | None = None,
    progress: bool = True,
) -> IntegrationResult:
    """
    Estimate the integral of ``func`` over ``[low, high]`` with stratified
    sampling.

    The interval is split into ``n_strata`` equal-width strata, each
    receiving ``n_samples / n_strata`` uniform samples. ``n_samples`` must
    be divisible by ``n_strata``.
    """
    if n_samples % n_strata != 0:
        raise ValueError("n_samples must be divisible by n_strata.")
    per_stratum = n_samples // n_strata
    edges = np.linspace(low, high, n_strata + 1)
    rng = sampling._rng(seed)

    stratum_means = np.empty(n_strata, dtype=float)
    stratum_widths = np.empty(n_strata, dtype=float)
    all_values: list[np.ndarray] = []

    iterator = range(n_strata)
    if progress:
        iterator = tqdm(iterator, desc="stratified integration")

    for k in iterator:
        a, b = edges[k], edges[k + 1]
        u = rng.uniform(a, b, per_stratum)
        try:
            fx = np.asarray(func(u), dtype=float)
        except Exception:
            fx = np.array([float(func(v)) for v in u])
        stratum_means[k] = np.mean(fx)
        stratum_widths[k] = b - a
        all_values.append((b - a) * fx)

    estimate = float(np.sum(stratum_widths * stratum_means))
    # variance combines per-stratum variances weighted by squared width
    variances = np.array([np.var(v / w, ddof=1) for v, w in zip(all_values, stratum_widths)])
    se = float(np.sqrt(np.sum(stratum_widths**2 * variances / per_stratum)))
    flat = np.concatenate(all_values)
    running = estimators.running_mean(flat) * n_strata  # scaled for visualization
    return IntegrationResult(estimate, se, n_samples, running, flat)


# ---------------------------------------------------------------------------
# Control-variate integration
# ---------------------------------------------------------------------------
def mc_integrate_control_variate(
    func: Callable[[np.ndarray], np.ndarray],
    control_func: Callable[[np.ndarray], np.ndarray],
    control_mean: float,
    low: float,
    high: float,
    n_samples: int,
    seed: int | None = None,
    progress: bool = True,
) -> IntegrationResult:
    """
    Estimate the integral of ``func`` over ``[low, high]`` using a control
    variate ``control_func`` with known expectation ``control_mean`` under
    the uniform distribution on ``[low, high]``.

    The variance-minimizing coefficient c* = Cov(f, h) / Var(h) is computed
    from the same samples.
    """
    volume = high - low
    x = sampling.sample_uniform(low, high, n_samples, seed=seed)

    iterator = range(n_samples) if not progress else tqdm(
        range(n_samples), desc="control variate integration"
    )
    try:
        fx = np.asarray(func(x), dtype=float)
        hx = np.asarray(control_func(x), dtype=float)
        if progress:
            for _ in iterator:
                pass
    except Exception:
        fx = np.empty(n_samples)
        hx = np.empty(n_samples)
        for i in iterator:
            fx[i] = float(func(x[i]))
            hx[i] = float(control_func(x[i]))

    var_h = float(np.var(hx, ddof=1))
    c_star = float(np.cov(fx, hx, ddof=1)[0, 1] / var_h) if var_h > 0 else 0.0
    adjusted = fx - c_star * (hx - control_mean)
    values = volume * adjusted
    estimate = float(np.mean(values))
    se = float(volume * estimators.standard_error(adjusted))
    running = estimators.running_mean(values)
    return IntegrationResult(estimate, se, n_samples, running, values)
