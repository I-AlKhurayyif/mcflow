"""
sampling.py
===========

Random sample generation for ``mcflow``.

This module is the engine of every Monte Carlo experiment. It exposes:

* Direct samplers for common distributions (uniform, normal, triangular,
  exponential, Bernoulli).
* General-purpose sampling algorithms (inverse transform, Box-Muller,
  rejection sampling).
* Variance-reduction sampling designs (Latin hypercube, stratified,
  antithetic).
"""

from __future__ import annotations

from typing import Callable, Tuple

import numpy as np
from tqdm.auto import tqdm

from . import distributions


_RNG = np.random.default_rng()


# ---------------------------------------------------------------------------
# RNG control
# ---------------------------------------------------------------------------
def set_seed(seed: int) -> None:
    """Set the global random seed used by ``mcflow`` samplers."""
    global _RNG
    _RNG = np.random.default_rng(seed)


def _rng(seed: int | None = None) -> np.random.Generator:
    """Return a generator: a fresh one if ``seed`` is given, else the global RNG."""
    if seed is None:
        return _RNG
    return np.random.default_rng(seed)


# ---------------------------------------------------------------------------
# Direct samplers
# ---------------------------------------------------------------------------
def sample_uniform(low: float, high: float, size: int, seed: int | None = None):
    """Draw ``size`` samples from Uniform(low, high)."""
    return _rng(seed).uniform(low, high, size)


def sample_normal(mean: float, std: float, size: int, seed: int | None = None):
    """Draw ``size`` samples from Normal(mean, std)."""
    return _rng(seed).normal(mean, std, size)


def sample_standard_normal(size: int, seed: int | None = None):
    """Draw ``size`` samples from the standard Normal distribution."""
    return _rng(seed).standard_normal(size)


def sample_triangular(
    low: float, mode: float, high: float, size: int, seed: int | None = None
):
    """Draw ``size`` samples from Triangular(low, mode, high)."""
    return _rng(seed).triangular(low, mode, high, size)


def sample_exponential(rate: float, size: int, seed: int | None = None):
    """Draw ``size`` samples from Exponential(rate)."""
    return _rng(seed).exponential(1.0 / rate, size)


def sample_bernoulli(p: float, size: int, seed: int | None = None):
    """Draw ``size`` samples from Bernoulli(p) as integers in {0, 1}."""
    return (_rng(seed).uniform(0.0, 1.0, size) < p).astype(int)


# ---------------------------------------------------------------------------
# General sampling algorithms
# ---------------------------------------------------------------------------
def inverse_transform_sample(
    cdf_inverse: Callable[[np.ndarray], np.ndarray],
    size: int,
    seed: int | None = None,
):
    """
    Inverse-transform sampling.

    Generates ``size`` uniform samples and feeds them through the supplied
    inverse CDF to produce samples from the target distribution.
    """
    u = _rng(seed).uniform(0.0, 1.0, size)
    return cdf_inverse(u)


def box_muller_transform(size: int, seed: int | None = None):
    """
    Box-Muller transform.

    Generates ``size`` independent standard normal samples from pairs of
    uniform samples. If ``size`` is odd, one extra sample is generated and
    discarded.
    """
    rng = _rng(seed)
    n_pairs = (size + 1) // 2
    u1 = rng.uniform(1e-12, 1.0, n_pairs)  # avoid log(0)
    u2 = rng.uniform(0.0, 1.0, n_pairs)
    r = np.sqrt(-2.0 * np.log(u1))
    z1 = r * np.cos(2.0 * np.pi * u2)
    z2 = r * np.sin(2.0 * np.pi * u2)
    z = np.concatenate([z1, z2])[:size]
    return z


def rejection_sample(
    target_pdf: Callable[[np.ndarray], np.ndarray],
    proposal_sampler: Callable[[int], np.ndarray],
    proposal_pdf: Callable[[np.ndarray], np.ndarray],
    M: float,
    size: int,
    max_iter: int = 1_000_000,
    seed: int | None = None,
    progress: bool = True,
):
    """
    Rejection sampling.

    Draws candidates from ``proposal_sampler`` and accepts each with
    probability ``target_pdf(x) / (M * proposal_pdf(x))``. The constant ``M``
    must satisfy ``target_pdf(x) <= M * proposal_pdf(x)`` for all ``x``.
    """
    rng = _rng(seed)
    accepted = np.empty(size, dtype=float)
    n_accepted = 0
    pbar = tqdm(total=size, disable=not progress, desc="rejection sampling")

    iterations = 0
    while n_accepted < size and iterations < max_iter:
        batch = max(size - n_accepted, 1)
        x = np.asarray(proposal_sampler(batch), dtype=float)
        u = rng.uniform(0.0, 1.0, batch)
        accept_prob = target_pdf(x) / (M * proposal_pdf(x))
        keep = u <= accept_prob
        n_keep = int(np.sum(keep))
        if n_keep > 0:
            take = min(n_keep, size - n_accepted)
            accepted[n_accepted : n_accepted + take] = x[keep][:take]
            n_accepted += take
            pbar.update(take)
        iterations += 1

    pbar.close()
    if n_accepted < size:
        raise RuntimeError(
            f"Rejection sampling did not collect {size} samples within "
            f"{max_iter} iterations (got {n_accepted}). Consider raising M "
            f"or improving the proposal."
        )
    return accepted


# ---------------------------------------------------------------------------
# Variance-reduction designs
# ---------------------------------------------------------------------------
def latin_hypercube_sample(
    n_samples: int, n_dim: int, seed: int | None = None
) -> np.ndarray:
    """
    Latin Hypercube Sampling on the unit hypercube [0, 1]^n_dim.

    Each dimension is partitioned into ``n_samples`` equal-probability strata
    and one sample is drawn from each stratum, with random permutations
    across dimensions.
    """
    rng = _rng(seed)
    cut = np.linspace(0.0, 1.0, n_samples + 1)
    u = rng.uniform(size=(n_samples, n_dim))
    a = cut[:n_samples][:, None]
    b = cut[1:][:, None]
    points = a + u * (b - a)
    for j in range(n_dim):
        rng.shuffle(points[:, j])
    return points


def stratified_sample(
    n_samples: int, low: float = 0.0, high: float = 1.0, seed: int | None = None
):
    """
    One-dimensional stratified sampling.

    Splits ``[low, high]`` into ``n_samples`` equal-width strata and draws
    one uniform sample per stratum.
    """
    rng = _rng(seed)
    edges = np.linspace(low, high, n_samples + 1)
    a = edges[:-1]
    b = edges[1:]
    u = rng.uniform(0.0, 1.0, n_samples)
    return a + u * (b - a)


def antithetic_sample(size: int, seed: int | None = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate antithetic uniform samples.

    Returns a pair ``(U, 1 - U)`` where ``U`` contains ``size`` uniform
    samples. Negative correlation between the two halves typically reduces
    estimator variance.
    """
    u = _rng(seed).uniform(0.0, 1.0, size)
    return u, 1.0 - u
