"""
mcmc.py
=======

Markov Chain Monte Carlo samplers for ``mcflow``.

This module implements the canonical MCMC algorithms used in modern
Bayesian computation:

* Metropolis-Hastings (general, asymmetric proposals).
* Random-walk Metropolis (symmetric Gaussian proposal).
* Gibbs sampling (component-wise conditional updates).
* Hamiltonian Monte Carlo (HMC) with leapfrog integration.
* No-U-Turn Sampler (NUTS), Algorithm 3 of Hoffman & Gelman (2014).

Plus a small set of post-processing helpers (burn-in, thinning,
acceptance rate, Geweke diagnostic) that are universally useful for
working with MCMC output.

All samplers operate in log-density space for numerical stability and
return a :class:`MCMCResult` containing the chain, log-probabilities,
and acceptance statistics.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Sequence

import numpy as np
from tqdm.auto import tqdm

from . import sampling


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------
@dataclass
class MCMCResult:
    """Output of an MCMC sampler."""
    samples: np.ndarray         # shape (n_samples, n_dim)
    log_probs: np.ndarray       # shape (n_samples,)
    acceptance_rate: float
    n_accepted: int
    n_samples: int

    def __repr__(self) -> str:
        return (
            f"MCMCResult(n={self.n_samples}, "
            f"acc_rate={self.acceptance_rate:.3f}, "
            f"shape={self.samples.shape})"
        )


# ---------------------------------------------------------------------------
# Metropolis-Hastings (general)
# ---------------------------------------------------------------------------
def metropolis_hastings(
    log_target: Callable[[np.ndarray], float],
    proposal_sampler: Callable[[np.ndarray], np.ndarray],
    log_proposal: Callable[[np.ndarray, np.ndarray], float],
    x0,
    n_samples: int,
    seed: int | None = None,
    progress: bool = True,
) -> MCMCResult:
    """
    General Metropolis-Hastings sampler.

    Parameters
    ----------
    log_target : callable
        Returns ``log p(x)`` up to an additive constant.
    proposal_sampler : callable
        Given the current state ``x``, returns a proposed state ``x'``.
    log_proposal : callable
        ``log_proposal(x_new, x_old)`` returns ``log q(x_new | x_old)``.
    x0 : array-like
        Initial state.
    n_samples : int
        Number of samples to produce (including ``x0`` is not counted).
    """
    rng = sampling._rng(seed)
    x = np.atleast_1d(np.asarray(x0, dtype=float)).copy()
    n_dim = x.size

    samples = np.empty((n_samples, n_dim), dtype=float)
    log_probs = np.empty(n_samples, dtype=float)
    n_accepted = 0

    log_target_x = float(log_target(x))

    iterator = range(n_samples)
    if progress:
        iterator = tqdm(iterator, desc="metropolis-hastings")

    for i in iterator:
        x_new = np.atleast_1d(np.asarray(proposal_sampler(x), dtype=float))
        log_target_new = float(log_target(x_new))
        log_alpha = (
            log_target_new - log_target_x
            + float(log_proposal(x, x_new))
            - float(log_proposal(x_new, x))
        )
        if np.log(rng.uniform()) < log_alpha:
            x = x_new
            log_target_x = log_target_new
            n_accepted += 1
        samples[i] = x
        log_probs[i] = log_target_x

    if n_dim == 1:
        samples = samples.ravel()
    return MCMCResult(
        samples=samples,
        log_probs=log_probs,
        acceptance_rate=n_accepted / n_samples,
        n_accepted=n_accepted,
        n_samples=n_samples,
    )


# ---------------------------------------------------------------------------
# Random-walk Metropolis
# ---------------------------------------------------------------------------
def random_walk_metropolis(
    log_target: Callable[[np.ndarray], float],
    x0,
    step_size,
    n_samples: int,
    seed: int | None = None,
    progress: bool = True,
) -> MCMCResult:
    """
    Random-walk Metropolis with a symmetric Gaussian proposal.

    ``step_size`` can be a scalar or a vector; it acts as the standard
    deviation of the Gaussian step in each dimension. Because the proposal
    is symmetric, the acceptance ratio reduces to
    ``exp(log p(x') - log p(x))``.
    """
    rng = sampling._rng(seed)
    x = np.atleast_1d(np.asarray(x0, dtype=float)).copy()
    n_dim = x.size
    step = np.broadcast_to(np.asarray(step_size, dtype=float), (n_dim,)).copy()

    samples = np.empty((n_samples, n_dim), dtype=float)
    log_probs = np.empty(n_samples, dtype=float)
    n_accepted = 0

    log_target_x = float(log_target(x))

    iterator = range(n_samples)
    if progress:
        iterator = tqdm(iterator, desc="random-walk MH")

    for i in iterator:
        x_new = x + step * rng.standard_normal(n_dim)
        log_target_new = float(log_target(x_new))
        log_alpha = log_target_new - log_target_x
        if np.log(rng.uniform()) < log_alpha:
            x = x_new
            log_target_x = log_target_new
            n_accepted += 1
        samples[i] = x
        log_probs[i] = log_target_x

    if n_dim == 1:
        samples = samples.ravel()
    return MCMCResult(
        samples=samples,
        log_probs=log_probs,
        acceptance_rate=n_accepted / n_samples,
        n_accepted=n_accepted,
        n_samples=n_samples,
    )


# ---------------------------------------------------------------------------
# Gibbs sampler
# ---------------------------------------------------------------------------
def gibbs_sampler(
    conditional_samplers: Sequence[Callable[[np.ndarray], float]],
    x0,
    n_samples: int,
    log_target: Callable[[np.ndarray], float] | None = None,
    progress: bool = True,
) -> MCMCResult:
    """
    Systematic-scan Gibbs sampler.

    Parameters
    ----------
    conditional_samplers : sequence of callables
        ``conditional_samplers[j](x)`` returns a single sample from the
        full conditional ``p(x_j | x_{-j})`` given the current full
        state ``x``.
    x0 : array-like
        Initial state. Its length must match ``len(conditional_samplers)``.
    log_target : callable, optional
        If given, used to record the log-density at each iteration. Has
        no effect on the sampling itself (Gibbs accepts every move).
    """
    x = np.atleast_1d(np.asarray(x0, dtype=float)).copy()
    n_dim = x.size
    if len(conditional_samplers) != n_dim:
        raise ValueError(
            "Number of conditional samplers must equal the dimension of x0."
        )

    samples = np.empty((n_samples, n_dim), dtype=float)
    log_probs = np.empty(n_samples, dtype=float)

    iterator = range(n_samples)
    if progress:
        iterator = tqdm(iterator, desc="gibbs")

    for i in iterator:
        for j in range(n_dim):
            x[j] = float(conditional_samplers[j](x))
        samples[i] = x
        log_probs[i] = float(log_target(x)) if log_target is not None else np.nan

    if n_dim == 1:
        samples = samples.ravel()
    return MCMCResult(
        samples=samples,
        log_probs=log_probs,
        acceptance_rate=1.0,
        n_accepted=n_samples,
        n_samples=n_samples,
    )


# ---------------------------------------------------------------------------
# Hamiltonian Monte Carlo
# ---------------------------------------------------------------------------
def _leapfrog(
    x: np.ndarray,
    p: np.ndarray,
    grad_log_target: Callable[[np.ndarray], np.ndarray],
    step_size: float,
    n_steps: int,
):
    """One trajectory of leapfrog integration."""
    x = x.copy()
    p = p.copy()
    p = p + 0.5 * step_size * np.asarray(grad_log_target(x), dtype=float)
    for _ in range(n_steps - 1):
        x = x + step_size * p
        p = p + step_size * np.asarray(grad_log_target(x), dtype=float)
    x = x + step_size * p
    p = p + 0.5 * step_size * np.asarray(grad_log_target(x), dtype=float)
    return x, p


def hamiltonian_monte_carlo(
    log_target: Callable[[np.ndarray], float],
    grad_log_target: Callable[[np.ndarray], np.ndarray],
    x0,
    step_size: float,
    n_leapfrog: int,
    n_samples: int,
    seed: int | None = None,
    progress: bool = True,
) -> MCMCResult:
    """
    Hamiltonian Monte Carlo with leapfrog integration and identity mass.

    Parameters
    ----------
    log_target : callable
        ``log p(x)`` up to a constant.
    grad_log_target : callable
        Gradient of ``log p(x)`` with respect to ``x``.
    step_size : float
        Leapfrog step size ``epsilon``.
    n_leapfrog : int
        Number of leapfrog steps per HMC iteration.
    """
    rng = sampling._rng(seed)
    x = np.atleast_1d(np.asarray(x0, dtype=float)).copy()
    n_dim = x.size

    samples = np.empty((n_samples, n_dim), dtype=float)
    log_probs = np.empty(n_samples, dtype=float)
    n_accepted = 0

    log_target_x = float(log_target(x))

    iterator = range(n_samples)
    if progress:
        iterator = tqdm(iterator, desc="hamiltonian MC")

    for i in iterator:
        p0 = rng.standard_normal(n_dim)
        x_new, p_new = _leapfrog(x, p0, grad_log_target, step_size, n_leapfrog)
        log_target_new = float(log_target(x_new))

        current_H = -log_target_x + 0.5 * float(np.dot(p0, p0))
        proposed_H = -log_target_new + 0.5 * float(np.dot(p_new, p_new))
        log_alpha = current_H - proposed_H

        if np.log(rng.uniform()) < log_alpha:
            x = x_new
            log_target_x = log_target_new
            n_accepted += 1

        samples[i] = x
        log_probs[i] = log_target_x

    if n_dim == 1:
        samples = samples.ravel()
    return MCMCResult(
        samples=samples,
        log_probs=log_probs,
        acceptance_rate=n_accepted / n_samples,
        n_accepted=n_accepted,
        n_samples=n_samples,
    )


# ---------------------------------------------------------------------------
# No-U-Turn Sampler (NUTS)
# ---------------------------------------------------------------------------
_DELTA_MAX = 1000.0  # divergence threshold from Hoffman & Gelman (2014)


def _build_tree(
    x: np.ndarray,
    p: np.ndarray,
    log_slice: float,
    direction: int,
    depth: int,
    step_size: float,
    log_target: Callable[[np.ndarray], float],
    grad_log_target: Callable[[np.ndarray], np.ndarray],
    rng: np.random.Generator,
):
    """
    Recursive tree-builder for NUTS (Algorithm 3, Hoffman & Gelman 2014).

    Returns a tuple of:
      (x_minus, p_minus, x_plus, p_plus, x_proposal, n, stop_flag).
    """
    if depth == 0:
        x_new, p_new = _leapfrog(x, p, grad_log_target, direction * step_size, 1)
        joint = float(log_target(x_new)) - 0.5 * float(np.dot(p_new, p_new))
        n_new = 1 if log_slice <= joint else 0
        s_new = 1 if log_slice < joint + _DELTA_MAX else 0
        return x_new, p_new, x_new, p_new, x_new, n_new, s_new

    # recurse
    x_m, p_m, x_p, p_p, x_prop, n_prop, s_prop = _build_tree(
        x, p, log_slice, direction, depth - 1, step_size,
        log_target, grad_log_target, rng,
    )
    if s_prop == 1:
        if direction == -1:
            x_m, p_m, _, _, x_prop2, n_prop2, s_prop2 = _build_tree(
                x_m, p_m, log_slice, direction, depth - 1, step_size,
                log_target, grad_log_target, rng,
            )
        else:
            _, _, x_p, p_p, x_prop2, n_prop2, s_prop2 = _build_tree(
                x_p, p_p, log_slice, direction, depth - 1, step_size,
                log_target, grad_log_target, rng,
            )
        # progressive sampling: replace x_prop with x_prop2 with prob n_prop2 / (n_prop + n_prop2)
        total = n_prop + n_prop2
        if total > 0 and rng.uniform() < n_prop2 / total:
            x_prop = x_prop2
        n_prop = total
        # U-turn check
        delta_x = x_p - x_m
        no_uturn = (float(np.dot(delta_x, p_m)) >= 0.0
                    and float(np.dot(delta_x, p_p)) >= 0.0)
        s_prop = int(s_prop2 == 1 and no_uturn)

    return x_m, p_m, x_p, p_p, x_prop, n_prop, s_prop


def nuts(
    log_target: Callable[[np.ndarray], float],
    grad_log_target: Callable[[np.ndarray], np.ndarray],
    x0,
    step_size: float,
    n_samples: int,
    max_tree_depth: int = 10,
    seed: int | None = None,
    progress: bool = True,
) -> MCMCResult:
    """
    No-U-Turn Sampler (NUTS), efficient slice-sampling variant
    (Algorithm 3 of Hoffman & Gelman, 2014).

    Eliminates the need to manually tune the trajectory length of HMC by
    extending the leapfrog trajectory in both directions until the path
    starts to double back on itself ("makes a U-turn"). ``step_size``
    still has to be supplied; for serious work, pair this with an
    adaptive step-size routine such as dual averaging.
    """
    rng = sampling._rng(seed)
    x = np.atleast_1d(np.asarray(x0, dtype=float)).copy()
    n_dim = x.size

    samples = np.empty((n_samples, n_dim), dtype=float)
    log_probs = np.empty(n_samples, dtype=float)
    log_target_x = float(log_target(x))

    iterator = range(n_samples)
    if progress:
        iterator = tqdm(iterator, desc="NUTS")

    for i in iterator:
        p0 = rng.standard_normal(n_dim)
        joint = log_target_x - 0.5 * float(np.dot(p0, p0))
        # log-slice variable
        log_slice = joint - rng.exponential(1.0)  # equiv to joint + log(Uniform)

        x_minus = x.copy()
        x_plus = x.copy()
        p_minus = p0.copy()
        p_plus = p0.copy()

        x_proposal = x.copy()
        n_chain = 1
        s_chain = 1
        depth = 0

        while s_chain == 1 and depth < max_tree_depth:
            direction = 1 if rng.uniform() < 0.5 else -1
            if direction == -1:
                x_minus, p_minus, _, _, x_new, n_new, s_new = _build_tree(
                    x_minus, p_minus, log_slice, direction, depth, step_size,
                    log_target, grad_log_target, rng,
                )
            else:
                _, _, x_plus, p_plus, x_new, n_new, s_new = _build_tree(
                    x_plus, p_plus, log_slice, direction, depth, step_size,
                    log_target, grad_log_target, rng,
                )

            if s_new == 1 and n_chain > 0 and rng.uniform() < n_new / n_chain:
                x_proposal = x_new

            n_chain += n_new
            delta_x = x_plus - x_minus
            no_uturn = (float(np.dot(delta_x, p_minus)) >= 0.0
                        and float(np.dot(delta_x, p_plus)) >= 0.0)
            s_chain = int(s_new == 1 and no_uturn)
            depth += 1

        x = x_proposal
        log_target_x = float(log_target(x))
        samples[i] = x
        log_probs[i] = log_target_x

    # NUTS always "accepts" in the slice-sampling sense, so report the
    # fraction of iterations that moved away from the starting point.
    moved = int(np.sum(np.any(np.diff(samples, axis=0) != 0, axis=1))) + 1
    acc = moved / n_samples

    if n_dim == 1:
        samples = samples.ravel()
    return MCMCResult(
        samples=samples,
        log_probs=log_probs,
        acceptance_rate=acc,
        n_accepted=moved,
        n_samples=n_samples,
    )


# ---------------------------------------------------------------------------
# Post-processing helpers
# ---------------------------------------------------------------------------
def burn_in(samples, n_burn: int):
    """Drop the first ``n_burn`` samples of a chain."""
    samples = np.asarray(samples)
    if n_burn < 0 or n_burn >= len(samples):
        raise ValueError("n_burn must satisfy 0 <= n_burn < len(samples).")
    return samples[n_burn:]


def thin(samples, k: int):
    """Keep every ``k``-th sample, starting from the first."""
    if k < 1:
        raise ValueError("k must be a positive integer.")
    return np.asarray(samples)[::k]


def acceptance_rate(accepts) -> float:
    """Fraction of accepted moves in a 0/1 (or boolean) array."""
    return float(np.mean(np.asarray(accepts, dtype=float)))


def geweke_diagnostic(
    samples,
    first_frac: float = 0.1,
    last_frac: float = 0.5,
) -> float:
    """
    Geweke convergence diagnostic.

    Computes a z-score comparing the mean of the first ``first_frac`` of
    the chain to the mean of the last ``last_frac``. Under convergence
    these means should agree, so |z| < ~2 is consistent with a stationary
    chain.
    """
    s = np.asarray(samples, dtype=float)
    n = s.shape[0]
    n_first = int(first_frac * n)
    n_last = int(last_frac * n)
    if n_first < 2 or n_last < 2:
        raise ValueError("Chain too short for the requested fractions.")
    first = s[:n_first]
    last = s[-n_last:]

    if s.ndim == 1:
        m1, m2 = first.mean(), last.mean()
        v1 = first.var(ddof=1) / n_first
        v2 = last.var(ddof=1) / n_last
        return float((m1 - m2) / np.sqrt(v1 + v2))

    # multivariate: return per-dimension z-scores
    m1 = first.mean(axis=0)
    m2 = last.mean(axis=0)
    v1 = first.var(axis=0, ddof=1) / n_first
    v2 = last.var(axis=0, ddof=1) / n_last
    return (m1 - m2) / np.sqrt(v1 + v2)
