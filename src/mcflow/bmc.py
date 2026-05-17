"""
bmc.py
======

Bayesian Monte Carlo for ``mcflow``.

This module covers two distinct but related uses of the term "BMC":

1. **Bayesian Monte Carlo workflow.** Posterior summaries (mean,
   variance, credible intervals, HPD intervals), posterior predictive
   sampling, posterior predictive checks, marginal-likelihood estimation
   (importance-sampling and harmonic-mean estimators), and Bayes factors.

2. **Bayesian Quadrature** (O'Hagan, 1991). Treat the integrand as a
   Gaussian process and infer the value of an integral against a known
   measure. This gives both a point estimate *and* a posterior variance
   over the integral itself, often with much better sample efficiency
   than ordinary Monte Carlo on smooth integrands.

The Bayesian-quadrature implementation provides closed-form integrals
for the common case of an RBF kernel paired with a Gaussian measure.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Tuple

import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.special import logsumexp
from tqdm.auto import tqdm

from . import estimators


# ---------------------------------------------------------------------------
# Posterior summaries
# ---------------------------------------------------------------------------
def log_posterior(
    log_prior: Callable[[np.ndarray], float],
    log_likelihood: Callable[[np.ndarray], float],
) -> Callable[[np.ndarray], float]:
    """
    Build a log unnormalized posterior ``log p(theta | D) = log p(D|theta)
    + log p(theta)`` (up to an additive constant) from a prior and a
    likelihood.

    The result is a single callable suitable for use as the ``log_target``
    argument of any sampler in :mod:`mcflow.mcmc`.
    """
    def _log_post(theta):
        return float(log_prior(theta)) + float(log_likelihood(theta))
    return _log_post


def posterior_mean(samples) -> float | np.ndarray:
    """Posterior mean (per coordinate if ``samples`` is 2-D)."""
    samples = np.asarray(samples, dtype=float)
    if samples.ndim == 1:
        return estimators.mc_mean(samples)
    return np.mean(samples, axis=0)


def posterior_variance(samples) -> float | np.ndarray:
    """Posterior variance (per coordinate if ``samples`` is 2-D)."""
    samples = np.asarray(samples, dtype=float)
    if samples.ndim == 1:
        return estimators.sample_variance(samples)
    return np.var(samples, axis=0, ddof=1)


def credible_interval(samples, level: float = 0.95):
    """
    Equal-tailed (percentile) credible interval at confidence ``level``.

    For 2-D samples a (n_dim, 2) array of intervals is returned.
    """
    samples = np.asarray(samples, dtype=float)
    alpha = 1.0 - level
    lo_pct = 100.0 * alpha / 2.0
    hi_pct = 100.0 * (1.0 - alpha / 2.0)
    if samples.ndim == 1:
        lo, hi = np.percentile(samples, [lo_pct, hi_pct])
        return float(lo), float(hi)
    lo = np.percentile(samples, lo_pct, axis=0)
    hi = np.percentile(samples, hi_pct, axis=0)
    return np.stack([lo, hi], axis=-1)


def highest_posterior_density(samples, level: float = 0.95):
    """
    Highest Posterior Density (HPD) interval.

    Returns the shortest interval containing ``level`` of the posterior
    mass, computed by sliding a window of the required length across the
    sorted samples. For 2-D samples HPD is computed per coordinate.
    """
    samples = np.asarray(samples, dtype=float)
    if samples.ndim == 1:
        return _hpd_1d(samples, level)
    out = np.empty((samples.shape[1], 2), dtype=float)
    for j in range(samples.shape[1]):
        out[j] = _hpd_1d(samples[:, j], level)
    return out


def _hpd_1d(samples: np.ndarray, level: float) -> Tuple[float, float]:
    sorted_samples = np.sort(samples)
    n = sorted_samples.size
    window = int(np.floor(level * n))
    if window < 1 or window >= n:
        raise ValueError("level produces an empty or full window; check inputs.")
    widths = sorted_samples[window:] - sorted_samples[: n - window]
    i = int(np.argmin(widths))
    return float(sorted_samples[i]), float(sorted_samples[i + window])


# ---------------------------------------------------------------------------
# Posterior predictive
# ---------------------------------------------------------------------------
def posterior_predictive_sample(
    posterior_samples: np.ndarray,
    likelihood_sampler: Callable[[np.ndarray], np.ndarray],
    n_per_theta: int = 1,
    progress: bool = True,
) -> np.ndarray:
    """
    Draw from the posterior predictive distribution.

    For each posterior sample ``theta_i`` we draw ``n_per_theta`` samples
    from the likelihood ``p(y | theta_i)`` via ``likelihood_sampler(theta_i)``.

    ``likelihood_sampler(theta)`` must return either a scalar or a 1-D
    array. The returned array is flattened across posterior samples and
    predictive draws.
    """
    posterior_samples = np.asarray(posterior_samples)
    n = posterior_samples.shape[0]

    iterator = range(n)
    if progress:
        iterator = tqdm(iterator, desc="posterior predictive")

    drawn = []
    for i in iterator:
        theta = posterior_samples[i]
        for _ in range(n_per_theta):
            y = np.atleast_1d(np.asarray(likelihood_sampler(theta), dtype=float))
            drawn.append(y)
    return np.concatenate(drawn)


def posterior_predictive_check(
    predictive_samples: np.ndarray,
    observed,
    statistic: Callable[[np.ndarray], float] = np.mean,
) -> Tuple[float, float, float]:
    """
    Posterior predictive check via a test statistic.

    Computes the Bayesian p-value
    ``P(T(y_rep) >= T(y_obs) | y_obs)``, the test statistic on the
    observed data, and the mean of the test statistic on predictive
    draws.

    For ``predictive_samples`` shaped as a flat 1-D array, the statistic
    is applied to the whole array of predictive draws as a single
    replicate (useful when each predictive draw is a scalar). For
    replicate-style 2-D input of shape ``(n_replicates, n_obs)`` the
    statistic is applied per replicate.
    """
    predictive_samples = np.asarray(predictive_samples, dtype=float)
    observed = np.atleast_1d(np.asarray(observed, dtype=float))
    t_obs = float(statistic(observed))

    if predictive_samples.ndim == 1:
        # treat each entry as one scalar replicate
        t_rep = predictive_samples
    else:
        t_rep = np.array([statistic(row) for row in predictive_samples])

    p_value = float(np.mean(t_rep >= t_obs))
    return p_value, t_obs, float(np.mean(t_rep))


# ---------------------------------------------------------------------------
# Marginal likelihood and Bayes factor
# ---------------------------------------------------------------------------
def marginal_likelihood_is(
    log_likelihood: Callable[[np.ndarray], float],
    prior_sampler: Callable[[int], np.ndarray],
    n_samples: int,
    progress: bool = True,
) -> float:
    """
    Estimate the log marginal likelihood ``log p(D) = log E_prior[p(D|theta)]``
    by drawing ``theta_i`` from the prior and averaging the likelihood.

    Returned as a *log* quantity for numerical stability:
        log p(D) ≈ logsumexp(log_lik_i) - log N

    This works well when the prior overlaps the high-likelihood region.
    For peaked posteriors, a more concentrated proposal (or bridge
    sampling) will be much more efficient.
    """
    thetas = np.asarray(prior_sampler(n_samples))
    log_lik = np.empty(n_samples, dtype=float)

    iterator = range(n_samples)
    if progress:
        iterator = tqdm(iterator, desc="marginal likelihood (IS)")
    for i in iterator:
        log_lik[i] = float(log_likelihood(thetas[i]))

    return float(logsumexp(log_lik) - np.log(n_samples))


def harmonic_mean_estimator(log_likelihood_at_posterior: np.ndarray) -> float:
    """
    Newton-Raftery harmonic-mean estimator of ``log p(D)``.

    Given ``log p(D | theta_i)`` evaluated at posterior samples,
        log p(D) ≈ log N - logsumexp(-log_lik_i)

    .. warning::
        The harmonic-mean estimator is consistent but has notoriously
        high (sometimes infinite) variance because a single small
        likelihood blows up its reciprocal. It is included here for
        completeness and quick checks; prefer
        :func:`marginal_likelihood_is` or bridge sampling for serious use.
    """
    log_lik = np.asarray(log_likelihood_at_posterior, dtype=float)
    n = log_lik.size
    return float(np.log(n) - logsumexp(-log_lik))


def bayes_factor(log_marginal_1: float, log_marginal_2: float) -> float:
    """
    Bayes factor BF_12 = p(D | M_1) / p(D | M_2), computed from log
    marginal likelihoods. Returned on the linear (not log) scale.
    """
    return float(np.exp(log_marginal_1 - log_marginal_2))


# ---------------------------------------------------------------------------
# Bayesian Quadrature (GP-based)
# ---------------------------------------------------------------------------
@dataclass
class BayesianQuadratureResult:
    """Output of GP-based Bayesian quadrature."""
    mean: float
    variance: float
    standard_error: float
    n_train: int

    def __repr__(self) -> str:
        return (
            f"BayesianQuadratureResult(mean={self.mean:.6g}, "
            f"se={self.standard_error:.3g}, n_train={self.n_train})"
        )


def rbf_kernel(
    x1: np.ndarray,
    x2: np.ndarray,
    length_scale: float = 1.0,
    variance: float = 1.0,
) -> np.ndarray:
    """
    Squared-exponential (RBF) kernel.

    ``k(x, x') = variance * exp(-0.5 * ||x - x'||^2 / length_scale^2)``.

    Accepts 1-D or 2-D arrays. For ``x1`` of shape ``(m, d)`` and ``x2``
    of shape ``(n, d)``, returns an ``(m, n)`` matrix.
    """
    x1 = np.atleast_2d(np.asarray(x1, dtype=float))
    x2 = np.atleast_2d(np.asarray(x2, dtype=float))
    if x1.shape[0] == 1 and x1.shape[1] != x2.shape[1]:
        x1 = x1.T
    if x2.shape[0] == 1 and x2.shape[1] != x1.shape[1]:
        x2 = x2.T
    diff = x1[:, None, :] - x2[None, :, :]
    sqdist = np.sum(diff**2, axis=-1)
    return variance * np.exp(-0.5 * sqdist / length_scale**2)


def bayesian_quadrature(
    func_values: np.ndarray,
    x_train: np.ndarray,
    measure_mean: np.ndarray,
    measure_cov: np.ndarray,
    length_scale: float = 1.0,
    variance: float = 1.0,
    jitter: float = 1e-8,
) -> BayesianQuadratureResult:
    """
    Bayesian Quadrature with an RBF kernel against a Gaussian measure.

    Estimates ``I = ∫ f(x) π(x) dx`` where ``π = N(measure_mean,
    measure_cov)``, by fitting a GP prior with an RBF kernel to the
    training data ``(x_train, func_values)`` and computing the closed-form
    posterior mean and variance of the integral.

    Parameters
    ----------
    func_values : array, shape (n_train,)
        ``f`` evaluated at ``x_train``.
    x_train : array, shape (n_train, d) or (n_train,) when ``d == 1``
        Training inputs.
    measure_mean : array, shape (d,) or scalar
        Mean of the Gaussian measure ``π``.
    measure_cov : array, shape (d, d) or scalar
        Covariance of the Gaussian measure ``π``.
    length_scale : float
        Common length scale ``l`` of the isotropic RBF kernel.
    variance : float
        Output variance ``sigma^2`` of the RBF kernel.
    jitter : float
        Diagonal regularization added to ``K`` for numerical stability.

    Returns
    -------
    BayesianQuadratureResult
        Carries posterior mean, posterior variance, and standard error
        of the integral.

    Notes
    -----
    Using the standard closed-form identities for an RBF kernel under a
    Gaussian measure ``π = N(b, B)`` with isotropic length scale ``l``
    (so ``L = l^2 I``):

        z_i = ∫ k(x, x_i) π(x) dx
            = σ² |B/l² + I|^{-1/2}
              exp(-½ (x_i − b)ᵀ (B + l² I)^{-1} (x_i − b))

        ∫∫ k(x, x') π(x) π(x') dx dx' = σ² |2 B/l² + I|^{-1/2}

    Then the posterior mean and variance of the integral are:

        E[I | data] = zᵀ K⁻¹ y
        Var[I | data] = (double integral) − zᵀ K⁻¹ z
    """
    x_train = np.atleast_2d(np.asarray(x_train, dtype=float))
    if x_train.shape[0] == 1:
        x_train = x_train.T  # treat 1-D input as a column of points
    y = np.asarray(func_values, dtype=float).ravel()
    n, d = x_train.shape

    b = np.atleast_1d(np.asarray(measure_mean, dtype=float))
    if b.size == 1 and d > 1:
        b = np.full(d, float(b.item()))
    elif b.size != d:
        raise ValueError(f"measure_mean has size {b.size}, expected {d}.")

    measure_cov_arr = np.asarray(measure_cov, dtype=float)
    if measure_cov_arr.ndim == 0 or measure_cov_arr.size == 1:
        B = float(measure_cov_arr.reshape(-1)[0]) * np.eye(d)
    else:
        B = np.atleast_2d(measure_cov_arr)
        if B.shape != (d, d):
            raise ValueError(
                f"measure_cov has shape {B.shape}, expected ({d}, {d})."
            )

    l2 = float(length_scale) ** 2
    sigma2 = float(variance)
    eye_d = np.eye(d)

    # K: train-train kernel + jitter for stability.
    K = rbf_kernel(x_train, x_train, length_scale, variance)
    K = K + jitter * np.eye(n)

    # z_i = sigma^2 * |B/l^2 + I|^{-1/2} * exp(-1/2 (x_i - b)^T (B + l^2 I)^{-1} (x_i - b))
    M = B / l2 + eye_d
    det_M = float(np.linalg.det(M))
    inv_B_plus_lI = np.linalg.inv(B + l2 * eye_d)
    diff = x_train - b  # (n, d)
    quad = np.einsum("ij,jk,ik->i", diff, inv_B_plus_lI, diff)
    z = sigma2 * (det_M ** -0.5) * np.exp(-0.5 * quad)

    # Double integral: sigma^2 * |2B/l^2 + I|^{-1/2}
    det_2M = float(np.linalg.det(2.0 * B / l2 + eye_d))
    double_int = sigma2 * (det_2M ** -0.5)

    # Posterior mean and variance via Cholesky.
    cho = cho_factor(K, lower=True)
    K_inv_y = cho_solve(cho, y)
    K_inv_z = cho_solve(cho, z)

    integral_mean = float(z @ K_inv_y)
    integral_var = float(double_int - z @ K_inv_z)
    if integral_var < 0:
        # numerical underflow; clamp at 0
        integral_var = 0.0

    return BayesianQuadratureResult(
        mean=integral_mean,
        variance=integral_var,
        standard_error=float(np.sqrt(integral_var)),
        n_train=n,
    )
