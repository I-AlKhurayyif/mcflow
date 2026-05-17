"""
distributions.py
================

Analytical probability density functions (PDFs), cumulative distribution
functions (CDFs), and inverse CDFs (quantile functions) for the
distributions supported by ``mcflow``.

While ``sampling.py`` is responsible for *generating* random draws, this
module is responsible for *evaluating* densities and distribution
functions. These are needed for importance sampling, weighting, and
analytical comparisons.
"""

from __future__ import annotations

import numpy as np
from scipy.special import erf, erfinv


# ---------------------------------------------------------------------------
# Uniform distribution
# ---------------------------------------------------------------------------
def uniform_pdf(x, low: float, high: float):
    """Probability density of the Uniform(low, high) distribution."""
    x = np.asarray(x, dtype=float)
    pdf = np.where((x >= low) & (x <= high), 1.0 / (high - low), 0.0)
    return pdf


def uniform_cdf(x, low: float, high: float):
    """Cumulative distribution of the Uniform(low, high) distribution."""
    x = np.asarray(x, dtype=float)
    cdf = np.clip((x - low) / (high - low), 0.0, 1.0)
    return cdf


def uniform_inverse_cdf(u, low: float, high: float):
    """Quantile function of Uniform(low, high). Used in inverse-transform sampling."""
    u = np.asarray(u, dtype=float)
    return low + u * (high - low)


# ---------------------------------------------------------------------------
# Normal distribution
# ---------------------------------------------------------------------------
def normal_pdf(x, mean: float = 0.0, std: float = 1.0):
    """Probability density of the Normal(mean, std) distribution."""
    x = np.asarray(x, dtype=float)
    z = (x - mean) / std
    return np.exp(-0.5 * z**2) / (std * np.sqrt(2.0 * np.pi))


def normal_cdf(x, mean: float = 0.0, std: float = 1.0):
    """Cumulative distribution of Normal(mean, std), based on the error function."""
    x = np.asarray(x, dtype=float)
    return 0.5 * (1.0 + erf((x - mean) / (std * np.sqrt(2.0))))


def normal_inverse_cdf(u, mean: float = 0.0, std: float = 1.0):
    """Quantile function of Normal(mean, std)."""
    u = np.asarray(u, dtype=float)
    return mean + std * np.sqrt(2.0) * erfinv(2.0 * u - 1.0)


def standard_normal_pdf(z):
    """Standard Normal PDF, phi(z)."""
    return normal_pdf(z, mean=0.0, std=1.0)


def standard_normal_cdf(z):
    """Standard Normal CDF, Phi(z)."""
    return normal_cdf(z, mean=0.0, std=1.0)


# ---------------------------------------------------------------------------
# Triangular distribution
# ---------------------------------------------------------------------------
def triangular_pdf(x, low: float, mode: float, high: float):
    """Probability density of the Triangular(low, mode, high) distribution."""
    x = np.asarray(x, dtype=float)
    pdf = np.zeros_like(x)

    left = (x >= low) & (x <= mode)
    right = (x > mode) & (x <= high)

    pdf[left] = 2.0 * (x[left] - low) / ((high - low) * (mode - low))
    pdf[right] = 2.0 * (high - x[right]) / ((high - low) * (high - mode))
    return pdf


def triangular_cdf(x, low: float, mode: float, high: float):
    """Cumulative distribution of the Triangular(low, mode, high) distribution."""
    x = np.asarray(x, dtype=float)
    cdf = np.zeros_like(x)

    left = (x >= low) & (x <= mode)
    right = (x > mode) & (x <= high)
    above = x > high

    cdf[left] = (x[left] - low) ** 2 / ((high - low) * (mode - low))
    cdf[right] = 1.0 - (high - x[right]) ** 2 / ((high - low) * (high - mode))
    cdf[above] = 1.0
    return cdf


def triangular_inverse_cdf(u, low: float, mode: float, high: float):
    """Quantile function of Triangular(low, mode, high)."""
    u = np.asarray(u, dtype=float)
    fc = (mode - low) / (high - low)
    left = u < fc
    out = np.empty_like(u)
    out[left] = low + np.sqrt(u[left] * (high - low) * (mode - low))
    out[~left] = high - np.sqrt((1.0 - u[~left]) * (high - low) * (high - mode))
    return out


# ---------------------------------------------------------------------------
# Exponential distribution
# ---------------------------------------------------------------------------
def exponential_pdf(x, rate: float = 1.0):
    """PDF of Exponential(rate)."""
    x = np.asarray(x, dtype=float)
    return np.where(x >= 0.0, rate * np.exp(-rate * x), 0.0)


def exponential_cdf(x, rate: float = 1.0):
    """CDF of Exponential(rate)."""
    x = np.asarray(x, dtype=float)
    return np.where(x >= 0.0, 1.0 - np.exp(-rate * x), 0.0)


def exponential_inverse_cdf(u, rate: float = 1.0):
    """Quantile function of Exponential(rate)."""
    u = np.asarray(u, dtype=float)
    return -np.log1p(-u) / rate


# ---------------------------------------------------------------------------
# Lognormal distribution
# ---------------------------------------------------------------------------
def lognormal_pdf(x, mean: float = 0.0, std: float = 1.0):
    """PDF of the Lognormal distribution with underlying Normal(mean, std)."""
    x = np.asarray(x, dtype=float)
    out = np.zeros_like(x)
    positive = x > 0
    z = (np.log(x[positive]) - mean) / std
    out[positive] = np.exp(-0.5 * z**2) / (x[positive] * std * np.sqrt(2.0 * np.pi))
    return out


def lognormal_cdf(x, mean: float = 0.0, std: float = 1.0):
    """CDF of the Lognormal distribution with underlying Normal(mean, std)."""
    x = np.asarray(x, dtype=float)
    out = np.zeros_like(x)
    positive = x > 0
    out[positive] = 0.5 * (
        1.0 + erf((np.log(x[positive]) - mean) / (std * np.sqrt(2.0)))
    )
    return out
