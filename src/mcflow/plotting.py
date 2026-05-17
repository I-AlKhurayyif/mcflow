"""
plotting.py
===========

Visualization utilities for ``mcflow``.

Each function follows the same convention: if no axis is supplied a new
figure is created, the plot is rendered, and the ``matplotlib`` Axes
object is returned. This keeps the API composable for users that want to
build dashboards or multi-panel figures.
"""

from __future__ import annotations

from typing import Callable, Optional

import matplotlib.pyplot as plt
import numpy as np

from . import aggregation, estimators


# ---------------------------------------------------------------------------
# Distribution plots
# ---------------------------------------------------------------------------
def plot_histogram(
    samples,
    bins: int = 30,
    density: bool = True,
    ax: Optional[plt.Axes] = None,
    title: str = "Sample histogram",
):
    """Plot a histogram of ``samples``."""
    samples = np.asarray(samples, dtype=float)
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 4))
    ax.hist(samples, bins=bins, density=density, alpha=0.7, edgecolor="black")
    ax.set_xlabel("x")
    ax.set_ylabel("density" if density else "count")
    ax.set_title(title)
    ax.grid(alpha=0.3)
    return ax


def plot_pdf_overlay(
    samples,
    pdf_func: Callable[[np.ndarray], np.ndarray],
    bins: int = 30,
    ax: Optional[plt.Axes] = None,
    title: str = "Histogram with PDF overlay",
):
    """Histogram of samples with an analytical PDF curve overlaid."""
    samples = np.asarray(samples, dtype=float)
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 4))
    ax.hist(samples, bins=bins, density=True, alpha=0.6, edgecolor="black", label="samples")
    x_grid = np.linspace(samples.min(), samples.max(), 400)
    ax.plot(x_grid, pdf_func(x_grid), "r-", linewidth=2, label="PDF")
    ax.set_xlabel("x")
    ax.set_ylabel("density")
    ax.set_title(title)
    ax.legend()
    ax.grid(alpha=0.3)
    return ax


def plot_empirical_cdf(
    samples,
    ax: Optional[plt.Axes] = None,
    title: str = "Empirical CDF",
):
    """Plot the empirical cumulative distribution of ``samples``."""
    samples = np.sort(np.asarray(samples, dtype=float))
    n = samples.size
    y = np.arange(1, n + 1) / n
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 4))
    ax.step(samples, y, where="post")
    ax.set_xlabel("x")
    ax.set_ylabel("F_N(x)")
    ax.set_title(title)
    ax.grid(alpha=0.3)
    return ax


# ---------------------------------------------------------------------------
# Convergence plots
# ---------------------------------------------------------------------------
def plot_running_mean(
    samples,
    true_value: Optional[float] = None,
    ax: Optional[plt.Axes] = None,
    title: str = "Running mean",
):
    """Plot the running mean against the sample index."""
    samples = np.asarray(samples, dtype=float)
    mu_n = estimators.running_mean(samples)
    n = np.arange(1, samples.size + 1)
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 4))
    ax.plot(n, mu_n, label="running mean")
    if true_value is not None:
        ax.axhline(true_value, color="red", linestyle="--", label="true value")
    ax.set_xlabel("n")
    ax.set_ylabel("estimate")
    ax.set_title(title)
    ax.legend()
    ax.grid(alpha=0.3)
    return ax


def plot_running_standard_error(
    samples,
    ax: Optional[plt.Axes] = None,
    title: str = "Running standard error",
):
    """Plot the running standard error against the sample index, log-log."""
    samples = np.asarray(samples, dtype=float)
    se_n = estimators.running_standard_error(samples)
    n = np.arange(1, samples.size + 1)
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 4))
    ax.loglog(n[1:], se_n[1:], label="running SE")
    # reference 1/sqrt(n) slope anchored to the first valid SE
    if not np.isnan(se_n[1]):
        ref = se_n[1] * np.sqrt(n[1]) / np.sqrt(n[1:])
        ax.loglog(n[1:], ref, "k--", alpha=0.6, label="O(1/sqrt(n))")
    ax.set_xlabel("n")
    ax.set_ylabel("standard error")
    ax.set_title(title)
    ax.legend()
    ax.grid(alpha=0.3, which="both")
    return ax


def plot_convergence(
    samples,
    true_value: Optional[float] = None,
    level: float = 0.95,
    ax: Optional[plt.Axes] = None,
    title: str = "Convergence with confidence band",
):
    """
    Convergence plot showing the running mean with a normal-approximation
    confidence band.
    """
    samples = np.asarray(samples, dtype=float)
    mu_n = estimators.running_mean(samples)
    se_n = estimators.running_standard_error(samples)
    n = np.arange(1, samples.size + 1)

    alpha = 1.0 - level
    from scipy.special import erfinv
    z = float(np.sqrt(2.0) * erfinv(1.0 - alpha))

    if ax is None:
        _, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(n, mu_n, label="running mean")
    ax.fill_between(
        n,
        mu_n - z * se_n,
        mu_n + z * se_n,
        alpha=0.25,
        label=f"{int(level * 100)}% CI",
    )
    if true_value is not None:
        ax.axhline(true_value, color="red", linestyle="--", label="true value")
    ax.set_xlabel("n")
    ax.set_ylabel("estimate")
    ax.set_title(title)
    ax.legend()
    ax.grid(alpha=0.3)
    return ax


def plot_confidence_band(samples, level: float = 0.95, ax: Optional[plt.Axes] = None):
    """Alias for ``plot_convergence`` that focuses on the band itself."""
    return plot_convergence(samples, level=level, ax=ax, title="Confidence band")


# ---------------------------------------------------------------------------
# Importance-sampling and correlation plots
# ---------------------------------------------------------------------------
def plot_weights(
    weights,
    bins: int = 40,
    ax: Optional[plt.Axes] = None,
    title: str = "Importance weights",
):
    """Histogram of importance weights, useful for diagnosing IS quality."""
    weights = np.asarray(weights, dtype=float)
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 4))
    ax.hist(weights, bins=bins, edgecolor="black", alpha=0.7)
    ax.set_xlabel("weight")
    ax.set_ylabel("count")
    ax.set_title(title)
    ax.grid(alpha=0.3)
    return ax


def plot_autocorrelation(
    samples,
    max_lag: int = 50,
    ax: Optional[plt.Axes] = None,
    title: str = "Autocorrelation",
):
    """Plot the sample autocorrelation up to ``max_lag``."""
    rho = aggregation.autocorrelation_function(samples, max_lag=max_lag)
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 4))
    lags = np.arange(rho.size)
    ax.vlines(lags, 0, rho, linewidth=2)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xlabel("lag")
    ax.set_ylabel("rho_k")
    ax.set_title(title)
    ax.grid(alpha=0.3)
    return ax


def plot_scatter_2d(
    samples_x,
    samples_y,
    ax: Optional[plt.Axes] = None,
    title: str = "Scatter",
):
    """Scatter plot of two equal-length sample arrays."""
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(samples_x, samples_y, alpha=0.4, s=12)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(title)
    ax.grid(alpha=0.3)
    return ax
