"""
mcflow
======

A small, dependable library for Monte Carlo methods.

``mcflow`` is organized into focused modules:

* ``distributions`` - analytical PDFs, CDFs, inverse CDFs.
* ``sampling`` - random sample generation and variance-reduction designs.
* ``estimators`` - means, variances, standard errors, confidence intervals.
* ``aggregation`` - summary statistics and convergence diagnostics.
* ``integration`` - Monte Carlo integration (1-D, n-D, importance,
  antithetic, stratified, control variates).
* ``importance`` - importance sampling and self-normalized variants.
* ``simulation`` - high-level orchestration (run, replicate, bootstrap).
* ``mcmc`` - Markov Chain Monte Carlo (MH, RWM, Gibbs, HMC, NUTS).
* ``bmc`` - Bayesian Monte Carlo: posterior summaries, posterior
  predictive, marginal likelihood / Bayes factors, GP-based Bayesian
  quadrature.
* ``plotting`` - visual diagnostics built on top of matplotlib.

The most commonly used functions are re-exported below so they can be
accessed as ``mcflow.mc_mean(...)`` instead of
``mcflow.estimators.mc_mean(...)``.
"""

from __future__ import annotations

__version__ = "0.1.0"
__author__ = "Ibrahim A. Al Khurayyif"

# ---------------------------------------------------------------------------
# Submodule access
# ---------------------------------------------------------------------------
from . import (
    aggregation,
    bmc,
    distributions,
    estimators,
    importance,
    integration,
    mcmc,
    plotting,
    sampling,
    simulation,
)

# ---------------------------------------------------------------------------
# Top-level re-exports
# ---------------------------------------------------------------------------
from .estimators import (
    mc_mean,
    weighted_mean,
    sample_variance,
    standard_deviation,
    standard_error,
    confidence_interval,
    t_confidence_interval,
    percentile_interval,
    running_mean,
    running_standard_error,
    running_variance,
    bias,
    mean_squared_error,
    root_mean_squared_error,
    coefficient_of_variation,
    skewness,
    kurtosis,
    covariance,
    correlation,
)

from .sampling import (
    set_seed,
    sample_uniform,
    sample_normal,
    sample_standard_normal,
    sample_triangular,
    sample_exponential,
    sample_bernoulli,
    inverse_transform_sample,
    box_muller_transform,
    rejection_sample,
    latin_hypercube_sample,
    stratified_sample,
    antithetic_sample,
)

from .simulation import (
    run_simulation,
    run_replicated_simulation,
    simulate_expectation,
    simulate_probability,
    simulate_variance,
    batch_means,
    bootstrap_resample,
    bootstrap_confidence_interval,
    SimulationResult,
)

from .integration import (
    mc_integrate_1d,
    mc_integrate_nd,
    mc_integrate_importance,
    mc_integrate_antithetic,
    mc_integrate_stratified,
    mc_integrate_control_variate,
    integration_error,
    IntegrationResult,
)

from .importance import (
    importance_sampling,
    self_normalized_importance_sampling,
    importance_weights,
    normalize_weights,
    effective_sample_size,
    variance_of_is_estimator,
    kl_divergence,
    ImportanceResult,
)

from .aggregation import (
    summarize,
    aggregate_replications,
    convergence_diagnostic,
    relative_error,
    gelman_rubin_diagnostic,
    autocorrelation,
    autocorrelation_function,
    integrated_autocorrelation_time,
    effective_sample_size_correlated,
)

from .plotting import (
    plot_histogram,
    plot_pdf_overlay,
    plot_empirical_cdf,
    plot_running_mean,
    plot_running_standard_error,
    plot_convergence,
    plot_confidence_band,
    plot_weights,
    plot_autocorrelation,
    plot_scatter_2d,
)

from .mcmc import (
    metropolis_hastings,
    random_walk_metropolis,
    gibbs_sampler,
    hamiltonian_monte_carlo,
    nuts,
    burn_in,
    thin,
    acceptance_rate,
    geweke_diagnostic,
    MCMCResult,
)

from .bmc import (
    log_posterior,
    posterior_mean,
    posterior_variance,
    credible_interval,
    highest_posterior_density,
    posterior_predictive_sample,
    posterior_predictive_check,
    marginal_likelihood_is,
    harmonic_mean_estimator,
    bayes_factor,
    rbf_kernel,
    bayesian_quadrature,
    BayesianQuadratureResult,
)

__all__ = [
    # metadata
    "__version__",
    "__author__",
    # submodules
    "aggregation",
    "bmc",
    "distributions",
    "estimators",
    "importance",
    "integration",
    "mcmc",
    "plotting",
    "sampling",
    "simulation",
    # estimators
    "mc_mean",
    "weighted_mean",
    "sample_variance",
    "standard_deviation",
    "standard_error",
    "confidence_interval",
    "t_confidence_interval",
    "percentile_interval",
    "running_mean",
    "running_standard_error",
    "running_variance",
    "bias",
    "mean_squared_error",
    "root_mean_squared_error",
    "coefficient_of_variation",
    "skewness",
    "kurtosis",
    "covariance",
    "correlation",
    # sampling
    "set_seed",
    "sample_uniform",
    "sample_normal",
    "sample_standard_normal",
    "sample_triangular",
    "sample_exponential",
    "sample_bernoulli",
    "inverse_transform_sample",
    "box_muller_transform",
    "rejection_sample",
    "latin_hypercube_sample",
    "stratified_sample",
    "antithetic_sample",
    # simulation
    "run_simulation",
    "run_replicated_simulation",
    "simulate_expectation",
    "simulate_probability",
    "simulate_variance",
    "batch_means",
    "bootstrap_resample",
    "bootstrap_confidence_interval",
    "SimulationResult",
    # integration
    "mc_integrate_1d",
    "mc_integrate_nd",
    "mc_integrate_importance",
    "mc_integrate_antithetic",
    "mc_integrate_stratified",
    "mc_integrate_control_variate",
    "integration_error",
    "IntegrationResult",
    # importance
    "importance_sampling",
    "self_normalized_importance_sampling",
    "importance_weights",
    "normalize_weights",
    "effective_sample_size",
    "variance_of_is_estimator",
    "kl_divergence",
    "ImportanceResult",
    # aggregation
    "summarize",
    "aggregate_replications",
    "convergence_diagnostic",
    "relative_error",
    "gelman_rubin_diagnostic",
    "autocorrelation",
    "autocorrelation_function",
    "integrated_autocorrelation_time",
    "effective_sample_size_correlated",
    # plotting
    "plot_histogram",
    "plot_pdf_overlay",
    "plot_empirical_cdf",
    "plot_running_mean",
    "plot_running_standard_error",
    "plot_convergence",
    "plot_confidence_band",
    "plot_weights",
    "plot_autocorrelation",
    "plot_scatter_2d",
    # mcmc
    "metropolis_hastings",
    "random_walk_metropolis",
    "gibbs_sampler",
    "hamiltonian_monte_carlo",
    "nuts",
    "burn_in",
    "thin",
    "acceptance_rate",
    "geweke_diagnostic",
    "MCMCResult",
    # bmc
    "log_posterior",
    "posterior_mean",
    "posterior_variance",
    "credible_interval",
    "highest_posterior_density",
    "posterior_predictive_sample",
    "posterior_predictive_check",
    "marginal_likelihood_is",
    "harmonic_mean_estimator",
    "bayes_factor",
    "rbf_kernel",
    "bayesian_quadrature",
    "BayesianQuadratureResult",
]
