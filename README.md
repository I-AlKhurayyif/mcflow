<p align="center">
  <img src="https://raw.githubusercontent.com/I-AlKhurayyif/mcflow/main/assets/mcflow_logo.png" alt="mcflow logo" width="520"/>
</p>

# mcflow

**A small, dependable library for Monte Carlo methods, MCMC, and Bayesian Monte Carlo.**

`mcflow` provides the building blocks every Monte Carlo project actually needs — samplers, estimators, confidence intervals, integrators, importance sampling, MCMC, Bayesian inference, and visualization — in one focused package with a clean, learnable API.

---

## Features

- **Sampling** — uniform, normal, triangular, exponential, Bernoulli, plus inverse-transform, Box–Muller, rejection sampling, Latin hypercube, stratified, and antithetic designs.
- **Estimators** — sample mean, weighted mean, sample variance, standard error, confidence and percentile intervals, running statistics, skewness, kurtosis, covariance, correlation.
- **Integration** — 1-D and multi-dimensional Monte Carlo integration with importance, antithetic, stratified, and control-variate variance reduction.
- **Importance sampling** — standard and self-normalized variants with effective sample size and KL diagnostics.
- **Simulation** — high-level orchestration, replicated runs, bootstrap resampling and bootstrap CIs.
- **Aggregation** — summary statistics, Gelman–Rubin diagnostic, autocorrelation, integrated autocorrelation time, ESS.
- **MCMC** — Metropolis–Hastings, random-walk MH, Gibbs, Hamiltonian Monte Carlo, and the No-U-Turn Sampler (NUTS).
- **Bayesian Monte Carlo** — posterior summaries, credible/HPD intervals, posterior predictive sampling and checks, marginal likelihood (IS and harmonic-mean), Bayes factors, GP-based Bayesian quadrature.
- **Plotting** — histograms, PDF overlays, empirical CDFs, running mean, running standard error, convergence bands, autocorrelation, weight diagnostics.
- **Progress bars** — every long-running routine ships with optional `tqdm` progress; pass `progress=False` to silence.

---

## Installation

```bash
pip install mcflow
```

For development:

```bash
git clone https://github.com/I-AlKhurayyif/mcflow.git
cd mcflow
pip install -e ".[dev]"
```

**Requirements:** Python ≥ 3.10, NumPy, SciPy, Matplotlib, tqdm.

---

## Quick start

```python
import mcflow

mcflow.set_seed(42)

# Draw samples and summarize
x = mcflow.sample_normal(0.0, 1.0, 5000)
print(f"mean = {mcflow.mc_mean(x):.4f}")
print(f"95% CI = {mcflow.confidence_interval(x)}")
```

---

## Examples

### Monte Carlo integration

```python
import mcflow

# 1-D: integrate x^2 over [0, 1]  (true value: 1/3)
result = mcflow.mc_integrate_1d(lambda x: x**2, 0, 1, n_samples=10_000)
print(result)
# IntegrationResult(estimate=0.3331, se=0.0030, n=10000)

# n-D: integrate x + y over [0, 1]^2  (true value: 1)
result = mcflow.mc_integrate_nd(
    lambda X: X[:, 0] + X[:, 1],
    bounds=[(0, 1), (0, 1)],
    n_samples=10_000,
)
print(result)
```

### Importance sampling

```python
import mcflow

result = mcflow.importance_sampling(
    func=lambda x: x**2,
    target_pdf=lambda x: mcflow.distributions.normal_pdf(x, 0, 1),
    proposal_pdf=lambda x: mcflow.distributions.normal_pdf(x, 0, 2),
    proposal_sampler=lambda n: mcflow.sample_normal(0, 2, n),
    n_samples=10_000,
)
print(f"estimate = {result.estimate:.4f}  (ESS = {result.effective_sample_size:.0f})")
```

### MCMC: random-walk Metropolis

```python
import numpy as np
import mcflow

# Target: standard 2-D normal
log_target = lambda x: -0.5 * np.sum(x**2)

chain = mcflow.random_walk_metropolis(
    log_target,
    x0=[0.0, 0.0],
    step_size=1.0,
    n_samples=10_000,
)
print(f"acc rate = {chain.acceptance_rate:.3f}")

# Drop burn-in, then summarize the posterior
samples = mcflow.burn_in(chain.samples, 1000)
print(mcflow.summarize(samples[:, 0]))
```

### Bayesian inference

```python
import numpy as np
import mcflow
from scipy.special import gammaln

# Beta(2,2) prior, Binomial(10, p) likelihood, 7 successes observed
def log_post(p):
    p = float(p[0])
    if not (0 < p < 1):
        return -np.inf
    log_prior = np.log(p) + np.log(1 - p)
    log_lik   = 7 * np.log(p) + 3 * np.log(1 - p)
    return log_prior + log_lik

chain = mcflow.random_walk_metropolis(log_post, x0=[0.5], step_size=0.1, n_samples=5000)
post = mcflow.burn_in(chain.samples, 500)

print(f"posterior mean: {mcflow.posterior_mean(post):.4f}")
print(f"95% credible interval: {mcflow.credible_interval(post)}")
print(f"95% HPD interval:      {mcflow.highest_posterior_density(post)}")
```

### Bayesian Quadrature

```python
import numpy as np
import mcflow

# Estimate ∫ x² · N(x | 0, 1) dx = 1 with only 11 evaluations
x = np.linspace(-3, 3, 11)
res = mcflow.bayesian_quadrature(
    func_values=x**2,
    x_train=x,
    measure_mean=0.0,
    measure_cov=1.0,
    length_scale=1.0,
    variance=10.0,
)
print(res)
# BayesianQuadratureResult(mean=0.998, se=0.0004, n_train=11)
```

### Convergence diagnostics

```python
import mcflow

x = mcflow.sample_normal(0, 1, 5000)
ax = mcflow.plot_convergence(x, true_value=0.0)
ax.figure.savefig("convergence.png", dpi=150)
```

---

## Package structure

```
mcflow_project/
├── pyproject.toml
├── README.md
├── LICENSE
├── assets/
│   ├── mcflow_logo.svg        # vector logo
│   └── mcflow_logo.png        # rasterized logo for PyPI / docs
├── src/
│   └── mcflow/
│       ├── __init__.py        # public API and re-exports
│       ├── distributions.py   # PDFs, CDFs, inverse CDFs
│       ├── sampling.py        # samplers and variance-reduction designs
│       ├── estimators.py      # means, variances, SEs, CIs, running stats
│       ├── aggregation.py     # summaries, convergence, autocorrelation
│       ├── integration.py     # Monte Carlo integration
│       ├── importance.py      # importance sampling + diagnostics
│       ├── simulation.py      # workflow orchestration, batch means, bootstrap
│       ├── mcmc.py            # MH, RWM, Gibbs, HMC, NUTS
│       ├── bmc.py             # Bayesian Monte Carlo + quadrature
│       └── plotting.py        # visualizations
└── tests/
    ├── __init__.py
    └── test_sampling.py
```

Most users only ever need top-level names:

```python
import mcflow
mcflow.mc_mean(samples)
mcflow.nuts(log_target, grad, x0, step_size=0.1, n_samples=2000)
```

For finer control the submodules are also exposed:

```python
from mcflow import mcmc, bmc, distributions
```

---

## Mathematical foundations

`mcflow` implements the standard formulas of Monte Carlo and Bayesian computation:

- **Monte Carlo mean:** μ̂ₙ = (1/N) Σ Xᵢ
- **Sample variance:** S² = (1/(N−1)) Σ (Xᵢ − μ̂ₙ)²
- **Standard error:** SE = S / √N
- **Normal CI:** μ̂ₙ ± z_{1−α/2} · SE
- **Monte Carlo integration:** I ≈ V(Ω) · (1/N) Σ f(Xᵢ),  Xᵢ ∼ U(Ω)
- **Importance sampling:** I_hat = (1/N) Σ f(Xᵢ) · p(Xᵢ)/q(Xᵢ),  Xᵢ ∼ q
- **Effective sample size:** ESS = (Σ wᵢ)² / Σ wᵢ²
- **Metropolis–Hastings acceptance:** α = min{1, [p(x′)q(x|x′)] / [p(x)q(x′|x)]}
- **HMC Hamiltonian:** H(x, p) = −log p(x) + ½ pᵀp
- **Bayesian Quadrature (RBF + Gaussian measure):** closed-form posterior mean and variance over ∫f(x)π(x)dx

See module docstrings for full derivations and references.

---

## Testing

```bash
pip install -e ".[dev]"
pytest                          # run all tests
pytest --cov=mcflow             # with coverage
pytest tests/test_sampling.py   # one module
```

---

## License

MIT.
