<p align="center">
  <img src="https://raw.githubusercontent.com/I-AlKhurayyif/mcflow/main/assets/mcflow_logo.svg?v=2" alt="mcflow logo" width="520"/>
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
pip install mcflow-py
```

Then import as `mcflow`:

```python
import mcflow
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

`mcflow` implements the standard formulas of Monte Carlo and Bayesian computation.

**Monte Carlo mean estimator.** The fundamental Monte Carlo approximation of an expectation:

$$\hat{\mu}_N = \frac{1}{N}\sum_{i=1}^{N} X_i \longrightarrow \mathbb{E}[X] \quad \text{as } N \to \infty$$

**Unbiased sample variance.** Dispersion estimator with Bessel's correction:

$$S^2 = \frac{1}{N-1}\sum_{i=1}^{N}(X_i - \hat{\mu}_N)^2$$

**Standard error of the mean.** Source of the famous $O(N^{-1/2})$ convergence rate:

$$\mathrm{SE}(\hat{\mu}_N) = \frac{S}{\sqrt{N}}$$

**Normal-approximation confidence interval** at confidence level $1-\alpha$:

$$\mathrm{CI}_{1-\alpha} = \left[ \hat{\mu}_N - z_{1-\alpha/2} \cdot \mathrm{SE}, \quad \hat{\mu}_N + z_{1-\alpha/2} \cdot \mathrm{SE} \right]$$

**Monte Carlo integration.** For an integrable function $f$ on a domain $\Omega \subset \mathbb{R}^d$ with volume $V(\Omega)$:

$$I = \int_{\Omega} f(\mathbf{x}) d\mathbf{x} \approx \frac{V(\Omega)}{N}\sum_{i=1}^{N} f(\mathbf{X}_i), \quad \mathbf{X}_i \sim \mathcal{U}(\Omega)$$

The convergence rate is independent of the dimension $d$ — Monte Carlo's defining advantage.

**Importance sampling estimator.** For a target density $p$ and a proposal $q$:

$$\hat{I}_{\mathrm{IS}} = \frac{1}{N}\sum_{i=1}^{N} f(X_i)\frac{p(X_i)}{q(X_i)}, \quad X_i \sim q$$

**Self-normalized importance sampling**, valid when $p$ and $q$ are known only up to a normalizing constant:

$$\hat{I}_{\mathrm{SNIS}} = \frac{\sum_{i=1}^{N} f(X_i) w_i}{\sum_{i=1}^{N} w_i}, \quad w_i = \frac{p(X_i)}{q(X_i)}$$

**Effective sample size.** Diagnoses the quality of an importance-sampling proposal:

$$\mathrm{ESS} = \frac{\left(\sum_{i=1}^{N} w_i\right)^2}{\sum_{i=1}^{N} w_i^2}$$

**Metropolis–Hastings acceptance probability** for a proposed move $x \to x'$:

$$\alpha(x, x') = \min\left( 1, \frac{p(x') q(x \mid x')}{p(x) q(x' \mid x)} \right)$$

**Hamiltonian Monte Carlo.** HMC augments the state with a momentum $\mathbf{p}$ and proposes moves by simulating Hamiltonian dynamics:

$$H(\mathbf{x}, \mathbf{p}) = -\log p(\mathbf{x}) + \frac{1}{2}\mathbf{p}^{\top} M^{-1} \mathbf{p}$$

`mcflow`'s HMC uses identity mass ($M = I$) with leapfrog integration; NUTS adapts the trajectory length automatically.

**Welford's online variance** (numerically stable running variance):

$$\hat{\mu}_n = \hat{\mu}_{n-1} + \frac{X_n - \hat{\mu}_{n-1}}{n}$$

$$M_n = M_{n-1} + (X_n - \hat{\mu}_{n-1})(X_n - \hat{\mu}_n)$$

$$S_n^2 = \frac{M_n}{n-1}$$

**Gelman–Rubin diagnostic** for assessing MCMC convergence across $m$ chains of length $n$:

$$\hat{R} = \sqrt{\frac{\frac{n-1}{n}W + \frac{1}{n}B}{W}}$$

where $W$ is the within-chain variance and $B$ is the between-chain variance. Values $\hat{R} \approx 1$ indicate convergence.

**Bayesian Quadrature** (O'Hagan, 1991). Treat the integrand as a Gaussian process. Under an RBF kernel paired with a Gaussian measure $\pi = \mathcal{N}(b, B)$, the posterior over the integral $I = \int f(\mathbf{x}) \pi(\mathbf{x}) d\mathbf{x}$ has closed form:

$$\mathbb{E}[I \mid \mathcal{D}] = \mathbf{z}^{\top} K^{-1} \mathbf{y}$$

$$\mathrm{Var}[I \mid \mathcal{D}] = \iint k(\mathbf{x}, \mathbf{x}') \pi(\mathbf{x}) \pi(\mathbf{x}') d\mathbf{x} d\mathbf{x}' - \mathbf{z}^{\top} K^{-1} \mathbf{z}$$

with kernel mean embedding $z_i = \int k(\mathbf{x}, \mathbf{x}_i) \pi(\mathbf{x}) d\mathbf{x}$. This gives both a point estimate **and** a posterior variance over the integral itself — often with far fewer evaluations than ordinary Monte Carlo on smooth integrands.

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
