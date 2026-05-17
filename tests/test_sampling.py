"""
Tests for ``mcflow.sampling``.

Tested behaviors:

* Output shape, dtype, and bounds for every direct sampler.
* Statistical sanity: empirical mean/std are close to theoretical values.
* Goodness-of-fit (Kolmogorov-Smirnov) for continuous samplers against the
  matching ``mcflow.distributions`` CDFs.
* Reproducibility under ``set_seed`` and per-call ``seed=``.
* Independence of the global RNG when a per-call seed is given.
* Inverse-transform, Box-Muller, and rejection samplers produce samples
  from the intended distribution.
* Variance-reduction designs (Latin hypercube, stratified, antithetic)
  satisfy their structural guarantees.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy import stats

import mcflow
from mcflow import distributions, sampling


# ---------------------------------------------------------------------------
# Helpers and fixtures
# ---------------------------------------------------------------------------
N_LARGE = 20_000        # sample size for statistical checks
KS_THRESHOLD = 0.02     # KS statistic threshold for "close enough" at N_LARGE
ATOL_MEAN = 0.05        # tolerance for empirical mean vs theoretical
ATOL_STD = 0.05         # tolerance for empirical std vs theoretical


@pytest.fixture(autouse=True)
def _reset_global_seed():
    """Reset the global RNG before every test for deterministic behavior."""
    sampling.set_seed(12345)
    yield


# ===========================================================================
# Direct samplers: shape, dtype, bounds, and statistical sanity
# ===========================================================================
class TestUniform:
    def test_returns_correct_shape(self):
        x = mcflow.sample_uniform(0.0, 1.0, size=500)
        assert x.shape == (500,)

    def test_returns_float_array(self):
        x = mcflow.sample_uniform(-1.0, 1.0, size=10)
        assert x.dtype == np.float64

    def test_samples_within_bounds(self):
        x = mcflow.sample_uniform(2.0, 5.0, size=N_LARGE)
        assert np.all(x >= 2.0)
        assert np.all(x <= 5.0)

    def test_empirical_mean_close_to_theoretical(self):
        x = mcflow.sample_uniform(-3.0, 7.0, size=N_LARGE)
        assert np.mean(x) == pytest.approx(2.0, abs=ATOL_MEAN)

    def test_empirical_std_close_to_theoretical(self):
        # std of Uniform(-3, 7) = (7 - (-3)) / sqrt(12)
        x = mcflow.sample_uniform(-3.0, 7.0, size=N_LARGE)
        theoretical_std = (7.0 - (-3.0)) / np.sqrt(12.0)
        assert np.std(x, ddof=1) == pytest.approx(theoretical_std, abs=ATOL_STD)

    def test_ks_against_uniform_cdf(self):
        x = mcflow.sample_uniform(0.0, 1.0, size=N_LARGE)
        ks_stat, _ = stats.kstest(x, "uniform")
        assert ks_stat < KS_THRESHOLD


class TestNormal:
    def test_returns_correct_shape(self):
        x = mcflow.sample_normal(0.0, 1.0, size=500)
        assert x.shape == (500,)

    def test_empirical_moments(self):
        x = mcflow.sample_normal(mean=3.0, std=2.0, size=N_LARGE)
        assert np.mean(x) == pytest.approx(3.0, abs=ATOL_MEAN)
        assert np.std(x, ddof=1) == pytest.approx(2.0, abs=ATOL_STD)

    def test_ks_against_normal_cdf(self):
        x = mcflow.sample_normal(0.0, 1.0, size=N_LARGE)
        ks_stat, _ = stats.kstest(x, "norm")
        assert ks_stat < KS_THRESHOLD

    def test_standard_normal_helper_matches_normal(self):
        # Two equivalent draws with the same seed should match exactly.
        a = mcflow.sample_standard_normal(size=200, seed=99)
        b = np.random.default_rng(99).standard_normal(200)
        np.testing.assert_array_equal(a, b)


class TestTriangular:
    def test_samples_within_bounds(self):
        x = mcflow.sample_triangular(0.0, 5.0, 10.0, size=N_LARGE)
        assert np.all(x >= 0.0)
        assert np.all(x <= 10.0)

    def test_empirical_mean_close_to_theoretical(self):
        # Mean of Triangular(a, c, b) = (a + b + c) / 3
        x = mcflow.sample_triangular(0.0, 4.0, 10.0, size=N_LARGE)
        assert np.mean(x) == pytest.approx((0.0 + 4.0 + 10.0) / 3.0, abs=ATOL_MEAN)

    def test_mode_near_peak(self):
        # Histogram peak should be near the declared mode.
        x = mcflow.sample_triangular(0.0, 8.0, 10.0, size=N_LARGE)
        counts, edges = np.histogram(x, bins=50)
        centers = 0.5 * (edges[:-1] + edges[1:])
        peak_center = centers[int(np.argmax(counts))]
        # Loose check: peak should be in the upper half of the support.
        assert peak_center > 5.0


class TestExponential:
    def test_samples_nonnegative(self):
        x = mcflow.sample_exponential(rate=1.5, size=N_LARGE)
        assert np.all(x >= 0.0)

    def test_empirical_mean_close_to_theoretical(self):
        rate = 2.0
        x = mcflow.sample_exponential(rate=rate, size=N_LARGE)
        assert np.mean(x) == pytest.approx(1.0 / rate, abs=ATOL_MEAN)

    def test_ks_against_exponential_cdf(self):
        x = mcflow.sample_exponential(rate=1.0, size=N_LARGE)
        ks_stat, _ = stats.kstest(x, "expon")
        assert ks_stat < KS_THRESHOLD


class TestBernoulli:
    def test_returns_zero_one_integers(self):
        x = mcflow.sample_bernoulli(p=0.5, size=200)
        assert set(np.unique(x)).issubset({0, 1})
        assert np.issubdtype(x.dtype, np.integer)

    def test_empirical_proportion_close_to_p(self):
        p = 0.3
        x = mcflow.sample_bernoulli(p=p, size=N_LARGE)
        assert np.mean(x) == pytest.approx(p, abs=0.01)

    def test_extremes(self):
        assert np.all(mcflow.sample_bernoulli(p=1.0, size=100) == 1)
        assert np.all(mcflow.sample_bernoulli(p=0.0, size=100) == 0)


# ===========================================================================
# Reproducibility: global seed and per-call seed
# ===========================================================================
class TestReproducibility:
    def test_global_seed_makes_identical_streams(self):
        mcflow.set_seed(7)
        a = mcflow.sample_normal(0, 1, size=100)
        mcflow.set_seed(7)
        b = mcflow.sample_normal(0, 1, size=100)
        np.testing.assert_array_equal(a, b)

    def test_per_call_seed_is_deterministic(self):
        a = mcflow.sample_uniform(0, 1, size=100, seed=42)
        b = mcflow.sample_uniform(0, 1, size=100, seed=42)
        np.testing.assert_array_equal(a, b)

    def test_different_seeds_yield_different_samples(self):
        a = mcflow.sample_normal(0, 1, size=100, seed=1)
        b = mcflow.sample_normal(0, 1, size=100, seed=2)
        assert not np.allclose(a, b)

    def test_per_call_seed_does_not_advance_global_rng(self):
        # The point: passing seed=... should NOT consume global RNG state.
        mcflow.set_seed(0)
        _ = mcflow.sample_normal(0, 1, size=50, seed=999)  # uses its own RNG
        a = mcflow.sample_normal(0, 1, size=50)

        mcflow.set_seed(0)
        b = mcflow.sample_normal(0, 1, size=50)
        np.testing.assert_array_equal(a, b)


# ===========================================================================
# Inverse transform sampling
# ===========================================================================
class TestInverseTransform:
    def test_inverse_transform_uniform_matches_direct(self):
        # Inverse-transform with the uniform inverse CDF must reproduce
        # the same statistical distribution as sample_uniform.
        x = mcflow.inverse_transform_sample(
            lambda u: distributions.uniform_inverse_cdf(u, 2.0, 5.0),
            size=N_LARGE,
        )
        assert np.all(x >= 2.0) and np.all(x <= 5.0)
        assert np.mean(x) == pytest.approx(3.5, abs=ATOL_MEAN)

    def test_inverse_transform_exponential(self):
        x = mcflow.inverse_transform_sample(
            lambda u: distributions.exponential_inverse_cdf(u, rate=2.0),
            size=N_LARGE,
        )
        # Compare against the analytical Exponential(2) CDF via KS.
        ks_stat, _ = stats.kstest(x, lambda t: 1 - np.exp(-2.0 * t))
        assert ks_stat < KS_THRESHOLD

    def test_inverse_transform_normal(self):
        x = mcflow.inverse_transform_sample(
            lambda u: distributions.normal_inverse_cdf(u, mean=0.0, std=1.0),
            size=N_LARGE,
        )
        ks_stat, _ = stats.kstest(x, "norm")
        assert ks_stat < KS_THRESHOLD


# ===========================================================================
# Box-Muller
# ===========================================================================
class TestBoxMuller:
    def test_produces_standard_normal(self):
        z = mcflow.box_muller_transform(size=N_LARGE, seed=0)
        assert z.shape == (N_LARGE,)
        ks_stat, _ = stats.kstest(z, "norm")
        assert ks_stat < KS_THRESHOLD

    def test_odd_size_returns_exact_length(self):
        z = mcflow.box_muller_transform(size=101, seed=0)
        assert z.shape == (101,)

    def test_moments(self):
        z = mcflow.box_muller_transform(size=N_LARGE, seed=1)
        assert np.mean(z) == pytest.approx(0.0, abs=ATOL_MEAN)
        assert np.std(z, ddof=1) == pytest.approx(1.0, abs=ATOL_STD)


# ===========================================================================
# Rejection sampling
# ===========================================================================
class TestRejectionSample:
    def test_produces_target_distribution(self):
        # Target: Triangular(0, 1, 2). Proposal: Uniform(0, 2). M = 2 * peak.
        target = lambda x: distributions.triangular_pdf(x, 0.0, 1.0, 2.0)
        proposal = lambda x: distributions.uniform_pdf(x, 0.0, 2.0)
        proposal_sampler = lambda n: sampling._rng().uniform(0.0, 2.0, n)
        M = 2.5  # safe envelope: peak is 1.0, uniform pdf is 0.5

        x = mcflow.rejection_sample(
            target, proposal_sampler, proposal,
            M=M, size=5000, seed=42, progress=False,
        )
        assert x.shape == (5000,)
        assert np.all(x >= 0.0) and np.all(x <= 2.0)
        # Mean of Triangular(0, 1, 2) is 1.0.
        assert np.mean(x) == pytest.approx(1.0, abs=ATOL_MEAN)

    def test_raises_when_envelope_too_small(self):
        # Set M = 0 so acceptance probability is always >= 1, which still
        # works -- so instead use a deliberately impossible scenario: an
        # extremely tight max_iter budget on a very rare event.
        target = lambda x: np.where(np.abs(x - 100.0) < 0.001,
                                    distributions.normal_pdf(x, 0, 1), 0.0)
        proposal = lambda x: distributions.normal_pdf(x, 0, 1)
        proposal_sampler = lambda n: sampling._rng().standard_normal(n)
        with pytest.raises(RuntimeError):
            mcflow.rejection_sample(
                target, proposal_sampler, proposal,
                M=1.0, size=100, max_iter=10, seed=0, progress=False,
            )


# ===========================================================================
# Variance-reduction designs
# ===========================================================================
class TestLatinHypercube:
    def test_shape(self):
        pts = mcflow.latin_hypercube_sample(n_samples=20, n_dim=3, seed=0)
        assert pts.shape == (20, 3)

    def test_values_in_unit_cube(self):
        pts = mcflow.latin_hypercube_sample(n_samples=50, n_dim=4, seed=0)
        assert np.all(pts >= 0.0) and np.all(pts <= 1.0)

    def test_one_sample_per_stratum_per_dimension(self):
        # In any dimension, the n_samples values should fall into n_samples
        # disjoint equal-probability strata of [0, 1].
        n = 30
        pts = mcflow.latin_hypercube_sample(n_samples=n, n_dim=2, seed=0)
        for j in range(pts.shape[1]):
            strata = np.floor(pts[:, j] * n).astype(int)
            strata[strata == n] = n - 1
            # Each stratum index 0..n-1 must appear exactly once.
            counts = np.bincount(strata, minlength=n)
            assert np.all(counts == 1), f"dim {j} strata counts: {counts}"


class TestStratifiedSample:
    def test_shape_and_bounds(self):
        x = mcflow.stratified_sample(n_samples=50, low=2.0, high=7.0, seed=0)
        assert x.shape == (50,)
        assert np.all(x >= 2.0) and np.all(x <= 7.0)

    def test_one_per_stratum(self):
        # Each stratum [low + (i-1) * w, low + i * w] gets exactly one sample.
        n = 40
        x = mcflow.stratified_sample(n_samples=n, low=0.0, high=1.0, seed=0)
        strata = np.floor(x * n).astype(int)
        strata[strata == n] = n - 1
        counts = np.bincount(strata, minlength=n)
        assert np.all(counts == 1)


class TestAntitheticSample:
    def test_shape(self):
        u, u_anti = mcflow.antithetic_sample(size=100, seed=0)
        assert u.shape == (100,)
        assert u_anti.shape == (100,)

    def test_antithetic_property(self):
        u, u_anti = mcflow.antithetic_sample(size=200, seed=0)
        np.testing.assert_allclose(u_anti, 1.0 - u)

    def test_negative_correlation_for_monotone_transform(self):
        # For a monotone f, f(U) and f(1 - U) are negatively correlated.
        u, u_anti = mcflow.antithetic_sample(size=N_LARGE, seed=0)
        f = lambda x: x**2
        rho = np.corrcoef(f(u), f(u_anti))[0, 1]
        assert rho < 0.0


# ===========================================================================
# Edge cases
# ===========================================================================
class TestEdgeCases:
    def test_zero_size_uniform(self):
        x = mcflow.sample_uniform(0.0, 1.0, size=0)
        assert x.shape == (0,)

    def test_single_sample(self):
        x = mcflow.sample_normal(0.0, 1.0, size=1)
        assert x.shape == (1,)

    def test_uniform_degenerate_interval(self):
        # When low == high, all samples must equal that point.
        x = mcflow.sample_uniform(2.5, 2.5, size=100)
        np.testing.assert_array_equal(x, np.full(100, 2.5))

    def test_normal_zero_std_is_degenerate(self):
        x = mcflow.sample_normal(mean=4.0, std=0.0, size=100)
        np.testing.assert_array_equal(x, np.full(100, 4.0))
