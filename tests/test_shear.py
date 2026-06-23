"""
Unit tests for a6cw.shear.

Key analytic reference values used:

  compute_tangential_ellipticities with halo at origin:
    Source at (r, 0)     → phi=0,    e_t = -e1,  e_x = -e2
    Source at (0, r)     → phi=π/2,  e_t = +e1,  e_x = +e2
    Source at (r,r)/√2   → phi=π/4,  e_t = -e2,  e_x = +e1

  mean_tangential_shear:
    Simple mean = arithmetic mean; error = std(ddof=1)/sqrt(n)
"""

import numpy as np
import pytest

from a6cw.shear import (
    N_BINS,
    BIN_EDGES,
    BIN_CENTRES,
    COLOURS,
    LABELS,
    load_positions,
    compute_tangential_ellipticities,
    mean_tangential_shear,
    estimate_shape_noise_variance,
    optimal_weights,
    build_nfw_theory,
    log_likelihood,
)


# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------

class TestModuleConstants:
    def test_bin_edges_has_nbins_plus_one_elements(self):
        assert len(BIN_EDGES) == N_BINS + 1

    def test_bin_centres_has_nbins_elements(self):
        assert len(BIN_CENTRES) == N_BINS

    def test_bin_centres_are_geometric_means(self):
        expected = np.sqrt(BIN_EDGES[:-1] * BIN_EDGES[1:])
        np.testing.assert_allclose(BIN_CENTRES, expected, rtol=1e-12)

    def test_bin_edges_strictly_increasing(self):
        assert np.all(np.diff(BIN_EDGES) > 0)

    def test_bin_edges_span_expected_range(self):
        assert BIN_EDGES[0] == pytest.approx(100.0, rel=1e-6)
        assert BIN_EDGES[-1] == pytest.approx(3600.0, rel=1e-6)

    def test_colours_has_all_three_methods(self):
        for key in ("unweighted", "weighted", "hsm"):
            assert key in COLOURS

    def test_labels_has_all_three_methods(self):
        for key in ("unweighted", "weighted", "hsm"):
            assert key in LABELS


# ---------------------------------------------------------------------------
# load_positions
# ---------------------------------------------------------------------------

class TestLoadPositions:
    def test_loads_two_column_data(self, tmp_path):
        data = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        path = tmp_path / "pos.txt"
        np.savetxt(str(path), data)
        result = load_positions(path)
        np.testing.assert_allclose(result, data)

    def test_output_shape(self, tmp_path):
        data = np.random.default_rng(0).uniform(0, 1000, (50, 2))
        path = tmp_path / "pos.txt"
        np.savetxt(str(path), data)
        result = load_positions(path)
        assert result.shape == (50, 2)

    def test_values_are_preserved(self, tmp_path):
        data = np.array([[100.5, -200.3], [-50.0, 999.9]])
        path = tmp_path / "pos.txt"
        np.savetxt(str(path), data)
        result = load_positions(path)
        np.testing.assert_allclose(result, data, rtol=1e-6)


# ---------------------------------------------------------------------------
# compute_tangential_ellipticities
# ---------------------------------------------------------------------------

class TestComputeTangentialEllipticities:
    """
    Reference geometry (halo at origin, source at angle phi):
        epsilon_t = -(e1 cos2φ + e2 sin2φ)
        epsilon_x = -(-e1 sin2φ + e2 cos2φ)
    """

    def _single(self, source_xy, e1, e2):
        halo   = np.array([[0.0, 0.0]])
        source = np.array([source_xy])
        r, et, ex = compute_tangential_ellipticities(
            halo, source, np.array([e1]), np.array([e2])
        )
        return r[0], et[0], ex[0]

    def test_source_east_pure_e1(self):
        # phi=0: e_t=-e1, e_x=-e2
        r, et, ex = self._single([100.0, 0.0], e1=1.0, e2=0.0)
        assert r  == pytest.approx(100.0)
        assert et == pytest.approx(-1.0)
        assert ex == pytest.approx(0.0, abs=1e-10)

    def test_source_north_pure_e1(self):
        # phi=π/2: cos(π)=-1, sin(π)≈0  → e_t=+e1, e_x=+e2
        r, et, ex = self._single([0.0, 100.0], e1=1.0, e2=0.0)
        assert r  == pytest.approx(100.0)
        assert et == pytest.approx(1.0)
        assert ex == pytest.approx(0.0, abs=1e-10)

    def test_source_northeast_pure_e2(self):
        # phi=π/4: cos(π/2)=0, sin(π/2)=1 → e_t=-e2, e_x=+e1
        s = 100.0 / np.sqrt(2)
        r, et, ex = self._single([s, s], e1=0.0, e2=1.0)
        assert r  == pytest.approx(100.0, rel=1e-10)
        assert et == pytest.approx(-1.0)
        assert ex == pytest.approx(0.0, abs=1e-10)

    def test_source_northeast_pure_e1(self):
        # phi=π/4, e1=1, e2=0 → e_t=0, e_x=+e1
        s = 100.0 / np.sqrt(2)
        r, et, ex = self._single([s, s], e1=1.0, e2=0.0)
        assert et == pytest.approx(0.0, abs=1e-10)
        assert ex == pytest.approx(1.0)

    def test_separation_formula(self):
        # 3-4-5 triangle
        r, _, _ = self._single([3.0, 4.0], e1=0.0, e2=0.0)
        assert r == pytest.approx(5.0)

    def test_output_shape_multiple_halos_sources(self):
        n_h, n_s = 7, 13
        rng = np.random.default_rng(0)
        halo_pos   = rng.uniform(0, 1e4, (n_h, 2))
        source_pos = rng.uniform(0, 1e4, (n_s, 2))
        r, et, ex = compute_tangential_ellipticities(
            halo_pos, source_pos, np.zeros(n_s), np.zeros(n_s)
        )
        assert r.shape  == (n_h * n_s,)
        assert et.shape == (n_h * n_s,)
        assert ex.shape == (n_h * n_s,)

    def test_zero_ellipticity_gives_zero_shear(self):
        r, et, ex = self._single([200.0, 300.0], e1=0.0, e2=0.0)
        assert et == pytest.approx(0.0)
        assert ex == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# mean_tangential_shear
# ---------------------------------------------------------------------------

class TestMeanTangentialShear:
    # Two bins: [0, 100) and [100, 1000)
    BIN_EDGES_2 = np.array([0.0, 100.0, 1000.0])

    def test_mean_is_correct(self):
        r  = np.array([50.0, 80.0, 150.0, 250.0])
        et = np.array([0.4,  0.6,  0.3,   0.7])
        gt, _, _ = mean_tangential_shear(r, et, self.BIN_EDGES_2)
        assert gt[0] == pytest.approx(0.5)   # (0.4 + 0.6) / 2
        assert gt[1] == pytest.approx(0.5)   # (0.3 + 0.7) / 2

    def test_n_pairs_counted_correctly(self):
        r  = np.array([50.0, 80.0, 150.0, 250.0])
        et = np.array([0.4,  0.6,  0.3,   0.7])
        _, _, n = mean_tangential_shear(r, et, self.BIN_EDGES_2)
        assert n[0] == 2
        assert n[1] == 2

    def test_error_is_standard_error_on_mean(self):
        r  = np.array([50.0, 80.0])
        et = np.array([0.4,  0.6])
        gt, gt_err, _ = mean_tangential_shear(r, et, self.BIN_EDGES_2)
        # std([0.4, 0.6], ddof=1) = sqrt(0.02); SEM = sqrt(0.02)/sqrt(2) = 0.1
        assert gt_err[0] == pytest.approx(0.1, rel=1e-6)

    def test_single_pair_returns_nan(self):
        r  = np.array([50.0])
        et = np.array([0.5])
        gt, gt_err, n = mean_tangential_shear(r, et, self.BIN_EDGES_2)
        assert np.isnan(gt[0]) and np.isnan(gt_err[0])
        assert n[0] == 1

    def test_nan_pairs_excluded_from_count(self):
        # One valid pair and one NaN in the same bin
        r  = np.array([50.0, 80.0])
        et = np.array([0.5,  np.nan])
        _, _, n = mean_tangential_shear(r, et, self.BIN_EDGES_2)
        assert n[0] == 1   # only one finite value

    def test_empty_bin_returns_nan(self):
        r  = np.array([150.0, 250.0])   # nothing in bin 0
        et = np.array([0.3,   0.7])
        gt, gt_err, n = mean_tangential_shear(r, et, self.BIN_EDGES_2)
        assert np.isnan(gt[0]) and np.isnan(gt_err[0])
        assert n[0] == 0

    def test_uniform_values_give_zero_error(self):
        r  = np.array([50.0, 80.0])
        et = np.array([0.5,  0.5])
        _, gt_err, _ = mean_tangential_shear(r, et, self.BIN_EDGES_2)
        assert gt_err[0] == pytest.approx(0.0, abs=1e-10)


# ---------------------------------------------------------------------------
# estimate_shape_noise_variance
# ---------------------------------------------------------------------------

class TestEstimateShapeNoiseVariance:
    def test_known_variance(self):
        # [1, -1]: mean=0, sum((x-mean)^2)=2, ddof=1 → var=2
        assert estimate_shape_noise_variance(np.array([1.0, -1.0])) == pytest.approx(2.0)

    def test_nan_values_excluded(self):
        # Adding NaN should not change the result
        result_clean = estimate_shape_noise_variance(np.array([1.0, -1.0]))
        result_nan   = estimate_shape_noise_variance(np.array([1.0, -1.0, np.nan]))
        assert result_clean == pytest.approx(result_nan)

    def test_zero_variance_for_constant(self):
        assert estimate_shape_noise_variance(np.array([0.5, 0.5, 0.5])) == pytest.approx(0.0, abs=1e-15)

    def test_returns_float(self):
        result = estimate_shape_noise_variance(np.array([0.1, -0.1, 0.2]))
        assert isinstance(result, float)


# ---------------------------------------------------------------------------
# optimal_weights
# ---------------------------------------------------------------------------

class TestOptimalWeights:
    def test_all_equal_for_uniform_noise(self):
        et = np.array([0.1, -0.2, 0.3, -0.1])
        w = optimal_weights(et)
        assert np.allclose(w, w[0])

    def test_nan_pairs_get_zero_weight(self):
        et = np.array([0.1, np.nan, 0.3])
        w = optimal_weights(et)
        assert w[1] == 0.0

    def test_finite_pairs_get_positive_weight(self):
        et = np.array([0.1, np.nan, 0.3])
        w = optimal_weights(et)
        assert w[0] > 0
        assert w[2] > 0

    def test_weights_equal_one_over_variance(self):
        et = np.array([1.0, -1.0])  # var = 2
        w = optimal_weights(et)
        assert np.all(w == pytest.approx(0.5))


# ---------------------------------------------------------------------------
# build_nfw_theory
# ---------------------------------------------------------------------------

class TestBuildNFWTheory:
    MASS = 5e14
    ZS   = 0.6

    def test_returns_nbins_values(self):
        gt = build_nfw_theory(self.MASS, self.ZS)
        assert len(gt) == N_BINS

    def test_all_positive(self):
        gt = build_nfw_theory(self.MASS, self.ZS)
        assert np.all(gt > 0)

    def test_decreasing_with_radius(self):
        # NFW tangential shear profile should be monotonically decreasing
        gt = build_nfw_theory(self.MASS, self.ZS)
        assert gt[0] > gt[-1]

    def test_higher_mass_gives_stronger_shear(self):
        gt_lo = build_nfw_theory(1e14, self.ZS)
        gt_hi = build_nfw_theory(1e15, self.ZS)
        assert np.all(gt_hi > gt_lo)

    def test_higher_source_redshift_gives_stronger_shear(self):
        # More distant sources experience stronger lensing
        gt_near = build_nfw_theory(self.MASS, source_redshift=0.4)
        gt_far  = build_nfw_theory(self.MASS, source_redshift=0.9)
        assert np.all(gt_far > gt_near)

    def test_cosmology_parameters_accepted(self):
        # Verify custom cosmology parameters are accepted without error
        gt = build_nfw_theory(self.MASS, self.ZS,
                              concentration=5.0, halo_redshift=0.3,
                              omega_m=0.3, omega_lam=0.7)
        assert len(gt) == N_BINS


# ---------------------------------------------------------------------------
# log_likelihood
# ---------------------------------------------------------------------------

class TestLogLikelihood:
    MASS = 5e14
    ZS   = 0.6

    def _theory(self):
        return build_nfw_theory(self.MASS, self.ZS)

    def test_perfect_fit_gives_zero(self):
        gt = self._theory()
        lnL = log_likelihood(gt, np.full(N_BINS, 0.01), self.MASS, self.ZS)
        assert lnL == pytest.approx(0.0, abs=1e-10)

    def test_imperfect_fit_gives_negative_lnL(self):
        gt = self._theory()
        lnL = log_likelihood(np.zeros(N_BINS), np.full(N_BINS, 0.01), self.MASS, self.ZS)
        assert lnL < 0

    def test_larger_residual_gives_smaller_lnL(self):
        gt  = self._theory()
        err = np.full(N_BINS, 0.01)
        lnL_small = log_likelihood(gt + 0.001, err, self.MASS, self.ZS)
        lnL_large = log_likelihood(gt + 0.1,   err, self.MASS, self.ZS)
        assert lnL_small > lnL_large

    def test_nan_bins_excluded(self):
        gt = self._theory()
        err = np.full(N_BINS, 0.01)
        gt_with_nan = gt.copy()
        gt_with_nan[0] = np.nan   # one missing bin
        lnL_full    = log_likelihood(gt,          err, self.MASS, self.ZS)
        lnL_missing = log_likelihood(gt_with_nan, err, self.MASS, self.ZS)
        # Perfect fit on remaining bins → 0; original is also 0 (perfect)
        assert lnL_missing == pytest.approx(0.0, abs=1e-10)
        assert lnL_full    == pytest.approx(0.0, abs=1e-10)

    def test_all_nan_returns_neg_inf(self):
        lnL = log_likelihood(
            np.full(N_BINS, np.nan), np.full(N_BINS, 0.01), self.MASS, self.ZS
        )
        assert lnL == -np.inf
