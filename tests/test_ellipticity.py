"""
Unit tests for a6cw.ellipticity.

Analytic reference values used throughout:

  Unweighted moments of a 2D Gaussian  exp(-x²/2σ_x² - y²/2σ_y²):
      Q11 = σ_x², Q22 = σ_y², Q12 = 0
      denom = (σ_x + σ_y)²
      e1 = (σ_x - σ_y) / (σ_x + σ_y),  e2 = 0

  Rotating that Gaussian by 45° (u = (x+y)/√2, v = (-x+y)/√2):
      Q11 = Q22 = (σ_x² + σ_y²)/2,  Q12 = (σ_x² - σ_y²)/2
      e1 = 0,  e2 = (σ_x - σ_y) / (σ_x + σ_y)

  For a GalSim Gaussian with reduced shear g1 applied:
      semi-axes a = σ√((1+g1)/(1-g1)), b = σ√((1-g1)/(1+g1))
      unweighted e1 = (a - b)/(a + b) = g1  (exact)
"""

import numpy as np
import pytest
import galsim
import galsim.hsm

from a6cw.ellipticity import (
    _pixel_grids,
    unweighted_ellipticity,
    _gaussian_weight,
    weighted_ellipticity_raw,
    hsm_ellipticity,
    _ols_through_origin,
    apply_simulation_calibration,
)

STAMP = 96
SCALE = 1.0


def _draw(profile: galsim.GSObject) -> galsim.Image:
    return profile.drawImage(nx=STAMP, ny=STAMP, scale=SCALE, method="no_pixel")


def _analytic_gaussian(sigma_x: float, sigma_y: float, angle_deg: float = 0.0) -> galsim.Image:
    """2D Gaussian with given axis widths, optionally rotated."""
    cx, cy = (STAMP - 1) / 2.0, (STAMP - 1) / 2.0
    xi = np.arange(STAMP, dtype=float) - cx
    yi = np.arange(STAMP, dtype=float) - cy
    x, y = np.meshgrid(xi, yi)
    if angle_deg != 0.0:
        θ = np.deg2rad(angle_deg)
        u = x * np.cos(θ) + y * np.sin(θ)
        v = -x * np.sin(θ) + y * np.cos(θ)
    else:
        u, v = x, y
    arr = np.exp(-u**2 / (2 * sigma_x**2) - v**2 / (2 * sigma_y**2))
    return galsim.Image(arr, scale=SCALE)


# ---------------------------------------------------------------------------
# _pixel_grids
# ---------------------------------------------------------------------------

class TestPixelGrids:
    def test_shape_matches_image(self):
        im = _draw(galsim.Gaussian(sigma=3.0))
        x, y = _pixel_grids(im)
        assert x.shape == im.array.shape
        assert y.shape == im.array.shape

    def test_grids_sum_to_zero(self):
        # Grids are symmetric about 0, so their sum is 0 regardless of parity.
        im = galsim.Image(np.zeros((10, 10)), scale=1.0)
        x, y = _pixel_grids(im)
        assert x.sum() == pytest.approx(0.0, abs=1e-10)
        assert y.sum() == pytest.approx(0.0, abs=1e-10)

    def test_center_pixel_is_zero_odd_stamp(self):
        im = galsim.Image(np.zeros((11, 11)), scale=1.0)
        x, y = _pixel_grids(im)
        assert x[5, 5] == pytest.approx(0.0)
        assert y[5, 5] == pytest.approx(0.0)

    def test_range_even_stamp(self):
        im = galsim.Image(np.zeros((10, 10)), scale=1.0)
        x, y = _pixel_grids(im)
        assert x.min() == pytest.approx(-4.5)
        assert x.max() == pytest.approx(4.5)

    def test_x_varies_along_columns(self):
        im = galsim.Image(np.zeros((11, 11)), scale=1.0)
        x, _ = _pixel_grids(im)
        # All rows in the same column share the same x value
        assert np.all(x[:, 0] == x[0, 0])

    def test_y_varies_along_rows(self):
        im = galsim.Image(np.zeros((11, 11)), scale=1.0)
        _, y = _pixel_grids(im)
        # All columns in the same row share the same y value
        assert np.all(y[0, :] == y[0, 0])


# ---------------------------------------------------------------------------
# _gaussian_weight
# ---------------------------------------------------------------------------

class TestGaussianWeight:
    def test_peak_at_origin(self):
        assert _gaussian_weight(np.array([0.0]), np.array([0.0]), sigma=3.0)[0] == pytest.approx(1.0)

    def test_value_at_one_sigma(self):
        W = _gaussian_weight(np.array([3.0]), np.array([0.0]), sigma=3.0)
        assert W[0] == pytest.approx(np.exp(-0.5), rel=1e-12)

    def test_symmetric(self):
        W_pos = _gaussian_weight(np.array([2.0]), np.array([1.0]), sigma=3.0)
        W_neg = _gaussian_weight(np.array([-2.0]), np.array([-1.0]), sigma=3.0)
        assert W_pos[0] == pytest.approx(W_neg[0])

    def test_strictly_positive(self):
        dx = np.linspace(-10, 10, 50)
        dy = np.linspace(-10, 10, 50)
        W = _gaussian_weight(dx, dy, sigma=3.0)
        assert np.all(W > 0)


# ---------------------------------------------------------------------------
# unweighted_ellipticity
# ---------------------------------------------------------------------------

class TestUnweightedEllipticity:
    def test_circular_gives_zero(self):
        im = _draw(galsim.Gaussian(sigma=6.0))
        e1, e2 = unweighted_ellipticity(im)
        assert e1 == pytest.approx(0.0, abs=1e-6)
        assert e2 == pytest.approx(0.0, abs=1e-6)

    def test_zero_flux_returns_nan(self):
        im = galsim.Image(np.zeros((48, 48)), scale=1.0)
        e1, e2 = unweighted_ellipticity(im)
        assert np.isnan(e1) and np.isnan(e2)

    def test_axis_aligned_gaussian_e1(self):
        # e1 = (σ_x - σ_y) / (σ_x + σ_y)  (see module docstring)
        sigma_x, sigma_y = 12.0, 4.0
        im = _analytic_gaussian(sigma_x, sigma_y)
        e1, e2 = unweighted_ellipticity(im)
        assert e1 == pytest.approx((sigma_x - sigma_y) / (sigma_x + sigma_y), rel=1e-3)
        assert e2 == pytest.approx(0.0, abs=1e-5)

    def test_axis_aligned_gaussian_negative_e1(self):
        sigma_x, sigma_y = 4.0, 12.0   # flipped → negative e1
        im = _analytic_gaussian(sigma_x, sigma_y)
        e1, e2 = unweighted_ellipticity(im)
        assert e1 == pytest.approx((sigma_x - sigma_y) / (sigma_x + sigma_y), rel=1e-3)

    def test_rotated_45_gaussian_gives_pure_e2(self):
        # Rotating by 45° transfers elongation from e1 into e2
        sigma_x, sigma_y = 12.0, 4.0
        im = _analytic_gaussian(sigma_x, sigma_y, angle_deg=45.0)
        e1, e2 = unweighted_ellipticity(im)
        assert e1 == pytest.approx(0.0, abs=1e-4)
        assert e2 == pytest.approx((sigma_x - sigma_y) / (sigma_x + sigma_y), rel=1e-3)

    def test_sheared_galsim_gaussian(self):
        # For a GalSim Gaussian with reduced shear g1, the unweighted moments
        # give e1 = (a-b)/(a+b) = g1 exactly (see module docstring).
        g1 = 0.3
        im = _draw(galsim.Gaussian(sigma=5.0).shear(g1=g1, g2=0.0))
        e1, e2 = unweighted_ellipticity(im)
        assert e1 == pytest.approx(g1, rel=1e-5)
        assert e2 == pytest.approx(0.0, abs=1e-10)

    def test_g2_shear(self):
        g2 = 0.25
        im = _draw(galsim.Gaussian(sigma=5.0).shear(g1=0.0, g2=g2))
        e1, e2 = unweighted_ellipticity(im)
        assert e1 == pytest.approx(0.0, abs=1e-10)
        assert e2 == pytest.approx(g2, rel=1e-5)

    def test_result_bounded_by_one(self):
        # For any valid stamp, |e| <= 1
        im = _draw(galsim.Gaussian(sigma=5.0).shear(g1=0.5, g2=0.0))
        e1, e2 = unweighted_ellipticity(im)
        assert abs(e1) <= 1.0
        assert abs(e2) <= 1.0


# ---------------------------------------------------------------------------
# weighted_ellipticity_raw
# ---------------------------------------------------------------------------

class TestWeightedEllipticity:
    def test_circular_gives_zero(self):
        im = _draw(galsim.Gaussian(sigma=6.0))
        e1, e2 = weighted_ellipticity_raw(im, fwhm_px=4.0)
        assert e1 == pytest.approx(0.0, abs=1e-5)
        assert e2 == pytest.approx(0.0, abs=1e-5)

    def test_zero_flux_returns_nan(self):
        im = galsim.Image(np.zeros((48, 48)), scale=1.0)
        e1, e2 = weighted_ellipticity_raw(im, fwhm_px=4.0)
        assert np.isnan(e1) and np.isnan(e2)

    def test_g1_shear_gives_positive_e1(self):
        # The response is biased relative to unweighted, but the sign must match
        im = _draw(galsim.Gaussian(sigma=6.0).shear(g1=0.3, g2=0.0))
        e1, e2 = weighted_ellipticity_raw(im, fwhm_px=4.0)
        assert e1 > 0
        assert e2 == pytest.approx(0.0, abs=1e-4)

    def test_g2_shear_gives_positive_e2(self):
        im = _draw(galsim.Gaussian(sigma=6.0).shear(g1=0.0, g2=0.3))
        e1, e2 = weighted_ellipticity_raw(im, fwhm_px=4.0)
        assert e2 > 0
        assert e1 == pytest.approx(0.0, abs=1e-4)

    def test_result_bounded_by_one(self):
        im = _draw(galsim.Gaussian(sigma=5.0).shear(g1=0.4, g2=0.0))
        e1, e2 = weighted_ellipticity_raw(im, fwhm_px=4.0)
        assert abs(e1) <= 1.0
        assert abs(e2) <= 1.0

    def test_wider_weight_increases_response(self):
        # A wider weight function captures more of the galaxy shape, so the
        # weighted e1 grows towards the unweighted value as fwhm increases.
        im = _draw(galsim.Gaussian(sigma=6.0).shear(g1=0.3, g2=0.0))
        e1_narrow, _ = weighted_ellipticity_raw(im, fwhm_px=2.0)
        e1_wide, _   = weighted_ellipticity_raw(im, fwhm_px=8.0)
        assert e1_wide > e1_narrow


# ---------------------------------------------------------------------------
# hsm_ellipticity
# ---------------------------------------------------------------------------

class TestHSMEllipticity:
    def test_circular_gives_zero_shear(self):
        im = _draw(galsim.Gaussian(sigma=3.0))
        g1, g2, sigma = hsm_ellipticity(im)
        assert g1 == pytest.approx(0.0, abs=1e-6)
        assert g2 == pytest.approx(0.0, abs=1e-6)

    def test_circular_gives_positive_size(self):
        im = _draw(galsim.Gaussian(sigma=3.0))
        _, _, sigma = hsm_ellipticity(im)
        assert sigma > 0

    def test_g1_recovered(self):
        g1_in = 0.2
        im = _draw(galsim.Gaussian(sigma=3.0).shear(g1=g1_in, g2=0.0))
        g1, g2, _ = hsm_ellipticity(im)
        assert g1 == pytest.approx(g1_in, rel=1e-4)
        assert g2 == pytest.approx(0.0, abs=1e-5)

    def test_g2_recovered(self):
        g2_in = 0.15
        im = _draw(galsim.Gaussian(sigma=3.0).shear(g1=0.0, g2=g2_in))
        g1, g2, _ = hsm_ellipticity(im)
        assert g1 == pytest.approx(0.0, abs=1e-5)
        assert g2 == pytest.approx(g2_in, rel=1e-4)

    def test_empty_image_returns_nan(self):
        im = galsim.Image(np.zeros((48, 48)), scale=1.0)
        g1, g2, sigma = hsm_ellipticity(im)
        assert np.isnan(g1) and np.isnan(g2) and np.isnan(sigma)

    def test_larger_galaxy_gives_larger_sigma(self):
        im_small = _draw(galsim.Gaussian(sigma=2.0))
        im_large = _draw(galsim.Gaussian(sigma=5.0))
        _, _, sigma_small = hsm_ellipticity(im_small)
        _, _, sigma_large = hsm_ellipticity(im_large)
        assert sigma_large > sigma_small


# ---------------------------------------------------------------------------
# _ols_through_origin
# ---------------------------------------------------------------------------

class TestOLSThroughOrigin:
    def test_perfect_slope(self):
        x = np.array([1.0, 2.0, 3.0, 4.0])
        R, sigma_R = _ols_through_origin(x, 2.0 * x)
        assert R == pytest.approx(2.0, rel=1e-10)
        assert sigma_R == pytest.approx(0.0, abs=1e-10)

    def test_slope_with_noise(self):
        rng = np.random.default_rng(42)
        x = np.arange(1.0, 2001.0)
        y = 3.0 * x + rng.normal(0, 0.1, size=len(x))
        R, _ = _ols_through_origin(x, y)
        assert R == pytest.approx(3.0, rel=1e-2)

    def test_nan_values_are_filtered(self):
        x = np.array([1.0, 2.0, np.nan, 4.0])
        y = np.array([2.0, 4.0, np.nan, 8.0])
        R, _ = _ols_through_origin(x, y)
        assert R == pytest.approx(2.0, rel=1e-10)

    def test_negative_slope(self):
        x = np.array([1.0, 2.0, 3.0])
        R, _ = _ols_through_origin(x, -0.5 * x)
        assert R == pytest.approx(-0.5, rel=1e-10)


# ---------------------------------------------------------------------------
# apply_simulation_calibration
# ---------------------------------------------------------------------------

class TestApplySimulationCalibration:
    def test_unit_response_is_identity(self):
        epsilon = np.array([[0.1, -0.05], [0.2, 0.1]])
        sigma_gs = np.array([2.0, 3.0])
        nodes_x = np.array([1.0, 5.0])
        nodes_R = np.array([1.0, 1.0])
        result = apply_simulation_calibration(epsilon, sigma_gs, nodes_x, nodes_R)
        np.testing.assert_allclose(result, epsilon, rtol=1e-10)

    def test_half_response_doubles_ellipticity(self):
        epsilon = np.array([[0.1, 0.2]])
        sigma_gs = np.array([3.0])
        nodes_x = np.array([1.0, 5.0])
        nodes_R = np.array([0.5, 0.5])
        result = apply_simulation_calibration(epsilon, sigma_gs, nodes_x, nodes_R)
        assert result[0, 0] == pytest.approx(0.2, rel=1e-10)
        assert result[0, 1] == pytest.approx(0.4, rel=1e-10)

    def test_nan_ellipticity_stays_nan(self):
        epsilon = np.array([[np.nan, np.nan], [0.1, 0.0]])
        sigma_gs = np.array([3.0, 3.0])
        nodes_x = np.array([1.0, 5.0])
        nodes_R = np.array([1.0, 1.0])
        result = apply_simulation_calibration(epsilon, sigma_gs, nodes_x, nodes_R)
        assert np.all(np.isnan(result[0]))
        assert result[1, 0] == pytest.approx(0.1, rel=1e-10)

    def test_nan_sigma_g_uses_median(self):
        # Galaxy with nan sigma_g is assigned the median of the others.
        epsilon = np.array([[0.1, 0.0], [0.2, 0.0]])
        sigma_gs = np.array([np.nan, 3.0])
        nodes_x = np.array([1.0, 5.0])
        nodes_R = np.array([1.0, 1.0])   # R=1 everywhere
        result = apply_simulation_calibration(epsilon, sigma_gs, nodes_x, nodes_R)
        # Median sigma_g = 3.0, R(3.0)=1 → no change
        assert result[0, 0] == pytest.approx(0.1, rel=1e-10)
        assert result[1, 0] == pytest.approx(0.2, rel=1e-10)

    def test_interpolation_at_boundary(self):
        # Values outside the calibration range are clamped, not extrapolated
        epsilon = np.array([[0.1, 0.0]])
        sigma_gs = np.array([100.0])   # far beyond nodes
        nodes_x = np.array([1.0, 5.0])
        nodes_R = np.array([0.5, 1.0])
        result = apply_simulation_calibration(epsilon, sigma_gs, nodes_x, nodes_R)
        # np.interp clamps to R=1.0 at the upper boundary
        assert result[0, 0] == pytest.approx(0.1, rel=1e-10)
