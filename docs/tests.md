# Test Suite Documentation

The test suite lives in `tests/` and is run with pytest:

```bash
python -m pytest tests/ -v
```

There are 82 tests spread across two files, all of which run in under 5 seconds on a standard laptop. No data files from `data/` or `outputs/` are required.

---

## Design principles

**Analytic ground truth where possible.** The most important tests do not rely on running the full pipeline and comparing to a stored baseline. Instead they construct inputs with known mathematical properties and verify the output against a closed-form prediction. This means a failing test tells you exactly which formula is wrong.

**GalSim profiles for the estimator tests.** The three shape estimators operate on `galsim.Image` objects. Rather than reading real galaxy stamps, the tests draw simple noiseless profiles (Gaussians with and without applied shear) using GalSim's `drawImage`. Using `method='no_pixel'` gives a pixel array that matches the analytical profile to machine precision, which is needed when checking the estimators against closed-form ellipticity values.

**Edge cases alongside the happy path.** Every estimator test includes a zero-flux (or empty) input that should return `nan`, alongside the positive-signal cases.

**Separation of concerns.** `test_ellipticity.py` covers only functions in `a6cw.ellipticity`; `test_shear.py` covers only `a6cw.shear`. This makes failures easy to localise.

---

## Analytic reference values

### Unweighted moments of a 2-D Gaussian

For an axis-aligned Gaussian $I(x, y) \propto \exp(-x^2/2\sigma_x^2 - y^2/2\sigma_y^2)$ the second moments are $Q_{11} = \sigma_x^2$, $Q_{22} = \sigma_y^2$, $Q_{12} = 0$, giving

$$\epsilon_1 = \frac{\sigma_x - \sigma_y}{\sigma_x + \sigma_y}, \qquad \epsilon_2 = 0.$$

This follows from $D = (\sigma_x + \sigma_y)^2$.

For the same Gaussian rotated 45° (elongated axis pointing northeast), the cross-term becomes $Q_{12} = (\sigma_x^2 - \sigma_y^2)/2$ while $Q_{11} = Q_{22} = (\sigma_x^2 + \sigma_y^2)/2$, giving

$$\epsilon_1 = 0, \qquad \epsilon_2 = \frac{\sigma_x - \sigma_y}{\sigma_x + \sigma_y}.$$

This confirms that rotating the elongation axis by 45° transfers the signal from $\epsilon_1$ into $\epsilon_2$ without changing its magnitude.

### GalSim shear and unweighted moments

For a GalSim `Gaussian(sigma=σ)` with reduced shear `g1=g` applied, the semi-axes satisfy

$$\frac{a}{b} = \frac{1 + g}{1 - g}, \qquad ab = \sigma^2,$$

from which

$$\frac{a - b}{a + b} = g.$$

Since the unweighted moments of a Gaussian give $\epsilon_1 = (a-b)/(a+b)$, the estimator should return $\epsilon_1 = g_1$ exactly (up to discretisation errors of the stamp). This provides a clean, parameter-free check of the moment formula.

### Tangential shear rotation

For a source at position angle $\phi$ relative to a halo, the tangential and cross ellipticities are

$$\epsilon_t = -(\epsilon_1 \cos 2\phi + \epsilon_2 \sin 2\phi), \qquad \epsilon_\times = -(-\epsilon_1 \sin 2\phi + \epsilon_2 \cos 2\phi).$$

Three exact cases are used:

| Source direction | $\phi$ | $\epsilon_t$ | $\epsilon_\times$ |
|---|---|---|---|
| East ($x > 0$) | 0 | $-\epsilon_1$ | $-\epsilon_2$ |
| North ($y > 0$) | $\pi/2$ | $+\epsilon_1$ | $+\epsilon_2$ |
| Northeast | $\pi/4$ | $-\epsilon_2$ | $+\epsilon_1$ |

### Mean tangential shear standard error

For a bin containing $n$ values with sample standard deviation $s$, the unweighted estimator has error $\sigma = s/\sqrt{n}$. For the two-element case $\{0.4, 0.6\}$ this gives $s = \sqrt{0.02}$ and $\sigma = \sqrt{0.02}/\sqrt{2} = 0.1$ exactly.

---

## test_ellipticity.py

### `TestPixelGrids` (6 tests)

Verifies that `_pixel_grids` returns arrays of the correct shape and that the coordinate system is correctly centred.

- **`test_shape_matches_image`** — output arrays have the same shape as `image.array`.
- **`test_grids_sum_to_zero`** — both grids are symmetric about zero, so their element sums vanish regardless of whether the stamp has odd or even pixel count.
- **`test_center_pixel_is_zero_odd_stamp`** — for an odd $11 \times 11$ stamp, the centre pixel at index `[5, 5]` has $x = y = 0$.
- **`test_range_even_stamp`** — for a $10 \times 10$ stamp, $x$ spans $[-4.5, +4.5]$ in unit steps (the centre falls between pixels 4 and 5).
- **`test_x_varies_along_columns`** — confirms the meshgrid orientation: all pixels in the same column share the same $x$ value.
- **`test_y_varies_along_rows`** — confirms the meshgrid orientation: all pixels in the same row share the same $y$ value.

---

### `TestGaussianWeight` (4 tests)

Verifies that `_gaussian_weight` implements $W = \exp(-(dx^2 + dy^2)/2\sigma^2)$ correctly.

- **`test_peak_at_origin`** — $W(0, 0) = 1$.
- **`test_value_at_one_sigma`** — $W(\sigma, 0) = e^{-1/2}$.
- **`test_symmetric`** — $W(dx, dy) = W(-dx, -dy)$.
- **`test_strictly_positive`** — weight is positive everywhere.

---

### `TestUnweightedEllipticity` (8 tests)

Tests `unweighted_ellipticity` against the closed-form Gaussian results derived above.

- **`test_circular_gives_zero`** — noiseless GalSim Gaussian with no shear returns $(\epsilon_1, \epsilon_2) \approx (0, 0)$ to $< 10^{-6}$.
- **`test_zero_flux_returns_nan`** — zero-valued stamp returns `(nan, nan)`.
- **`test_axis_aligned_gaussian_e1`** — analytically constructed Gaussian with $\sigma_x = 12$, $\sigma_y = 4$ (on a $96 \times 96$ stamp to reduce truncation error) returns $\epsilon_1 = 0.5 \pm 0.1\%$.
- **`test_axis_aligned_gaussian_negative_e1`** — same with $\sigma_x < \sigma_y$ gives negative $\epsilon_1$.
- **`test_rotated_45_gaussian_gives_pure_e2`** — 45°-rotated Gaussian transfers elongation signal from $\epsilon_1$ into $\epsilon_2$.
- **`test_sheared_galsim_gaussian`** — GalSim Gaussian with `g1=0.3` applied returns $\epsilon_1 = 0.3$ to $< 10^{-5}$ relative error; demonstrates the exact $\epsilon_1 = g_1$ identity.
- **`test_g2_shear`** — same identity for the $\epsilon_2$ component.
- **`test_result_bounded_by_one`** — for any valid stamp, $|\epsilon| \le 1$.

---

### `TestWeightedEllipticity` (6 tests)

Tests `weighted_ellipticity_raw`. The weighted estimator is biased relative to the unweighted one (the circular weight suppresses the signal), so these tests check sign and qualitative behaviour rather than exact values.

- **`test_circular_gives_zero`** — circular galaxy returns $(\epsilon_1, \epsilon_2) \approx (0, 0)$ (symmetry argument; exact to $< 10^{-5}$).
- **`test_zero_flux_returns_nan`** — zero stamp returns `(nan, nan)`.
- **`test_g1_shear_gives_positive_e1`** — positive applied shear produces positive $\epsilon_1$ (correct sign).
- **`test_g2_shear_gives_positive_e2`** — positive $g_2$ shear produces positive $\epsilon_2$.
- **`test_result_bounded_by_one`** — physical bound enforced.
- **`test_wider_weight_increases_response`** — increasing the weight FWHM from 2 to 8 pixels enlarges the raw $\epsilon_1$ response, because a wider weight captures more of the galaxy's ellipticity signal.

---

### `TestHSMEllipticity` (6 tests)

Tests `hsm_ellipticity` using noiseless GalSim Gaussians. HSM should recover the input shear to very high precision on such profiles.

- **`test_circular_gives_zero_shear`** — circular Gaussian gives $(g_1, g_2) \approx (0, 0)$.
- **`test_circular_gives_positive_size`** — $\sigma_g > 0$ for any galaxy with signal.
- **`test_g1_recovered`** — `g1=0.2` recovered to $< 10^{-4}$ relative error.
- **`test_g2_recovered`** — `g2=0.15` recovered to $< 10^{-4}$ relative error.
- **`test_empty_image_returns_nan`** — zero stamp returns `(nan, nan, nan)`.
- **`test_larger_galaxy_gives_larger_sigma`** — $\sigma_g$ increases monotonically with galaxy size.

---

### `TestOLSThroughOrigin` (4 tests)

Tests `_ols_through_origin`, the force-through-origin regression used to compute method-vs-HSM slopes.

- **`test_perfect_slope`** — $y = 2x$ gives $R = 2$ and $\sigma_R = 0$ exactly.
- **`test_slope_with_noise`** — $y = 3x + \mathcal{N}(0, 0.1)$ with 2000 points recovers $R \approx 3$ to $< 1\%$.
- **`test_nan_values_are_filtered`** — `nan` entries in `x` or `y` are excluded before fitting.
- **`test_negative_slope`** — negative slopes ($y = -0.5x$) are recovered correctly.

---

### `TestApplySimulationCalibration` (5 tests)

Tests `apply_simulation_calibration`.

- **`test_unit_response_is_identity`** — calibration with $R = 1$ everywhere leaves ellipticities unchanged.
- **`test_half_response_doubles_ellipticity`** — calibration with $R = 0.5$ doubles both components.
- **`test_nan_ellipticity_stays_nan`** — galaxies with `nan` raw ellipticity remain `nan` after calibration.
- **`test_nan_sigma_g_uses_median`** — a galaxy with `nan` $\sigma_g$ is assigned the median $\sigma_g$ of the sample before interpolation; with $R = 1$ everywhere this leaves the ellipticity unchanged.
- **`test_interpolation_at_boundary`** — a $\sigma_g$ far above the calibration range is clamped to the highest calibration node (boundary behaviour of `np.interp`).

---

## test_shear.py

### `TestModuleConstants` (7 tests)

Verifies the correctness of the module-level constants defined at import time.

- **`test_bin_edges_has_nbins_plus_one_elements`** — `len(BIN_EDGES) == N_BINS + 1`.
- **`test_bin_centres_has_nbins_elements`** — `len(BIN_CENTRES) == N_BINS`.
- **`test_bin_centres_are_geometric_means`** — checks `BIN_CENTRES[k] == sqrt(BIN_EDGES[k] * BIN_EDGES[k+1])` element-wise; this is the correct log-space centre.
- **`test_bin_edges_strictly_increasing`** — ensures the bin structure is valid (monotone bin edges).
- **`test_bin_edges_span_expected_range`** — first edge is 100 arcsec, last is 3600 arcsec.
- **`test_colours_has_all_three_methods`** — `COLOURS` contains keys `unweighted`, `weighted`, `hsm`.
- **`test_labels_has_all_three_methods`** — same for `LABELS`.

---

### `TestLoadPositions` (3 tests)

Tests `load_positions` using `tmp_path` (pytest's built-in temporary-directory fixture) to avoid any dependency on the real data files.

- **`test_loads_two_column_data`** — round-trip: write `np.savetxt`, read with `load_positions`, check values.
- **`test_output_shape`** — 50 rows give a `(50, 2)` array.
- **`test_values_are_preserved`** — negative and decimal coordinates survive the text round-trip.

---

### `TestComputeTangentialEllipticities` (7 tests)

Uses the three exact angle cases from the analytic reference section to verify the rotation formula. All tests place a single halo at the origin and a single source at a known angle.

- **`test_source_east_pure_e1`** — $\phi = 0$: verifies $\epsilon_t = -\epsilon_1$ and $\epsilon_\times = 0$.
- **`test_source_north_pure_e1`** — $\phi = \pi/2$: verifies $\epsilon_t = +\epsilon_1$.
- **`test_source_northeast_pure_e2`** — $\phi = \pi/4$, $\epsilon_2 = 1$: verifies $\epsilon_t = -\epsilon_2$ and $\epsilon_\times = 0$.
- **`test_source_northeast_pure_e1`** — $\phi = \pi/4$, $\epsilon_1 = 1$: verifies $\epsilon_t = 0$ and $\epsilon_\times = +\epsilon_1$.
- **`test_separation_formula`** — a 3–4–5 right triangle gives $r = 5$ arcsec, independently of the ellipticity.
- **`test_output_shape_multiple_halos_sources`** — with $N_h = 7$ halos and $N_s = 13$ sources the output arrays have length $91 = 7 \times 13$.
- **`test_zero_ellipticity_gives_zero_shear`** — zero input ellipticity gives zero tangential shear at any angle.

---

### `TestMeanTangentialShear` (7 tests)

Uses a two-bin layout `[0, 100)` and `[100, 1000)` arcsec with four known pairs to test every branch of `mean_tangential_shear`.

- **`test_mean_is_correct`** — mean of $\{0.4, 0.6\}$ is $0.5$; mean of $\{0.3, 0.7\}$ is $0.5$.
- **`test_n_pairs_counted_correctly`** — both bins contain exactly 2 pairs.
- **`test_error_is_standard_error_on_mean`** — for $\{0.4, 0.6\}$: $\sigma = \sqrt{0.02}/\sqrt{2} = 0.1$.
- **`test_single_pair_returns_nan`** — a bin with only one finite pair cannot estimate a variance, so both `gt` and `gt_err` are `nan`, but `n=1`.
- **`test_nan_pairs_excluded_from_count`** — one finite and one `nan` pair gives $n = 1$ and hence `nan` output.
- **`test_empty_bin_returns_nan`** — no pairs in a bin gives $n = 0$ and `nan`.
- **`test_uniform_values_give_zero_error`** — all-identical ellipticities give zero standard deviation and hence zero uncertainty.

---

### `TestEstimateShapeNoiseVariance` (4 tests)

- **`test_known_variance`** — $\{1, -1\}$ has variance 2 (with `ddof=1`).
- **`test_nan_values_excluded`** — inserting a `nan` does not change the result.
- **`test_zero_variance_for_constant`** — all-equal values give zero variance.
- **`test_returns_float`** — return type is a Python `float`, not a numpy scalar.

---

### `TestOptimalWeights` (4 tests)

- **`test_all_equal_for_uniform_noise`** — when all ellipticities have the same variance, all weights are identical.
- **`test_nan_pairs_get_zero_weight`** — `nan` pair receives weight 0.
- **`test_finite_pairs_get_positive_weight`** — non-`nan` pairs receive positive weight.
- **`test_weights_equal_one_over_variance`** — for $\{1, -1\}$ with variance 2, weight is $1/2$.

---

### `TestBuildNFWTheory` (6 tests)

Tests the wrapper around `NFWHalo_theory.get_tangential_shear` at `BIN_CENTRES`.

- **`test_returns_nbins_values`** — output length equals `N_BINS`.
- **`test_all_positive`** — tangential shear is positive for all bins (mass concentrates matter, always producing a tangential shear signal).
- **`test_decreasing_with_radius`** — the NFW profile produces a profile that decreases with angular separation.
- **`test_higher_mass_gives_stronger_shear`** — at fixed $z_s$, a factor-10 increase in mass gives strictly larger shear at all radii.
- **`test_higher_source_redshift_gives_stronger_shear`** — more distant sources are lensed more strongly by the same halo.
- **`test_cosmology_parameters_accepted`** — passing explicit `concentration`, `halo_redshift`, `omega_m`, `omega_lam` raises no error.

---

### `TestLogLikelihood` (5 tests)

- **`test_perfect_fit_gives_zero`** — when the data equal the theory prediction exactly, all residuals are zero and $\ln L = 0$.
- **`test_imperfect_fit_gives_negative_lnL`** — data far from the theory prediction gives $\ln L < 0$.
- **`test_larger_residual_gives_smaller_lnL`** — adding a larger offset to the data gives a more negative log-likelihood.
- **`test_nan_bins_excluded`** — a `nan` in the data array is silently dropped; the remaining bins still give $\ln L = 0$ for a perfect fit.
- **`test_all_nan_returns_neg_inf`** — if no valid bins remain, `log_likelihood` returns $-\infty$ (convention for an uninformative observation).
