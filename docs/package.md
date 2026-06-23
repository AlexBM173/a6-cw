# a6cw Package Reference

The `a6cw` package contains all analysis code for the A6 weak lensing coursework.
It is organised into two modules:

| Module | Purpose |
|---|---|
| [`a6cw.ellipticity`](#a6cwellipticity) | Galaxy shape estimators and response calibration |
| [`a6cw.shear`](#a6cwshear) | Shear profiles, NFW theory, posterior inference |

---

## a6cw.ellipticity

Implements three estimators of galaxy ellipticity from postage-stamp images, plus a simulation-based calibration scheme for the weighted-moment estimator.

### Ellipticity convention

All three estimators return ellipticities in the *distortion* convention. The quadrupole moments of the surface brightness $I(x, y)$ are

$$Q_{ij} = \frac{\int dx\,dy\; I(x,y)\,(x_i - \bar{x}_i)(x_j - \bar{x}_j)}{\int dx\,dy\; I(x,y)},$$

with centroid

$$\bar{x}_i = \frac{\int dx\,dy\; I(x,y)\,x_i}{\int dx\,dy\; I(x,y)}.$$

The two ellipticity components are

$$\epsilon_1 = \frac{Q_{11} - Q_{22}}{D}, \qquad \epsilon_2 = \frac{2\,Q_{12}}{D}, \qquad D = Q_{11} + Q_{22} + 2\sqrt{Q_{11}Q_{22} - Q_{12}^2}.$$

For a pure ellipse with semi-major and semi-minor axes $a \ge b$, this gives $|\epsilon| = (a-b)/(a+b)$ and $\epsilon_1 > 0$ ($\epsilon_2 = 0$) for an east–west elongation.

---

### Private helpers

#### `_pixel_grids(image) → (x, y)`

Returns two 2-D arrays of pixel offsets from the stamp centre, both shaped `(ny, nx)`.

| Name | Type | Description |
|---|---|---|
| `image` | `galsim.Image` | Input postage stamp |

**Returns** `(x, y)`: pixel $x$ varies along columns, pixel $y$ along rows. For a stamp of width $n_x$ pixels, column $j$ has $x = j - (n_x-1)/2$.

---

#### `_gaussian_weight(dx, dy, sigma) → W`

Evaluates the circular Gaussian weight function

$$W(x, y) = \exp\!\left(-\frac{dx^2 + dy^2}{2\sigma^2}\right).$$

| Parameter | Type | Description |
|---|---|---|
| `dx`, `dy` | `np.ndarray` | Offsets from the weighted centroid (pixels) |
| `sigma` | `float` | Gaussian standard deviation (pixels) |

**Returns** `W`: array of the same shape as `dx`.

---

### Public API

#### `unweighted_ellipticity(image) → (ε₁, ε₂)`

Estimates $(\epsilon_1, \epsilon_2)$ from unweighted second-order moments computed directly on the pixel grid. The centroid is found from unweighted first moments.

Measurements where the determinant $Q_{11}Q_{22} - Q_{12}^2 < 0$ (noise-dominated), or where $|\epsilon| > 1$ (unphysical for non-negative profiles), are returned as `(nan, nan)`.

| Parameter | Type | Description |
|---|---|---|
| `image` | `galsim.Image` | Galaxy postage stamp |

**Returns** `(float, float)`: $(\epsilon_1, \epsilon_2)$, or `(nan, nan)` on failure.

---

#### `weighted_ellipticity_raw(image, fwhm_px=4.0, max_iter=20, tol=1e-3) → (ε₁, ε₂)`

Estimates $(\epsilon_1, \epsilon_2)$ from weighted second-order moments, replacing $I(x,y)$ by $W(x,y)\,I(x,y)$ in all integrals. The weight function is a circular Gaussian

$$W(x, y) = \exp\!\left(-\frac{(x - \bar{x}^W)^2 + (y - \bar{y}^W)^2}{2\sigma_w^2}\right), \qquad \sigma_w = \frac{\text{FWHM}}{2\sqrt{2\ln 2}}.$$

The weighted centroid $(\bar{x}^W, \bar{y}^W)$ is found iteratively: the weight is re-centred on the current centroid estimate until the shift between iterations is less than `tol` pixels. This ensures the weight is self-consistently centred.

The raw ellipticities returned by this function are biased relative to the true shear because the fixed circular weight suppresses the tangential ellipticity signal. Use `apply_simulation_calibration` to correct for this.

| Parameter | Type | Description |
|---|---|---|
| `image` | `galsim.Image` | Galaxy postage stamp |
| `fwhm_px` | `float` | FWHM of the Gaussian weight (pixels); default `4.0` |
| `max_iter` | `int` | Maximum centroid iterations; default `20` |
| `tol` | `float` | Centroid convergence threshold (pixels); default `1e-3` |

**Returns** `(float, float)`: raw $(\epsilon_1, \epsilon_2)$, or `(nan, nan)` on failure.

---

#### `hsm_ellipticity(image) → (g₁, g₂, σ_g)`

Estimates the galaxy shape using the Hirata–Seljak–Mandelbaum (HSM) adaptive-moments algorithm as implemented in `galsim.hsm.FindAdaptiveMom`. HSM iteratively adapts an elliptical Gaussian weight function to match the galaxy profile. The returned shear components are in the *reduced shear* convention: for semi-axes $a \ge b$, $|g| = (a - b)/(a + b)$.

| Parameter | Type | Description |
|---|---|---|
| `image` | `galsim.Image` | Galaxy postage stamp |

**Returns** `(float, float, float)`: $(g_1, g_2, \sigma_g)$ where $\sigma_g$ is the best-fit adaptive-moment size in pixels. Returns `(nan, nan, nan)` if HSM fails (e.g. on a noise-dominated stamp).

---

#### `calibrate_weighted_response(fwhm_px=4.0, stamp_size=48, pixel_scale=1.0, shear_cal=0.05, n_per_hlr=1000, seed=42) → (σ_g_nodes, R_nodes)`

Measures the shear *response* $R(\sigma_g)$ of the weighted-moment estimator by simulation. For each galaxy half-light radius in a pre-defined grid, `n_per_hlr` noiseless GalSim Exponential-profile galaxies are drawn with a known applied shear $g_1 = \texttt{shear\_cal}$, then measured with `weighted_ellipticity_raw`. The response is

$$R = \frac{\langle \epsilon_1^{\rm measured} \rangle}{g_1^{\rm applied}}.$$

The interpolation axis is the median HSM adaptive-moment size $\sigma_g$ in each size bin.

| Parameter | Type | Description |
|---|---|---|
| `fwhm_px` | `float` | Must match the value used in `weighted_ellipticity_raw` |
| `stamp_size` | `int` | Stamp side length in pixels; must match the data |
| `pixel_scale` | `float` | Arcsec per pixel |
| `shear_cal` | `float` | Applied calibration shear (should be small, $\ll 1$) |
| `n_per_hlr` | `int` | Simulated galaxies per half-light radius grid point |
| `seed` | `int` | RNG seed for reproducibility |

**Returns** `(σ_g_nodes, R_nodes)`: both `(N_grid,)` arrays sorted by $\sigma_g$, with any failed grid points removed.

---

#### `apply_simulation_calibration(epsilon, sigma_gs, sigma_g_nodes, R_nodes) → epsilon_corr`

Corrects raw weighted-moment ellipticities for the per-galaxy shear response by dividing each galaxy's $(\epsilon_1, \epsilon_2)$ by its interpolated response $R_i$:

$$\epsilon^{\rm corr}_{i} = \frac{\epsilon^{\rm raw}_{i}}{R(\sigma_{g,i})}.$$

`np.interp` is used, which clamps to the boundary response values for galaxies outside the calibrated $\sigma_g$ range. Galaxies with `nan` $\sigma_g$ are assigned the median $\sigma_g$ before interpolation.

| Parameter | Type | Description |
|---|---|---|
| `epsilon` | `np.ndarray` | `(N, 2)` raw ellipticity array |
| `sigma_gs` | `np.ndarray` | `(N,)` per-galaxy HSM sizes (pixels) |
| `sigma_g_nodes` | `np.ndarray` | `(M,)` calibration x-axis (sorted) |
| `R_nodes` | `np.ndarray` | `(M,)` calibration y-axis |

**Returns** `(N, 2)` calibrated ellipticity array; `nan` where the raw ellipticity is `nan`.

---

#### `measure_all_ellipticities(stamps_path) → dict`

Top-level function: loads all galaxy stamps from `stamps_path` and runs all three estimators. For the weighted estimator, the response calibration is computed via `calibrate_weighted_response` and applied via `apply_simulation_calibration`.

| Parameter | Type | Description |
|---|---|---|
| `stamps_path` | `Path` | Path to a multi-extension FITS file of galaxy stamps |

**Returns** `dict` with keys `"unweighted"`, `"weighted"`, `"hsm"`, each an `(N, 2)` array of $(\epsilon_1, \epsilon_2)$ values (`nan` on failure).

---

#### `_ols_through_origin(x, y) → (R, σ_R)`

Force-through-origin ordinary least-squares regression: finds the slope $R$ minimising $\sum_i (y_i - R x_i)^2$. The closed-form solution is

$$R = \frac{\sum_i x_i y_i}{\sum_i x_i^2}, \qquad \sigma_R = \sqrt{\frac{\sum_i(y_i - Rx_i)^2 / (n-1)}{\sum_i x_i^2}}.$$

`nan` values in `x` or `y` are silently excluded. Used internally by `plot_ellipticity_comparison` to compute shear response slopes.

---

#### `plot_ellipticity_comparison(results, output_dir)`

Generates three diagnostic figures and saves them to `output_dir`:

1. **Scatter $\epsilon_1$ vs $\epsilon_2$** — one panel per method.
2. **Histograms** — $\epsilon_1$ and $\epsilon_2$ distributions per method.
3. **Method vs HSM** — direct scatter comparison with 1:1 line, Pearson correlation $r$, and force-through-origin regression slope $R \pm \sigma_R$.

| Parameter | Type | Description |
|---|---|---|
| `results` | `dict` | Output of `measure_all_ellipticities` |
| `output_dir` | `Path` | Directory to write PNG files |

**Saves** `q1_scatter_e1_e2.png`, `q1_histograms.png`, `q1_comparison_vs_hsm.png`.

---

## a6cw.shear

Implements the full weak lensing shear measurement pipeline: tangential shear estimation, binning, NFW theoretical predictions, and Bayesian mass inference.

### Module-level constants

| Name | Value | Description |
|---|---|---|
| `N_BINS` | `10` | Number of radial bins |
| `BIN_EDGES` | `np.logspace(log10(100), log10(3600), 11)` | Bin edges in arcseconds |
| `BIN_CENTRES` | `sqrt(BIN_EDGES[:-1] * BIN_EDGES[1:])` | Geometric-mean bin centres |
| `COLOURS` | `dict` | Matplotlib colours per estimator method |
| `LABELS` | `dict` | Display labels per estimator method |

`BIN_CENTRES` are the *geometric* (log-space) means of adjacent bin edges, appropriate for log-spaced bins.

---

### Public API

#### `load_positions(path) → pos`

Loads `(x, y)` sky positions in arcseconds from a two-column whitespace-delimited text file.

| Parameter | Type | Description |
|---|---|---|
| `path` | `Path` | Input file path |

**Returns** `np.ndarray` of shape `(N, 2)`.

---

#### `compute_tangential_ellipticities(halo_pos, source_pos, epsilon_1, epsilon_2) → (r, ε_t, ε_×)`

Rotates source ellipticities into the tangential–cross frame relative to each halo. For each halo–source pair:

$$\phi = \arctan2(\Delta y,\, \Delta x), \qquad \Delta x = x_s - x_h, \quad \Delta y = y_s - y_h,$$

$$\epsilon_t = -(\epsilon_1 \cos 2\phi + \epsilon_2 \sin 2\phi),$$

$$\epsilon_\times = -(-\epsilon_1 \sin 2\phi + \epsilon_2 \cos 2\phi).$$

$\epsilon_t$ is the *E-mode* (tangential) component; gravitational lensing contributes only to $\epsilon_t$. $\epsilon_\times$ is the *B-mode* (cross) component and should average to zero in the absence of systematics.

| Parameter | Type | Description |
|---|---|---|
| `halo_pos` | `(N_h, 2)` array | Halo $(x, y)$ positions [arcsec] |
| `source_pos` | `(N_s, 2)` array | Source $(x, y)$ positions [arcsec] |
| `epsilon_1`, `epsilon_2` | `(N_s,)` arrays | Source ellipticity components |

**Returns** `(r, ε_t, ε_×)`: three flat arrays of length $N_h \times N_s$ containing the separation [arcsec] and the two projected ellipticity components for every halo–source pair.

---

#### `mean_tangential_shear(r_pairs, epsilon_t_pairs, bin_edges, weights=None) → (γ_t, σ_{γ_t}, n)`

Bins tangential ellipticities and returns the mean tangential shear profile. In the weak-lensing regime $\langle \epsilon_t \rangle = \gamma_t$, so the mean tangential ellipticity directly estimates the shear.

**Unweighted** (`weights=None`):

$$\hat\gamma_t^k = \langle \epsilon_t \rangle_k, \qquad \sigma_k = \frac{\mathrm{std}(\epsilon_t)_k}{\sqrt{n_k}}.$$

**Inverse-variance weighted** (`weights = 1/\sigma_\epsilon^2` per pair):

$$\hat\gamma_t^k = \frac{\sum_i w_i \epsilon_{t,i}}{\sum_i w_i}, \qquad \sigma_k = \frac{1}{\sqrt{\sum_i w_i}}.$$

Bins with fewer than two valid pairs return `nan`.

| Parameter | Type | Description |
|---|---|---|
| `r_pairs` | `(M,)` array | Separations for all halo–source pairs [arcsec] |
| `epsilon_t_pairs` | `(M,)` array | Tangential ellipticities |
| `bin_edges` | `(N_bins+1,)` array | Radial bin boundaries |
| `weights` | `(M,)` array or `None` | Per-pair weights; `None` for simple mean |

**Returns** `(γ_t, σ_{γ_t}, n)`: mean shear, 1-$\sigma$ uncertainty, and pair count per bin.

---

#### `estimate_shape_noise_variance(epsilon_t_pairs) → σ²`

Estimates the per-component shape-noise variance from all valid tangential ellipticities:

$$\hat\sigma_\epsilon^2 = \mathrm{Var}(\epsilon_t),$$

with `ddof=1`. Because the lensing signal is much smaller than the intrinsic ellipticity scatter, $\hat\sigma_\epsilon^2$ is a good proxy for the per-galaxy noise variance.

| Parameter | Type | Description |
|---|---|---|
| `epsilon_t_pairs` | array | All tangential ellipticity values (finite ones used) |

**Returns** `float`.

---

#### `optimal_weights(epsilon_t_pairs) → w`

Returns per-pair inverse-variance weights $w_i = 1/\hat\sigma_\epsilon^2$ computed from `estimate_shape_noise_variance`. Pairs with `nan` ellipticity receive weight 0. Because all galaxies share the same intrinsic shape-noise variance in this simulation, the weighted estimator reduces to the simple mean — but the framework generalises to real surveys where per-source variances differ.

---

#### `plot_tangential_shear_profiles(profiles, output_dir, filename='q2c_tangential_shear_profiles.png')`

Plots $\hat\gamma_t(\theta)$ with 1-$\sigma$ error bars for all three shape estimators on a log–log scale.

| Parameter | Type | Description |
|---|---|---|
| `profiles` | `dict` | Keyed by method; each value has `'gt_simple'`, `'gt_err_simple'` |
| `output_dir` | `Path` | Output directory |
| `filename` | `str` | Output filename |

---

#### `plot_cross_shear_null_test(profiles, output_dir, filename='q2c_cross_shear_null_test.png')`

Plots the B-mode (cross-shear) profile $\langle\gamma_\times\rangle(\theta)$ on a linear $y$-scale. Gravitational lensing contributes only E-mode signal, so a consistent-with-zero cross-shear profile is a necessary null test for systematic errors.

---

#### `build_nfw_theory(mass, source_redshift, concentration=4.0, halo_redshift=0.3, omega_m=0.3, omega_lam=0.7) → γ_t^theory`

Computes the predicted mean tangential shear at `BIN_CENTRES` for an NFW halo using `NFWHalo_theory` (from `nfw_theory.py`). The default cosmological and halo parameters match the simulation truth.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `mass` | `float` | — | Halo mass $[M_\odot]$ |
| `source_redshift` | `float` | — | Source galaxy redshift |
| `concentration` | `float` | `4.0` | NFW concentration $c$ |
| `halo_redshift` | `float` | `0.3` | Lens redshift $z_l$ |
| `omega_m` | `float` | `0.3` | Matter density $\Omega_m$ |
| `omega_lam` | `float` | `0.7` | Dark energy density $\Omega_\Lambda$ |

**Returns** `(N_bins,)` array of predicted $\gamma_t$ at `BIN_CENTRES`.

---

#### `log_likelihood(gt_data, gt_err, mass, source_redshift) → ln L`

Evaluates the Gaussian log-likelihood

$$\ln L = -\frac{1}{2} \sum_k \left(\frac{\hat\gamma_t^k - \gamma_t^{\rm theory}(k)}{\sigma_k}\right)^2,$$

summing only over bins where both data and theory are finite. Returns $-\infty$ if no valid bins remain.

| Parameter | Type | Description |
|---|---|---|
| `gt_data` | `(N_bins,)` | Measured $\hat\gamma_t$ per bin |
| `gt_err` | `(N_bins,)` | 1-$\sigma$ uncertainties |
| `mass` | `float` | Halo mass for the theory prediction |
| `source_redshift` | `float` | Source redshift for the theory prediction |

**Returns** `float`.

---

#### `compute_posterior_grid(gt_data, gt_err, mass_grid, zs_grid) → log_post`

Evaluates the (unnormalised) log-posterior on a 2-D grid of (mass, $z_s$) under a flat prior on both parameters. Each grid point calls `log_likelihood` with the simulation-truth cosmology defaults.

| Parameter | Type | Description |
|---|---|---|
| `gt_data` | `(N_bins,)` | Measured shear profile |
| `gt_err` | `(N_bins,)` | Per-bin 1-$\sigma$ uncertainties |
| `mass_grid` | `(N_m,)` | Mass values to evaluate |
| `zs_grid` | `(N_z,)` | Source redshift values to evaluate |

**Returns** `(N_m, N_z)` log-posterior array (uninitialised entries are $-\infty$).

---

#### `plot_joint_posterior(log_post, mass_grid, zs_grid, output_dir, filename='q2d_joint_posterior.png') → (mass_MAP, zs_MAP, m_lo, m_hi, zs_lo, zs_hi)`

Plots the joint posterior $p(M, z_s \mid \text{data})$ as filled contours at the 68% and 95% credible levels, alongside marginalised 1-D posteriors for each parameter. The 2-D contour levels are computed from the $\chi^2_{df=2}$ distribution:

$$\Delta\ln L = -\frac{1}{2}\Delta\chi^2, \quad \Delta\chi^2_{68\%} = 2.30, \quad \Delta\chi^2_{95\%} = 6.18.$$

The 1-D credible intervals are computed by trapezoidal integration of the marginalised posteriors.

**Returns** the MAP mass and redshift, and the 68% credible interval boundaries for both parameters.

---

#### `plot_mass_posterior_fixed_zs(log_post, mass_grid, zs_grid, fixed_zs, output_dir, filename='q2d_mass_posterior_fixed_zs.png') → (mass_MAP, mass_lo, mass_hi)`

Slices the 2-D log-posterior at the nearest grid point to `fixed_zs`, normalises, and plots the resulting 1-D mass posterior with a 68% credible interval shaded.

---

#### `plot_best_fit_overlay(gt_data, gt_err, mass_map, zs_map, mass_map_zs06, output_dir, filename='q2d_best_fit_overlay.png')`

Overplots two NFW theory curves — the joint MAP and the best-fit mass at fixed $z_s = 0.6$ — on the measured tangential shear profile.
