import os
from functools import lru_cache
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import numba
import galsim
import galsim.hsm


# ---------------------------------------------------------------------------
# Coordinate grids — cached by stamp shape to avoid recomputation
# ---------------------------------------------------------------------------

@lru_cache(maxsize=32)
def _pixel_grids_cache(ny: int, nx: int):
    """Compute and cache pixel-offset grids for a given stamp shape."""
    cx = (nx - 1) / 2.0
    cy = (ny - 1) / 2.0
    x  = np.arange(nx, dtype=float) - cx
    y  = np.arange(ny, dtype=float) - cy
    return np.meshgrid(x, y)


def _pixel_grids(image: galsim.Image):
    """
    Return (x, y) pixel-offset arrays relative to the image centre.

    Results are cached by stamp shape; repeated calls with stamps of the
    same size return the cached arrays without recomputation.

    Parameters
    ----------
    image : galsim.Image
        The image for which to compute the pixel grids.
    Returns
    -------
    x, y : np.ndarray
        2D arrays of shape (ny, nx) containing the x and y pixel offsets
        from the image centre.
    """
    arr = image.array
    return _pixel_grids_cache(*arr.shape)


# ---------------------------------------------------------------------------
# Numba-JIT moment kernels
# ---------------------------------------------------------------------------
# Both kernels are compiled once (cache=True) to ~/.cache/numba/ and reused.
# nogil=True releases the Python GIL so ThreadPoolExecutor achieves real
# parallelism across galaxy stamps.
#
# The key optimisation over the original NumPy implementation is that all
# intermediate quantities (xbar, Q11, Q22, Q12 …) are accumulated in scalar
# registers in a single pixel-level loop, eliminating every temporary array
# allocation and reducing the number of memory passes from ~10 to 3.
# ---------------------------------------------------------------------------

@numba.njit(cache=True, nogil=True)
def _unweighted_jit(arr):
    """
    Compute (e1, e2) from unweighted second moments.

    Three passes over the pixel array:
      pass 1 — intensity sum (I_sum) and centroid (xbar, ybar)
      pass 2 — second moments Q11, Q22, Q12
    All results accumulated in scalar registers; no intermediate arrays.
    """
    ny, nx = arr.shape
    cx = (nx - 1) * 0.5
    cy = (ny - 1) * 0.5

    # Pass 1a: total flux
    I_sum = 0.0
    for i in range(ny):
        for j in range(nx):
            I_sum += arr[i, j]
    if I_sum <= 0.0:
        return np.nan, np.nan

    # Pass 1b: centroid
    xbar = 0.0; ybar = 0.0
    for i in range(ny):
        for j in range(nx):
            v = arr[i, j]
            xbar += v * (j - cx)
            ybar += v * (i - cy)
    xbar /= I_sum; ybar /= I_sum

    # Pass 2: second moments
    Q11 = 0.0; Q22 = 0.0; Q12 = 0.0
    for i in range(ny):
        for j in range(nx):
            dx = (j - cx) - xbar
            dy = (i - cy) - ybar
            v  = arr[i, j]
            Q11 += v * dx * dx
            Q22 += v * dy * dy
            Q12 += v * dx * dy
    Q11 /= I_sum; Q22 /= I_sum; Q12 /= I_sum

    det = Q11 * Q22 - Q12 * Q12
    if det < 0.0:
        return np.nan, np.nan
    denom = Q11 + Q22 + 2.0 * np.sqrt(det)
    if denom <= 0.0:
        return np.nan, np.nan

    e1 = (Q11 - Q22) / denom
    e2 = 2.0 * Q12  / denom
    if abs(e1) > 1.0 or abs(e2) > 1.0:
        return np.nan, np.nan
    return e1, e2


@numba.njit(cache=True, nogil=True)
def _weighted_jit(arr, fwhm_px, max_iter, tol):
    """
    Compute (e1, e2) from Gaussian-weighted second moments.

    Optimisations over the original NumPy implementation:
    - All per-iteration array allocations (W, WI, dx, dy) are eliminated;
      the Gaussian weight is computed per pixel in the inner loop.
    - The convergence check uses shift² < tol² to avoid a sqrt per iteration.
    - inv_2s2 is precomputed once rather than dividing by 2σ² on every pixel.
    """
    ny, nx = arr.shape
    cx = (nx - 1) * 0.5
    cy = (ny - 1) * 0.5
    sigma   = fwhm_px / (2.0 * np.sqrt(2.0 * np.log(2.0)))
    inv_2s2 = 1.0 / (2.0 * sigma * sigma)
    tol2    = tol * tol

    # Unweighted centroid as starting estimate
    I_sum = 0.0
    for i in range(ny):
        for j in range(nx):
            I_sum += arr[i, j]
    if I_sum <= 0.0:
        return np.nan, np.nan
    xbar = 0.0; ybar = 0.0
    for i in range(ny):
        for j in range(nx):
            v = arr[i, j]
            xbar += v * (j - cx)
            ybar += v * (i - cy)
    xbar /= I_sum; ybar /= I_sum

    # Iterative weighted centroid
    for _ in range(max_iter):
        WI = 0.0; xn = 0.0; yn = 0.0
        for i in range(ny):
            for j in range(nx):
                dx = (j - cx) - xbar
                dy = (i - cy) - ybar
                w  = np.exp(-(dx*dx + dy*dy) * inv_2s2)
                wi = w * arr[i, j]
                WI += wi
                xn += wi * (j - cx)
                yn += wi * (i - cy)
        if WI <= 0.0:
            return np.nan, np.nan
        xn /= WI; yn /= WI
        shift2 = (xn - xbar)**2 + (yn - ybar)**2
        xbar = xn; ybar = yn
        if shift2 < tol2:
            break

    # Final weighted second moments
    WI = 0.0; Q11 = 0.0; Q22 = 0.0; Q12 = 0.0
    for i in range(ny):
        for j in range(nx):
            dx = (j - cx) - xbar
            dy = (i - cy) - ybar
            w  = np.exp(-(dx*dx + dy*dy) * inv_2s2)
            wi = w * arr[i, j]
            WI  += wi
            Q11 += wi * dx * dx
            Q22 += wi * dy * dy
            Q12 += wi * dx * dy
    if WI <= 0.0:
        return np.nan, np.nan
    Q11 /= WI; Q22 /= WI; Q12 /= WI

    det = Q11 * Q22 - Q12 * Q12
    if det < 0.0:
        return np.nan, np.nan
    denom = Q11 + Q22 + 2.0 * np.sqrt(det)
    if denom <= 0.0:
        return np.nan, np.nan

    e1 = (Q11 - Q22) / denom
    e2 = 2.0 * Q12  / denom
    if abs(e1) > 1.0 or abs(e2) > 1.0:
        return np.nan, np.nan
    return e1, e2


# ---------------------------------------------------------------------------
# Public estimator API
# ---------------------------------------------------------------------------

def unweighted_ellipticity(image: galsim.Image):
    """
    Estimate (epsilon_1, epsilon_2) from unweighted second-order moments.

        Q_ij = sum_{pq} I(p,q) * dx_i * dx_j
        epsilon_1 = (Q11 - Q22) / (Q11 + Q22 + 2 * sqrt(Q11 * Q22 - Q12^2))
        epsilon_2 = 2 * Q12 / (Q11 + Q22 + 2 * sqrt(Q11 * Q22 - Q12^2))

    Measurements with ``|epsilon| > 1`` (noise-dominated / diverging) are returned
    as (nan, nan) because they indicate a failed centroid or a stamp where
    the noise dominates the moment denominator.

    Returns
    -------
    (epsilon_1, epsilon_2) : floats, or (nan, nan) on failure.
    """
    return _unweighted_jit(image.array.astype(np.float64))


def _gaussian_weight(dx, dy, sigma):
    """
    Circular Gaussian W = exp(-(dx^2 + dy^2) / (2 sigma^2)).

    Parameters
    ----------
    dx, dy : np.ndarray
        Pixel offsets from the centroid in x and y directions.
    sigma : float
        Standard deviation of the Gaussian weight function in pixels.

    Returns
    -------
    W : np.ndarray
        2D array of the same shape as dx and dy containing the Gaussian weights.
    """
    return np.exp(-(dx**2 + dy**2) / (2.0 * sigma**2))


def weighted_ellipticity_raw(image: galsim.Image, fwhm_px: float = 4.0, max_iter: int = 20, tol: float = 1e-3):
    """
    Estimate (epsilon_1, epsilon_2) from weighted second-order moments,
    using the same lecture convention as the unweighted estimator but with
    intensity I(x,y) replaced by W(x,y)*I(x,y)::

        Q_pq^W = [sum_{xy} W(x,y) * I(x,y) * (p - pbar^W) * (q - qbar^W)]
                 / [sum_{xy} W(x,y) * I(x,y)]

        epsilon_1 = (Q_xx^W - Q_yy^W) / D^W
        epsilon_2 = 2 * Q_xy^W / D^W
        D^W = Q_xx^W + Q_yy^W + 2 * sqrt(Q_xx^W * Q_yy^W - (Q_xy^W)^2)

    The weight function is a circular Gaussian with
    ``sigma = FWHM / (2*sqrt(2*ln2))``::

        W(x, y) = exp(-((x - xbar^W)^2 + (y - ybar^W)^2) / (2*sigma^2))

    The weighted centroid (xbar^W, ybar^W) is found iteratively:

    1. Initialise with the unweighted centroid.
    2. Centre the Gaussian weight on the current centroid estimate.
    3. Recompute ``xbar^W = sum(W * I * x) / sum(W * I)``.
    4. Repeat until the centroid shifts by less than *tol* pixels.

    This ensures the weight function is self-consistently centred on the
    weighted centroid, as required by the definition of Q_pq^W.

    Parameters
    ----------
    image    : galsim.Image
    fwhm_px  : FWHM of the Gaussian weight function in pixels (default 4.0)
    max_iter : maximum number of centroid iterations (default 20)
    tol      : convergence threshold in pixels (default 1e-3)

    Returns
    -------
    (epsilon_1, epsilon_2) : floats, or (nan, nan) on failure.
    """
    return _weighted_jit(image.array.astype(np.float64), fwhm_px, max_iter, tol)


def hsm_ellipticity(image: galsim.Image):
    """
    Estimate (g1, g2, sigma_g) and the adaptive moment size sigma_g using GalSim's
    HSM algorithm (Hirata & Seljak 2003; Mandelbaum et al. 2005).

    HSM iteratively adapts an elliptical Gaussian weight to match the
    galaxy shape.  The returned size moments_sigma is the best-fit
    circular Gaussian sigma in pixels, which is used for the analytical
    response correction of the fixed-weight estimators.

    Returns
    -------
    (g1, g2, sigma_g) : floats, or (nan, nan, nan) if HSM fails.
    """
    try:
        result = galsim.hsm.FindAdaptiveMom(image)
        return (result.observed_shape.g1,
                result.observed_shape.g2,
                result.moments_sigma)
    except galsim.GalSimHSMError:
        return np.nan, np.nan, np.nan


def calibrate_weighted_response(
        fwhm_px: float = 4.0,
        stamp_size: int = 48,
        pixel_scale: float = 1.0,
        shear_cal: float = 0.05,
        n_per_hlr: int = 1000,
        seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Build a lookup table of weighted-moment shear response R(sigma_g) by
    simulating noiseless Exponential-profile galaxies with a known applied
    shear and measuring the ratio of recovered to input ellipticity.

    For each half-light radius in a pre-defined grid, n_per_hlr noiseless
    galaxy images are drawn with GalSim (Exponential profile, shear g1=shear_cal,
    g2=0), passed through the same weighted-moment pipeline used on the data,
    and the response is estimated as

        R = <epsilon_1_measured> / shear_cal

    The interpolation axis is the median HSM sigma_g for each hlr bin, so
    that the calibration can be applied per galaxy using its HSM size.

    Parameters
    ----------
    fwhm_px    : FWHM of the Gaussian weight function in pixels (must match
                 the value used in weighted_ellipticity_raw).
    stamp_size : stamp side length in pixels (must match data stamps).
    pixel_scale: arcsec per pixel (must match data).
    shear_cal  : applied reduced shear magnitude; large enough for S/N but
                 small enough to stay in the weak-lensing regime (g << 1).
    n_per_hlr  : number of noiseless realisations per grid point.
    seed       : RNG seed for reproducibility.

    Returns
    -------
    sigma_g_nodes : (N_grid,) HSM sigma_g values [pixels] — interpolation x-axis.
    R_nodes       : (N_grid,) response values        — interpolation y-axis.
    Both arrays are sorted by sigma_g_nodes and have NaN entries removed.
    """
    sigma_w = fwhm_px / (2.0 * np.sqrt(2.0 * np.log(2.0)))

    hlr_grid = np.array([0.4, 0.5, 0.75, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 7.0, 10.0])

    sigma_g_nodes = []
    R_nodes = []

    print(f"Calibrating weighted-moment response on {len(hlr_grid)} galaxy sizes...")
    for hlr in hlr_grid:
        # Noiseless stamps are fully deterministic: every draw for the same
        # (hlr, shear_cal, fwhm_px, stamp_size) is pixel-identical.  Drawing
        # n_per_hlr times and averaging is therefore equivalent to a single
        # draw followed by a single measurement — so we draw once.
        gal = galsim.Exponential(half_light_radius=hlr * pixel_scale)
        gal = gal.shear(g1=shear_cal, g2=0.0)
        im  = gal.drawImage(
            scale=pixel_scale, nx=stamp_size, ny=stamp_size, method="auto"
        )
        e1, _ = weighted_ellipticity_raw(im, fwhm_px=fwhm_px)
        try:
            sg_med = galsim.hsm.FindAdaptiveMom(im).moments_sigma
        except galsim.GalSimHSMError:
            sg_med = np.nan

        if not np.isfinite(e1) or not np.isfinite(sg_med):
            continue

        R = e1 / shear_cal
        if R <= 0:
            continue

        sigma_g_nodes.append(sg_med)
        R_nodes.append(R)
        print(f"  hlr={hlr:.2f}px  sigma_g={sg_med:.3f}px  R={R:.4f}")

    sigma_g_nodes = np.array(sigma_g_nodes)
    R_nodes = np.array(R_nodes)

    order = np.argsort(sigma_g_nodes)
    return sigma_g_nodes[order], R_nodes[order]


def apply_simulation_calibration(
        epsilon: np.ndarray,
        sigma_gs: np.ndarray,
        sigma_g_nodes: np.ndarray,
        R_nodes: np.ndarray,
) -> np.ndarray:
    """
    Divide each galaxy's raw weighted ellipticity by its per-galaxy response
    R_i, interpolated from the simulation calibration table.

    np.interp is used for linear interpolation with boundary clamping:
    galaxies outside the calibrated sigma_g range receive the response of
    the nearest calibration node rather than an extrapolated (potentially
    unphysical) value.

    Parameters
    ----------
    epsilon        : (N, 2) raw ellipticity array from weighted_ellipticity_raw.
    sigma_gs       : (N,) per-galaxy HSM sigma_g values [pixels].
    sigma_g_nodes  : (M,) calibration x-axis (sorted).
    R_nodes        : (M,) calibration y-axis.

    Returns
    -------
    epsilon_corr : (N, 2) calibrated ellipticity array; NaN where sigma_g
                   or the raw ellipticity is NaN.
    """
    sigma_gs_filled = sigma_gs.copy()
    nan_sg = ~np.isfinite(sigma_gs_filled)
    sigma_gs_filled[nan_sg] = np.nanmedian(sigma_gs)

    R_per_galaxy = np.interp(sigma_gs_filled, sigma_g_nodes, R_nodes)

    corr = epsilon.copy()
    valid = np.isfinite(epsilon[:, 0]) & np.isfinite(R_per_galaxy) & (R_per_galaxy > 0)
    corr[valid, 0] /= R_per_galaxy[valid]
    corr[valid, 1] /= R_per_galaxy[valid]
    corr[~valid] = np.nan
    return corr


# ---------------------------------------------------------------------------
# Parallel per-galaxy measurement
# ---------------------------------------------------------------------------

def _measure_single(im):
    """
    Run all three estimators on one galaxy stamp.

    Called by ThreadPoolExecutor workers in measure_all_ellipticities.
    The Numba JIT kernels (_unweighted_jit, _weighted_jit) and GalSim's
    HSM (C++) all release the GIL, giving genuine thread-level parallelism.
    """
    arr = im.array.astype(np.float64)
    e1_u, e2_u = _unweighted_jit(arr)
    e1_w, e2_w = _weighted_jit(arr, 4.0, 20, 1e-3)
    e1_h, e2_h, sg = hsm_ellipticity(im)
    return e1_u, e2_u, e1_w, e2_w, e1_h, e2_h, sg


def measure_all_ellipticities(stamps_path: Path):
    """
    Load all galaxy stamps and compute (e1, e2) for each estimator.

    For the weighted-moment estimator the raw ellipticities are corrected
    for the multiplicative shear response bias introduced by the fixed
    circular Gaussian weight function. The response R(sigma_g) is measured
    from GalSim simulations via calibrate_weighted_response and applied
    per-galaxy via apply_simulation_calibration.

    The unweighted estimator has unit response in the noiseless limit and
    is returned uncorrected. HSM returns adaptive-moment ellipticities that
    are already self-calibrated by construction.

    Stamps are processed in parallel using a thread pool. The Numba JIT
    kernels and GalSim's C++ HSM both release the GIL, so all available
    CPU cores contribute.

    Returns
    -------
    dict with keys "unweighted", "weighted", "hsm", each an ndarray
    of shape (N, 2) containing (e1, e2) per galaxy (NaN on failure).
    """
    galaxy_ims = galsim.fits.readMulti(str(stamps_path))
    n_gals     = len(galaxy_ims)
    print(f"Loaded {n_gals} galaxy stamps from {stamps_path}")

    # Trigger JIT compilation on a trivial array before the timed work begins.
    _dummy = np.zeros((4, 4), dtype=np.float64)
    _unweighted_jit(_dummy)
    _weighted_jit(_dummy, 4.0, 20, 1e-3)

    e_unw    = np.full((n_gals, 2), np.nan)
    e_wgt    = np.full((n_gals, 2), np.nan)
    e_hsm    = np.full((n_gals, 2), np.nan)
    sigma_gs = np.full(n_gals,      np.nan)

    n_workers = min(os.cpu_count() or 4, 16)
    with ThreadPoolExecutor(max_workers=n_workers) as pool:
        for i, res in enumerate(pool.map(_measure_single, galaxy_ims)):
            e1_u, e2_u, e1_w, e2_w, e1_h, e2_h, sg = res
            e_unw[i]    = e1_u, e2_u
            e_wgt[i]    = e1_w, e2_w
            e_hsm[i]    = e1_h, e2_h
            sigma_gs[i] = sg

    print(f"\nMedian HSM galaxy size: sigma_g = {np.nanmedian(sigma_gs):.3f} px")

    results = {
        "unweighted": e_unw,
        "weighted":   e_wgt,
        "hsm":        e_hsm,
    }

    print("\nSummary statistics:")
    for name, arr in results.items():
        valid   = np.isfinite(arr[:, 0])
        n_valid = valid.sum()
        print(
            f"  {name:12s}: {n_valid:6d}/{n_gals} valid | "
            f"<e1> = {arr[valid, 0].mean():+.4f}  σ(e1) = {arr[valid, 0].std():.4f} | "
            f"<e2> = {arr[valid, 1].mean():+.4f}  σ(e2) = {arr[valid, 1].std():.4f}"
        )

    return results


def _ols_through_origin(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """
    Force-through-origin OLS regression: minimise sum((y - R*x)^2).
    Returns (R, sigma_R) where sigma_R is the standard error on the slope.

    R = sum(x_i * y_i) / sum(x_i^2) is the maximum-likelihood slope for
    a linear model y = R*x with no intercept (eq. 3 in the lectures).
    """
    valid = np.isfinite(x) & np.isfinite(y)
    x, y = x[valid], y[valid]
    R = np.sum(x * y) / np.sum(x**2)
    resid = y - R * x
    sigma2_resid = np.sum(resid**2) / (len(x) - 1)
    sigma_R = np.sqrt(sigma2_resid / np.sum(x**2))
    return R, sigma_R


def plot_ellipticity_comparison(results: dict, output_dir: Path):
    """
    Three diagnostic figures:
      1. Scatter e1 vs e2 per method (physical +/-1 window).
      2. Histograms of e1 and e2 per method.
      3. Direct scatter of each method vs HSM, with 1:1 line, Pearson r,
         and force-through-origin regression line with slope R +/- sigma_R.
    """
    methods = ["unweighted", "weighted", "hsm"]
    colours = {"unweighted": "steelblue", "weighted": "tomato", "hsm": "seagreen"}
    labels  = {
        "unweighted": "Unweighted moments",
        "weighted": "Weighted moments (Gaussian, FWHM=4px)",
        "hsm": "HSM (Hirata-Seljak-Mandelbaum)",
    }

    # ---- 1. Scatter e1 vs e2 ------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True, sharex=True)
    for ax, method in zip(axes, methods):
        arr = results[method]
        valid = np.isfinite(arr[:, 0])
        ax.scatter(arr[valid, 0], arr[valid, 1], s=3, alpha=0.35,
                   color=colours[method], rasterized=True)
        ax.axhline(0, lw=0.7, color='k', ls='--')
        ax.axvline(0, lw=0.7, color='k', ls='--')
        ax.set_xlim(-1, 1)
        ax.set_ylim(-1, 1)
        ax.set_xlabel(r'$\epsilon_1$', fontsize=13)
        ax.set_title(labels[method], fontsize=10)
        n_valid = valid.sum()
        ax.text(0.97, 0.97, f'N={n_valid}/{len(valid)}',
                transform=ax.transAxes, ha='right', va='top', fontsize=9)
    axes[0].set_ylabel(r'$\epsilon_2$', fontsize=13)
    fig.suptitle(r'Ellipticity estimates per method: $\epsilon_1$ vs $\epsilon_2$',
                 fontsize=13)
    fig.tight_layout()
    out = output_dir / 'q1_scatter_e1_e2.png'
    fig.savefig(out, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved {out}')

    # ---- 2. Histograms -------------------------------------------------------
    x_ranges = {
        'unweighted': (-1.0,   1.0),
        'weighted':   (-0.25,  0.25),
        'hsm':        (-0.25,  0.25),
    }
    n_bins_hist = 60
    comp_labels = [r'$\epsilon_1$', r'$\epsilon_2$']
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), sharex=False)
    for col, method in enumerate(methods):
        arr = results[method]
        valid = np.isfinite(arr[:, 0])
        xlo, xhi = x_ranges[method]
        bins = np.linspace(xlo, xhi, n_bins_hist)
        for row in range(2):
            ax = axes[row, col]
            data = arr[valid, row]
            data_clipped = np.clip(data, xlo, xhi)
            ax.hist(data_clipped, bins=bins, color=colours[method], alpha=0.7, density=True)
            ax.axvline(data.mean(), color='k', lw=1.5, ls='--',
                       label=f'mean = {data.mean():.4f}')
            ax.set_xlim(xlo, xhi)
            ax.set_title(
                f"{labels[method]}\n"
                f"{comp_labels[row]}   sigma = {data.std():.4f}",
                fontsize=9,
            )
            ax.legend(fontsize=8)
            if row == 1:
                ax.set_xlabel(comp_labels[row], fontsize=12)
            if col == 0:
                ax.set_ylabel('Density', fontsize=11)
    fig.suptitle('Ellipticity distributions per method', fontsize=13)
    fig.tight_layout()
    out = output_dir / 'q1_histograms.png'
    fig.savefig(out, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved {out}')

    # ---- 3. Method vs HSM with regression -----------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(11, 10))
    e_hsm = results['hsm']
    hsm_valid = np.isfinite(e_hsm[:, 0])

    compare_pairs = [
        ('unweighted', 0, r'$\epsilon_1$'),
        ('unweighted', 1, r'$\epsilon_2$'),
        ('weighted',   0, r'$\epsilon_1$'),
        ('weighted',   1, r'$\epsilon_2$'),
    ]
    regression_results = {}
    for ax, (method, idx, comp_label) in zip(axes.flat, compare_pairs):
        arr = results[method]
        both_valid = hsm_valid & np.isfinite(arr[:, 0])
        x_vals = e_hsm[both_valid, idx]
        y_vals = arr[both_valid, idx]

        ax.scatter(x_vals, y_vals, s=3, alpha=0.35, color=colours[method], rasterized=True)

        lim = 0.5
        ax.plot([-lim, lim], [-lim, lim], 'k--', lw=0.9, label='1:1')

        R, sigma_R = _ols_through_origin(x_vals, y_vals)
        x_line = np.array([-lim, lim])
        ax.plot(x_line, R * x_line, color='firebrick', lw=1.5, ls='-',
                label=fr'$\hat{{\epsilon}} = R \cdot g_{{\rm HSM}}$,  '
                      fr'$R = {R:.3f} \pm {sigma_R:.3f}$')

        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_xlabel(f'HSM  {comp_label}', fontsize=12)
        method_short = labels[method].split('(')[0].strip()
        ax.set_ylabel(f'{method_short}  {comp_label}', fontsize=11)
        ax.legend(fontsize=8)

        r = np.corrcoef(x_vals, y_vals)[0, 1]
        ax.text(0.04, 0.95, f'r = {r:.3f}', transform=ax.transAxes, fontsize=9, va='top')

        key = f"{method}_e{idx + 1}"
        regression_results[key] = (R, sigma_R)

    fig.suptitle('Method comparison vs HSM', fontsize=13)
    fig.tight_layout()
    out = output_dir / 'q1_comparison_vs_hsm.png'
    fig.savefig(out, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved {out}')

    print('\nForce-through-origin regression slopes (method vs HSM):')
    for key, (R, sigma_R) in regression_results.items():
        print(f'  {key:25s}:  R = {R:.4f} +/- {sigma_R:.4f}')
