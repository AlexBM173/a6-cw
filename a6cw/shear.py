import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from pathlib import Path
from scipy.stats import chi2 as scipy_chi2

from nfw_theory import NFWHalo_theory


N_BINS = 10
BIN_EDGES = np.logspace(np.log10(100.0), np.log10(3600.0), N_BINS + 1)
BIN_CENTRES = np.sqrt(BIN_EDGES[:-1] * BIN_EDGES[1:])

COLOURS = {"unweighted": "steelblue", "weighted": "tomato", "hsm": "seagreen"}
LABELS  = {
    "unweighted": "Unweighted moments",
    "weighted": "Weighted moments (FWHM=4px)",
    "hsm": "HSM",
}


def load_positions(path: Path) -> np.ndarray:
    """
    Load (x, y) positions in arcseconds from a two-column text file.
    Parameters -----------
    path : Path
        Path to the text file containing positions.
    Returns ---------
    pos : np.ndarray
        Array of shape (N, 2) with columns [x, y] in arcseconds.
    """
    return np.loadtxt(path)


def compute_tangential_ellipticities(
    halo_pos: np.ndarray,
    source_pos: np.ndarray,
    epsilon_1: np.ndarray,
    epsilon_2: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute per-pair tangential and cross ellipticity components.

    For each (halo h, source s) pair:
        dx        = x_s - x_h
        dy        = y_s - y_h
        r         = sqrt(dx^2 + dy^2)          [arcsec]
        phi       = arctan2(dy, dx)
        epsilon_t = -(epsilon_1 * cos(2*phi) + epsilon_2 * sin(2*phi))
        epsilon_x = -(-epsilon_1 * sin(2*phi) + epsilon_2 * cos(2*phi))

    cos(2φ) and sin(2φ) are evaluated via the trig-free double-angle
    identities, avoiding arctan2 / cos / sin on the full (N_h × N_s) array:

        cos(2φ) = (dx² − dy²) / r²
        sin(2φ) = 2·dx·dy / r²

    epsilon_x is the cross (B-mode) component, rotated 45 degrees from
    the tangential direction.  Gravitational lensing produces only the
    tangential (E-mode) component, so <epsilon_x> = 0 is a null test:
    a non-zero cross-shear profile indicates systematic errors in the
    ellipticity measurements or position angles.

    Parameters
    ----------
    halo_pos             : (N_h, 2) halo (x, y) positions [arcsec]
    source_pos           : (N_s, 2) source (x, y) positions [arcsec]
    epsilon_1, epsilon_2 : (N_s,) source ellipticity components

    Returns
    -------
    r_pairs         : (N_h * N_s,) separations [arcsec]
    epsilon_t_pairs : (N_h * N_s,) tangential ellipticities
    epsilon_x_pairs : (N_h * N_s,) cross ellipticities  (null-test quantity)
    """
    dx = source_pos[:, 0][np.newaxis, :] - halo_pos[:, 0][:, np.newaxis]
    dy = source_pos[:, 1][np.newaxis, :] - halo_pos[:, 1][:, np.newaxis]
    r2 = dx**2 + dy**2
    r  = np.sqrt(r2)

    # Trig-free double-angle formulae: eliminates arctan2, cos, sin calls.
    # Guard against r²=0 (source coincident with halo) with np.where.
    safe    = r2 > 0
    inv_r2  = np.where(safe, 1.0 / np.where(safe, r2, 1.0), 0.0)
    cos2phi = (dx**2 - dy**2) * inv_r2
    sin2phi = 2.0 * dx * dy   * inv_r2

    e1 = epsilon_1[np.newaxis, :]
    e2 = epsilon_2[np.newaxis, :]

    epsilon_t = -(e1 * cos2phi + e2 * sin2phi)
    epsilon_x = -(-e1 * sin2phi + e2 * cos2phi)

    return r.ravel(), epsilon_t.ravel(), epsilon_x.ravel()


def mean_tangential_shear(
    r_pairs: np.ndarray,
    epsilon_t_pairs: np.ndarray,
    bin_edges: np.ndarray,
    weights: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Bin tangential ellipticities and return mean tangential shear.

    The mean tangential shear is <epsilon_t> because in the weak-lensing
    limit <epsilon_t> = gamma_t.

    Uncertainty is the standard error on the mean, dominated by shape noise:

        sigma(gt) = sqrt(Var(epsilon_t) / N)       [unweighted]
        sigma(gt) = sqrt(1 / sum(w_i))             [weighted, w_i = 1/sigma_i^2]

    For the unweighted case, bin assignment is done with np.digitize (one
    O(N log B) pass), and per-bin sums are accumulated with np.bincount,
    replacing the original O(N·B) loop over repeated boolean masks.

    Parameters
    ----------
    r_pairs         : separations for all pairs [arcsec]
    epsilon_t_pairs : tangential ellipticities for all pairs
    bin_edges       : radial bin edges [arcsec]
    weights         : optional per-pair inverse-variance weights;
                      if None the simple unweighted mean is used.

    Returns
    -------
    gt      : mean tangential shear per bin  (N_bins,)
    gt_err  : 1-sigma uncertainty per bin    (N_bins,)
    n_pairs : number of valid (finite) pairs per bin  (N_bins,)
    """
    n_bins = len(bin_edges) - 1
    gt      = np.full(n_bins, np.nan)
    gt_err  = np.full(n_bins, np.nan)
    n_pairs = np.zeros(n_bins, dtype=int)

    # Single-pass bin assignment: O(N log B) vs the original O(N·B)
    bin_idx = np.digitize(r_pairs, bin_edges) - 1   # 0-indexed; <0 or >=n_bins → outside
    finite  = np.isfinite(epsilon_t_pairs)
    valid   = finite & (bin_idx >= 0) & (bin_idx < n_bins)

    # Count finite pairs per bin in one vectorised call
    n_pairs[:] = np.bincount(bin_idx[valid], minlength=n_bins)
    has2 = n_pairs >= 2

    if weights is None:
        # Fully vectorised path: sums and sum-of-squares via np.bincount
        if has2.any():
            bv   = bin_idx[valid]
            et_v = epsilon_t_pairs[valid]
            s1   = np.bincount(bv, weights=et_v,    minlength=n_bins)
            s2   = np.bincount(bv, weights=et_v**2, minlength=n_bins)
            n_f  = n_pairs.astype(float)
            mean = np.where(has2, s1 / n_f, np.nan)
            # Bessel-corrected variance: (Σx² − (Σx)²/n) / (n − 1)
            var  = np.where(has2, (s2 - s1**2 / n_f) / (n_f - 1), np.nan)
            gt[:]     = mean
            gt_err[:] = np.where(has2, np.sqrt(np.maximum(var, 0.0) / n_f), np.nan)
    else:
        # Weighted path: per-bin loop (less common; bincount with weights
        # cannot handle the 1/sqrt(Σw) error formula directly)
        for k in range(n_bins):
            if not has2[k]:
                continue
            mk    = valid & (bin_idx == k)
            et_k  = epsilon_t_pairs[mk]
            w_k   = weights[mk]
            w_sum = w_k.sum()
            gt[k]     = (w_k * et_k).sum() / w_sum
            gt_err[k] = np.sqrt(1.0 / w_sum)

    return gt, gt_err, n_pairs


def estimate_shape_noise_variance(epsilon_t_pairs: np.ndarray) -> float:
    """
    Estimate the per-galaxy shape-noise variance from all valid tangential
    ellipticity measurements.

    The shape noise dominates the per-pair scatter (the shear signal is tiny
    compared to intrinsic ellipticities), so Var(e_t) ~ sigma_epsilon^2.

    Returns sigma_epsilon^2 per component.
    """
    finite = np.isfinite(epsilon_t_pairs)
    return float(np.var(epsilon_t_pairs[finite], ddof=1))


def optimal_weights(epsilon_t_pairs: np.ndarray) -> np.ndarray:
    """
    Return per-pair inverse-variance weights  w_i = 1 / sigma_epsilon^2.

    Since the shape noise is the same for all galaxies in this simulation,
    the weights are uniform and the optimal estimator reduces to the simple
    mean.  In real surveys galaxies have different sizes and S/N, so
    sigma_epsilon^2 would vary per source and the weights would differ.

    NaN pairs receive weight 0.
    """
    sigma2 = estimate_shape_noise_variance(epsilon_t_pairs)
    w = np.where(np.isfinite(epsilon_t_pairs), 1.0 / sigma2, 0.0)
    return w


def plot_tangential_shear_profiles(
    profiles: dict,
    output_dir: Path,
    filename: str = "q2c_tangential_shear_profiles.png",
):
    """
    Plot gamma_t(r) with 1-sigma error bars for all three Q1 methods
    on a log-log scale. Uses the simple (unweighted) mean, which is
    identical to the optimal inverse-variance weighted mean for this
    simulation since all galaxies share the same shape noise variance.

    Parameters
    ----------
    profiles : dict keyed by method name, each value a dict with keys
               'gt_simple', 'gt_err_simple'
    """
    fig, ax = plt.subplots(figsize=(8, 6))

    for method in ["unweighted", "weighted", "hsm"]:
        gt     = profiles[method]['gt_simple']
        gt_err = profiles[method]['gt_err_simple']
        valid  = np.isfinite(gt)
        ax.errorbar(
            BIN_CENTRES[valid], gt[valid], yerr=gt_err[valid],
            fmt='o', ms=5, capsize=3,
            color=COLOURS[method], label=LABELS[method],
        )

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel(r'Separation $\theta$ [arcsec]', fontsize=13)
    ax.set_ylabel(r'Mean tangential shear $\langle\gamma_t\rangle$', fontsize=13)
    ax.set_title(
        'Mean tangential shear profiles around NFW halos\n'
        'Three shape estimators compared',
        fontsize=13,
    )
    ax.legend(fontsize=9)
    ax.grid(True, which='both', ls=':', alpha=0.4)
    fig.tight_layout()
    out = output_dir / filename
    fig.savefig(out, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved {out}')


def plot_cross_shear_null_test(
    profiles: dict,
    output_dir: Path,
    filename: str = 'q2c_cross_shear_null_test.png',
):
    """
    Plot the mean cross-shear (B-mode) profile <gamma_x>(theta) for all
    three shape estimators as a null test.

    Gravitational lensing is a pure E-mode signal: it produces tangential
    shear around mass concentrations but zero cross-shear by symmetry.
    A non-zero <gamma_x> profile would indicate systematic errors such
    as a miscalibrated position angle, PSF anisotropy residuals, or
    intrinsic alignment contamination.

    The profile is plotted on a linear y-scale (not log, since it should
    be consistent with zero) with a shaded zero line for reference.
    Error bars are computed in the same way as for gamma_t.
    """
    fig, ax = plt.subplots(figsize=(8, 5))

    for method in ['unweighted', 'weighted', 'hsm']:
        gx     = profiles[method]['gx_simple']
        gx_err = profiles[method]['gx_err_simple']
        valid  = np.isfinite(gx)
        ax.errorbar(
            BIN_CENTRES[valid], gx[valid], yerr=gx_err[valid],
            fmt='o', ms=5, capsize=3,
            color=COLOURS[method], label=LABELS[method],
        )

    ax.axhline(0, color='k', lw=1.0, ls='--', zorder=0)
    ax.set_xscale('log')
    ax.set_xlabel(r'Separation $\theta$ [arcsec]', fontsize=13)
    ax.set_ylabel(r'Mean cross-shear $\langle\gamma_\times\rangle$', fontsize=13)
    ax.set_title(
        'Cross-shear null test\n'
        r'($\langle\gamma_\times\rangle = 0$ expected for pure gravitational lensing)',
        fontsize=12,
    )
    ax.legend(fontsize=9)
    ax.grid(True, which='both', ls=':', alpha=0.4)
    fig.tight_layout()
    out = output_dir / filename
    fig.savefig(out, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved {out}')


def build_nfw_theory(
    mass: float,
    source_redshift: float,
    concentration: float = 4.0,
    halo_redshift: float = 0.3,
    omega_m: float = 0.3,
    omega_lam: float = 0.7,
) -> np.ndarray:
    """
    Compute the theoretical mean tangential shear at BIN_CENTRES for a
    single NFW halo with the given concentration / cosmology.

    Parameters
    ----------
    mass            : halo mass [M_sun / h]
    source_redshift : source galaxy redshift
    concentration   : NFW concentration parameter (default 4.0)
    halo_redshift   : halo redshift (default 0.3)
    omega_m         : matter density parameter (default 0.3)
    omega_lam       : dark energy density parameter (default 0.7)

    Returns
    -------
    gt_theory : (N_bins,) array of predicted mean tangential shear
    """
    nfw = NFWHalo_theory(
        mass=mass,
        conc=concentration,
        redshift=halo_redshift,
        omega_m=omega_m,
        omega_lam=omega_lam,
    )
    return nfw.get_tangential_shear(source_redshift, BIN_CENTRES)


def log_likelihood(
    gt_data: np.ndarray,
    gt_err: np.ndarray,
    mass: float,
    source_redshift: float,
) -> float:
    """
    Gaussian log-likelihood:
        ln L = -0.5 * sum_k [ (gt_data_k - gt_theory_k)^2 / gt_err_k^2 ]

    Bins with NaN data or theory are excluded.

    Parameters
    ----------
    gt_data         : measured mean tangential shear per bin
    gt_err          : 1-sigma uncertainty per bin
    mass            : halo mass
    source_redshift : source redshift

    Returns
    -------
    lnL : float
    """
    gt_theory = build_nfw_theory(mass, source_redshift)
    valid = np.isfinite(gt_data) & np.isfinite(gt_err) & np.isfinite(gt_theory)
    if valid.sum() == 0:
        return -np.inf
    residuals = gt_data[valid] - gt_theory[valid]
    return -0.5 * np.sum((residuals / gt_err[valid])**2)


# ---------------------------------------------------------------------------
# Parallel posterior grid
# ---------------------------------------------------------------------------

def _posterior_cell(args):
    """
    Top-level worker for ProcessPoolExecutor in compute_posterior_grid.

    Must be a module-level function so it can be pickled by multiprocessing.
    """
    gt_data, gt_err, mass, zs = args
    return log_likelihood(gt_data, gt_err, mass, zs)


def compute_posterior_grid(
    gt_data: np.ndarray,
    gt_err: np.ndarray,
    mass_grid: np.ndarray,
    zs_grid: np.ndarray,
    n_workers: int | None = None,
) -> np.ndarray:
    """
    Evaluate the (unnormalised) posterior on a 2D grid of (mass, z_source)
    using a flat (uniform) prior on both parameters.

    Each grid cell is independent, so the evaluation is parallelised across
    all available CPU cores using ProcessPoolExecutor. The Python GIL prevents
    thread-based parallelism here (build_nfw_theory is pure Python), so
    separate processes are used instead.

    Parameters
    ----------
    gt_data    : (N_bins,) measured tangential shear
    gt_err     : (N_bins,) 1-sigma uncertainties
    mass_grid  : (N_m,) halo mass values to evaluate
    zs_grid    : (N_z,) source redshift values to evaluate
    n_workers  : number of worker processes (default: os.cpu_count())

    Returns
    -------
    log_post : (N_m, N_z) array of log-posterior values
    """
    n_m, n_z = len(mass_grid), len(zs_grid)
    n_cells  = n_m * n_z

    if n_workers is None:
        n_workers = os.cpu_count() or 4

    cells = [
        (gt_data, gt_err, float(mass), float(zs))
        for mass in mass_grid
        for zs   in zs_grid
    ]

    if n_workers == 1 or n_cells < 50:
        flat = [_posterior_cell(c) for c in cells]
    else:
        try:
            chunksize = max(1, n_cells // (n_workers * 4))
            with ProcessPoolExecutor(max_workers=n_workers) as exc:
                flat = list(exc.map(_posterior_cell, cells, chunksize=chunksize))
        except Exception:
            # Fall back to serial (e.g., in restricted execution environments)
            flat = [_posterior_cell(c) for c in cells]

    return np.array(flat, dtype=float).reshape(n_m, n_z)


def plot_joint_posterior(
    log_post: np.ndarray,
    mass_grid: np.ndarray,
    zs_grid: np.ndarray,
    output_dir: Path,
    filename: str = 'q2d_joint_posterior.png',
):
    """
    Plot the 2D joint posterior of (halo mass, source redshift) as filled
    contours at the 68% and 95% credible levels, plus the MAP estimate,
    with marginalised 1D posteriors and 68% credible intervals on each axis.

    The 2D credible contours use the correct chi^2(df=2) quantiles:
        Delta chi^2 = 2.30  (68%,  chi2.ppf(0.6827, df=2))
        Delta chi^2 = 6.18  (95%,  chi2.ppf(0.9545, df=2))
    """
    from matplotlib.patches import Patch

    log_post_shifted = log_post - log_post.max()
    posterior = np.exp(log_post_shifted)

    idx_map = np.unravel_index(np.argmax(log_post), log_post.shape)
    mass_map = mass_grid[idx_map[0]]
    zs_map   = zs_grid[idx_map[1]]
    print(f'MAP: mass = {mass_map:.3e} M_sun,  z_s = {zs_map:.3f}')

    p_mass = np.trapezoid(posterior, zs_grid, axis=1)
    p_mass /= np.trapezoid(p_mass, mass_grid)

    p_zs = np.trapezoid(posterior, mass_grid, axis=0)
    p_zs /= np.trapezoid(p_zs, zs_grid)

    def credible_interval_1d(x, p):
        """Return (lo, peak, hi) for the 68% credible interval."""
        cdf = np.array([np.trapezoid(p[:k+1], x[:k+1]) for k in range(len(x))])
        cdf /= cdf[-1]
        lo   = x[np.searchsorted(cdf, 0.16)]
        hi   = x[np.searchsorted(cdf, 0.84)]
        peak = x[np.argmax(p)]
        return lo, peak, hi

    m_lo,  m_map_marg,  m_hi  = credible_interval_1d(mass_grid, p_mass)
    zs_lo, zs_map_marg, zs_hi = credible_interval_1d(zs_grid,   p_zs)

    print(f'Marginal mass:  MAP = {m_map_marg:.3e}  '
          f'68% CI = [{m_lo:.3e}, {m_hi:.3e}] M_sun')
    print(f'  -> {m_map_marg/1e14:.3f} +{(m_hi-m_map_marg)/1e14:.3f} '
          f'-{(m_map_marg-m_lo)/1e14:.3f}  x10^14 M_sun')
    print(f'Marginal z_s:   MAP = {zs_map_marg:.3f}  '
          f'68% CI = [{zs_lo:.3f}, {zs_hi:.3f}]')
    print(f'  -> {zs_map_marg:.3f} +{zs_hi-zs_map_marg:.3f} '
          f'-{zs_map_marg-zs_lo:.3f}')

    delta_chi2_68 = scipy_chi2.ppf(0.6827, df=2)
    delta_chi2_95 = scipy_chi2.ppf(0.9545, df=2)
    level_68 = np.exp(-delta_chi2_68 / 2.0)
    level_95 = np.exp(-delta_chi2_95 / 2.0)

    fig = plt.figure(figsize=(10, 8))
    gs  = fig.add_gridspec(2, 2, width_ratios=[3, 1], height_ratios=[1, 3],
                           hspace=0.05, wspace=0.05)
    ax_joint  = fig.add_subplot(gs[1, 0])
    ax_marg_z = fig.add_subplot(gs[0, 0], sharex=ax_joint)
    ax_marg_m = fig.add_subplot(gs[1, 1], sharey=ax_joint)
    ax_legend = fig.add_subplot(gs[0, 1])
    ax_legend.axis('off')

    ax_joint.contourf(
        zs_grid, mass_grid / 1e14, posterior,
        levels=[level_95, level_68, 1.0],
        colors=['#aec6e8', '#4a90d9', '#1a3f6f'],
        alpha=0.85,
    )
    ax_joint.contour(
        zs_grid, mass_grid / 1e14, posterior,
        levels=[level_95, level_68],
        colors=['#2166ac', '#053061'],
        linewidths=1.2,
    )
    ax_joint.axvline(zs_map,          color='k',         ls='--', lw=1.0,
                     label=f'MAP $z_s$ = {zs_map:.2f}')
    ax_joint.axhline(mass_map / 1e14, color='k',         ls=':',  lw=1.0,
                     label=f'MAP $M$ = {mass_map:.2e} $M_\\odot$')
    ax_joint.axvline(0.6,             color='firebrick', ls='-.', lw=1.2,
                     label='True $z_s$ = 0.6')
    ax_joint.set_xlabel(r'Source redshift $z_s$',                 fontsize=13)
    ax_joint.set_ylabel(r'Halo mass $M$ [$10^{14}\,M_\odot$]',  fontsize=13)
    ax_joint.set_xlim(zs_grid.min(),          zs_grid.max())
    ax_joint.set_ylim(mass_grid.min() / 1e14, mass_grid.max() / 1e14)

    ax_marg_z.plot(zs_grid, p_zs, color='#2166ac', lw=1.8)
    ax_marg_z.fill_between(zs_grid, p_zs,
                           where=(zs_grid >= zs_lo) & (zs_grid <= zs_hi),
                           color='#4a90d9', alpha=0.4, label='68% CI')
    ax_marg_z.axvline(zs_map_marg, color='k',         ls='--', lw=1.0)
    ax_marg_z.axvline(0.6,         color='firebrick', ls='-.', lw=1.2)
    ax_marg_z.set_ylabel(r'$p(z_s)$', fontsize=11)
    ax_marg_z.set_title(
        f'$z_s = {zs_map_marg:.3f}^{{+{zs_hi-zs_map_marg:.3f}}}_{{-{zs_map_marg-zs_lo:.3f}}}$',
        fontsize=10)
    plt.setp(ax_marg_z.get_xticklabels(), visible=False)

    ax_marg_m.plot(p_mass * 1e14, mass_grid / 1e14, color='#2166ac', lw=1.8)
    ax_marg_m.fill_betweenx(mass_grid / 1e14, p_mass * 1e14,
                             where=(mass_grid >= m_lo) & (mass_grid <= m_hi),
                             color='#4a90d9', alpha=0.4, label='68% CI')
    ax_marg_m.axhline(m_map_marg / 1e14, color='k', ls='--', lw=1.0)
    ax_marg_m.set_xlabel(r'$p(M)$', fontsize=11)
    m_lo14, m_hi14, m_map14 = m_lo/1e14, m_hi/1e14, m_map_marg/1e14
    ax_marg_m.set_title(
        f'$M = {m_map14:.2f}^{{+{m_hi14-m_map14:.2f}}}_{{-{m_map14-m_lo14:.2f}}}$\n'
        r'$[10^{14}\,M_\odot]$',
        fontsize=10)
    plt.setp(ax_marg_m.get_yticklabels(), visible=False)

    handles = [
        Patch(facecolor='#4a90d9', label='68% credible region'),
        Patch(facecolor='#aec6e8', label='95% credible region'),
    ]
    ax_legend.legend(handles=handles, loc='center', fontsize=9, frameon=False)
    ax_joint.legend(loc='upper right', fontsize=8)

    fig.suptitle('Joint posterior: halo mass and source redshift\n'
                 '(HSM estimator, flat prior)', fontsize=12, y=1.01)
    out = output_dir / filename
    fig.savefig(out, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved {out}')

    return mass_map, zs_map, m_lo, m_hi, zs_lo, zs_hi


def plot_mass_posterior_fixed_zs(
    log_post: np.ndarray,
    mass_grid: np.ndarray,
    zs_grid: np.ndarray,
    fixed_zs: float,
    output_dir: Path,
    filename: str = "q2d_mass_posterior_fixed_zs.png",
):
    """
    Plot the marginal posterior on halo mass with source redshift fixed
    at fixed_zs, by slicing the 2D posterior at the nearest z_s grid point.
    """
    j_fixed    = np.argmin(np.abs(zs_grid - fixed_zs))
    zs_actual  = zs_grid[j_fixed]
    log_p_mass = log_post[:, j_fixed]

    log_p_mass -= log_p_mass.max()
    p_mass      = np.exp(log_p_mass)
    norm   = np.trapezoid(p_mass, mass_grid)
    p_mass /= norm

    cdf      = np.array([np.trapezoid(p_mass[:k+1], mass_grid[:k+1]) for k in range(len(mass_grid))])
    cdf     /= cdf[-1]
    mass_lo  = mass_grid[np.searchsorted(cdf, 0.16)]
    mass_hi  = mass_grid[np.searchsorted(cdf, 0.84)]
    mass_map = mass_grid[np.argmax(p_mass)]

    print(f"  Fixed z_s = {zs_actual:.3f}:")
    print(f"    MAP mass = {mass_map:.3e} M_sun")
    print(f"    68% CI   = [{mass_lo:.3e}, {mass_hi:.3e}] M_sun")

    log_map  = np.log10(mass_map)
    log_lo   = np.log10(max(mass_lo * 0.5, mass_grid[0]))
    log_hi   = np.log10(min(mass_hi * 2.0, mass_grid[-1]))
    x_lo     = 10**log_lo / 1e14
    x_hi     = 10**log_hi / 1e14

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(mass_grid / 1e14, p_mass * 1e14, color="steelblue", lw=2)
    ax.fill_between(
        mass_grid / 1e14, p_mass * 1e14,
        where=(mass_grid >= mass_lo) & (mass_grid <= mass_hi),
        color="steelblue", alpha=0.35, label="68% credible interval",
    )
    ax.axvline(mass_map / 1e14, color="k", ls="--", lw=1.2,
               label=f"MAP = {mass_map / 1e14:.2f} $\\times 10^{{14}}\\,M_\\odot$")
    ax.axvline(mass_lo / 1e14,  color="steelblue", ls=":", lw=1.0,
               label=f"68% CI: [{mass_lo/1e14:.2f}, {mass_hi/1e14:.2f}] $\\times10^{{14}}\\,M_\\odot$")
    ax.axvline(mass_hi / 1e14,  color="steelblue", ls=":", lw=1.0)
    ax.set_xlim(x_lo, x_hi)
    ax.set_xlabel(r"Halo mass $M$ [$10^{14}\,M_\odot$]", fontsize=13)
    ax.set_ylabel(r"Posterior $p(M \mid z_s = %.1f)$" % zs_actual, fontsize=12)
    ax.set_title(f"Mass posterior with source redshift fixed at $z_s = {zs_actual:.1f}$", fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, ls=":", alpha=0.4)
    fig.tight_layout()
    out = output_dir / filename
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out}")

    return mass_map, mass_lo, mass_hi


def plot_best_fit_overlay(
    gt_data: np.ndarray,
    gt_err: np.ndarray,
    mass_map: float,
    zs_map: float,
    mass_map_zs06: float,
    output_dir: Path,
    filename: str = "q2d_best_fit_overlay.png",
):
    """
    Overplot the MAP NFW theory curve on the measured tangential shear profile.
    Shows both the joint MAP (mass_map, zs_map) and the best-fit mass at
    fixed z_s = 0.6.
    """
    gt_theory = build_nfw_theory(mass_map, zs_map)
    gt_theory_zs06 = build_nfw_theory(mass_map_zs06, 0.6)

    valid = np.isfinite(gt_data) & np.isfinite(gt_err)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.errorbar(
        BIN_CENTRES[valid], gt_data[valid], yerr=gt_err[valid],
        fmt="o", ms=6, capsize=3, color="seagreen", zorder=5, label="HSM data",
    )
    ax.plot(BIN_CENTRES, gt_theory, color="k", lw=2,
            label=f"Joint MAP: $M={mass_map/1e14:.2f}\\times10^{{14}}\\,M_\\odot$,"
                  f" $z_s={zs_map:.2f}$")
    ax.plot(BIN_CENTRES, gt_theory_zs06, color="firebrick", lw=1.5, ls="--",
            label=f"Best fit ($z_s=0.6$): $M={mass_map_zs06/1e14:.2f}\\times10^{{14}}\\,M_\\odot$")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"Separation $\theta$ [arcsec]", fontsize=13)
    ax.set_ylabel(r"Mean tangential shear $\langle\gamma_t\rangle$", fontsize=13)
    ax.set_title("MAP NFW theory vs measured tangential shear (HSM)", fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(True, which="both", ls=":", alpha=0.4)
    fig.tight_layout()
    out = output_dir / filename
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")
