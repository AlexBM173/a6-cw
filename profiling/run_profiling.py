#!/usr/bin/env python3
"""
Performance profiling for the a6cw package.

Measures wall-clock time (median over repeated calls) and peak heap memory
(tracemalloc) for the four computationally significant routines:

  1. Ellipticity estimators (unweighted, weighted, HSM) vs stamp size
  2. compute_tangential_ellipticities vs N_sources (N_halos fixed)
  3. build_nfw_theory: baseline cost per posterior-grid cell
  4. compute_posterior_grid vs grid resolution

Run from the repository root with the a6_coursework venv active:

    python profiling/run_profiling.py

Outputs are written to outputs/profiling/.
"""

import sys
import time
import tracemalloc
import numpy as np
import galsim
import matplotlib.pyplot as plt
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from a6cw.ellipticity import (
    unweighted_ellipticity,
    weighted_ellipticity_raw,
    hsm_ellipticity,
)
from a6cw.shear import (
    compute_tangential_ellipticities,
    build_nfw_theory,
    compute_posterior_grid,
    N_BINS,
)

OUTPUT_DIR = Path("outputs/profiling")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

COLOURS = {"unweighted": "steelblue", "weighted": "tomato", "hsm": "seagreen"}
LABELS  = {
    "unweighted": "Unweighted moments",
    "weighted":   "Gaussian-weighted moments",
    "hsm":        "HSM (GalSim)",
}


# ---------------------------------------------------------------------------
# Generic helpers
# ---------------------------------------------------------------------------

def time_fn(fn, args: tuple, n_repeats: int) -> tuple[float, float]:
    """Median and std wall-clock time in ms over n_repeats calls."""
    fn(*args)                              # warm-up
    ts = []
    for _ in range(n_repeats):
        t0 = time.perf_counter()
        fn(*args)
        ts.append((time.perf_counter() - t0) * 1e3)
    arr = np.array(ts)
    return float(np.median(arr)), float(arr.std())


def peak_memory_kb(fn, args: tuple) -> float:
    """Peak heap allocated in KB for a single call (tracemalloc)."""
    tracemalloc.start()
    tracemalloc.clear_traces()
    fn(*args)
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return peak / 1024.0


def make_stamp(size_px: int, g1: float = 0.1) -> galsim.Image:
    """Noiseless Exponential galaxy stamp filling ~1/8 of the stamp."""
    hlr = max(size_px / 8.0, 0.6)
    gal = galsim.Exponential(half_light_radius=hlr).shear(g1=g1, g2=0.0)
    return gal.drawImage(nx=size_px, ny=size_px, scale=1.0, method="no_pixel")


# ---------------------------------------------------------------------------
# Section 1 — Ellipticity estimators: time and memory vs stamp size
# ---------------------------------------------------------------------------

STAMP_SIZES   = [16, 24, 32, 48, 64, 96, 128, 192]
N_REP_STAMP   = 100
TYPICAL_SIZE  = 48

estimators = {
    "unweighted": unweighted_ellipticity,
    "weighted":   weighted_ellipticity_raw,
    "hsm":        hsm_ellipticity,
}

print("=" * 65)
print("1. Ellipticity estimators — time and memory vs stamp size")
print("=" * 65)

stamps = {sz: make_stamp(sz) for sz in STAMP_SIZES}   # pre-render all stamps

e_time_med = {k: [] for k in estimators}
e_time_std = {k: [] for k in estimators}
e_mem_kb   = {k: [] for k in estimators}

for sz in STAMP_SIZES:
    print(f"  {sz:3d}×{sz:3d}  ({sz**2:6d} px²)", end="", flush=True)
    stamp = stamps[sz]
    for name, fn in estimators.items():
        med, std = time_fn(fn, (stamp,), n_repeats=N_REP_STAMP)
        mem      = peak_memory_kb(fn, (stamp,))
        e_time_med[name].append(med)
        e_time_std[name].append(std)
        e_mem_kb[name].append(mem)
        print(f"   {name}: {med:.3f}ms {mem:.0f}KB", end="", flush=True)
    print()

sizes_arr  = np.array(STAMP_SIZES, float)
e_time_med = {k: np.array(v) for k, v in e_time_med.items()}
e_time_std = {k: np.array(v) for k, v in e_time_std.items()}
e_mem_kb   = {k: np.array(v) for k, v in e_mem_kb.items()}

# Figure 1a — two-panel: time and memory vs N² (stamp area)
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# Panel A: timing
ax = axes[0]
for name in estimators:
    ax.errorbar(
        sizes_arr**2, e_time_med[name], yerr=e_time_std[name],
        marker="o", ms=5, capsize=3, lw=1.8,
        color=COLOURS[name], label=LABELS[name],
    )
# N² reference anchored to unweighted at largest size
ref_t = e_time_med["unweighted"][-1] / sizes_arr[-1] ** 2
ax.plot(sizes_arr**2, ref_t * sizes_arr**2, "k--", lw=0.9, label=r"$\propto N^2$ ref")
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlabel(r"Stamp area $N^2$ (pixels)", fontsize=12)
ax.set_ylabel("Time per stamp (ms)", fontsize=12)
ax.set_title("Wall-clock time vs stamp size", fontsize=11)
ax.legend(fontsize=9); ax.grid(True, which="both", ls=":", alpha=0.4)

# Panel B: peak memory
ax = axes[1]
for name in estimators:
    ax.plot(
        sizes_arr**2, e_mem_kb[name],
        marker="o", ms=5, lw=1.8,
        color=COLOURS[name], label=LABELS[name],
    )
# Theoretical N² reference: 8 intermediate float64 arrays
ref_m = 8 * 8 * sizes_arr**2 / 1024
ax.plot(sizes_arr**2, ref_m, "k--", lw=0.9, label=r"8×$N^2$ float64 ref")
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlabel(r"Stamp area $N^2$ (pixels)", fontsize=12)
ax.set_ylabel("Peak memory (KB)", fontsize=12)
ax.set_title("Peak heap memory vs stamp size", fontsize=11)
ax.legend(fontsize=9); ax.grid(True, which="both", ls=":", alpha=0.4)

fig.suptitle("Ellipticity estimators: scaling with stamp size", fontsize=13)
fig.tight_layout()
fig.savefig(OUTPUT_DIR / "perf_ellipticity_scaling.png", dpi=200, bbox_inches="tight")
plt.close(fig)
print(f"\n  Saved {OUTPUT_DIR}/perf_ellipticity_scaling.png")

# Figure 1b — throughput bar chart at the typical 48×48 size
idx_typ = STAMP_SIZES.index(TYPICAL_SIZE)
throughput = {k: 1000.0 / e_time_med[k][idx_typ] for k in estimators}
mem_typ    = {k: e_mem_kb[k][idx_typ]              for k in estimators}

fig, ax = plt.subplots(figsize=(7, 4))
names = list(estimators)
x     = np.arange(len(names))
bars  = ax.bar(x, [throughput[n] for n in names],
               color=[COLOURS[n] for n in names], edgecolor="k", linewidth=0.7)
ax.bar_label(bars, fmt="{:.0f}", padding=3, fontsize=10)
ax.set_xticks(x); ax.set_xticklabels([LABELS[n] for n in names], fontsize=9)
ax.set_ylabel("Galaxies / second", fontsize=12)
ax.set_title(f"Throughput at {TYPICAL_SIZE}×{TYPICAL_SIZE} px stamps", fontsize=12)
ax.grid(axis="y", ls=":", alpha=0.4)
fig.tight_layout()
fig.savefig(OUTPUT_DIR / "perf_throughput.png", dpi=200, bbox_inches="tight")
plt.close(fig)
print(f"  Saved {OUTPUT_DIR}/perf_throughput.png")

# Print throughput table
print(f"\n  Throughput at {TYPICAL_SIZE}×{TYPICAL_SIZE}:")
print(f"  {'Estimator':20s}  {'ms/gal':>8s}  {'gal/s':>10s}  {'peak KB':>9s}")
for name in names:
    ms  = e_time_med[name][idx_typ]
    gps = throughput[name]
    kb  = mem_typ[name]
    print(f"  {LABELS[name]:20s}  {ms:8.3f}  {gps:10.0f}  {kb:9.0f}")


# ---------------------------------------------------------------------------
# Section 2 — compute_tangential_ellipticities: time and memory vs N_sources
# ---------------------------------------------------------------------------

N_HALOS_FIXED    = 10
N_SOURCES_SERIES = [100, 500, 1000, 5000, 10_000, 50_000, 100_000]

print("\n" + "=" * 65)
print("2. compute_tangential_ellipticities — scaling vs N_sources")
print(f"   (N_halos = {N_HALOS_FIXED} fixed)")
print("=" * 65)

rng = np.random.default_rng(42)
te_times  = []
te_mem_kb = []

for n_s in N_SOURCES_SERIES:
    halo_pos   = rng.uniform(0, 1e4, (N_HALOS_FIXED, 2))
    source_pos = rng.uniform(0, 1e4, (n_s, 2))
    e1 = rng.normal(0, 0.3, n_s)
    e2 = rng.normal(0, 0.3, n_s)
    args = (halo_pos, source_pos, e1, e2)

    n_rep = max(3, min(30, 50_000 // n_s))
    med, _ = time_fn(compute_tangential_ellipticities, args, n_repeats=n_rep)
    mem    = peak_memory_kb(compute_tangential_ellipticities, args)
    te_times.append(med)
    te_mem_kb.append(mem)
    print(f"  N_s={n_s:7d}  time={med:8.2f}ms  mem={mem/1024:6.2f}MB")

te_times  = np.array(te_times)
te_mem_kb = np.array(te_mem_kb)
ns_arr    = np.array(N_SOURCES_SERIES, float)

fig, ax1 = plt.subplots(figsize=(8, 5))
c_time = "#2166ac"; c_mem = "#d73027"

l1, = ax1.loglog(ns_arr, te_times, "o-", color=c_time, lw=2, ms=6, label="Wall time (ms)")
ref_t = te_times[0] / ns_arr[0]
ax1.loglog(ns_arr, ref_t * ns_arr, "k--", lw=0.9, label=r"$\propto N_s$ ref")
ax1.set_xlabel(f"N_sources  (N_halos = {N_HALOS_FIXED})", fontsize=12)
ax1.set_ylabel("Wall time (ms)", fontsize=12, color=c_time)
ax1.tick_params(axis="y", labelcolor=c_time)

ax2 = ax1.twinx()
l2, = ax2.loglog(ns_arr, te_mem_kb / 1024, "s--", color=c_mem, lw=1.5, ms=6,
                  label="Peak memory (MB)")
ref_m = N_HALOS_FIXED * ns_arr * 8 * 8 / 1024**2
ax2.loglog(ns_arr, ref_m, color=c_mem, ls=":", lw=0.9, alpha=0.7,
           label=fr"8 × $N_h N_s$ float64 ref")
ax2.set_ylabel("Peak memory (MB)", fontsize=12, color=c_mem)
ax2.tick_params(axis="y", labelcolor=c_mem)

lines1, labs1 = ax1.get_legend_handles_labels()
lines2, labs2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labs1 + labs2, fontsize=9, loc="upper left")
ax1.set_title(
    "compute_tangential_ellipticities: time and memory vs N_sources\n"
    f"(N_halos = {N_HALOS_FIXED}; vectorised over halos × sources)",
    fontsize=11,
)
ax1.grid(True, which="both", ls=":", alpha=0.4)
fig.tight_layout()
fig.savefig(OUTPUT_DIR / "perf_tangential_scaling.png", dpi=200, bbox_inches="tight")
plt.close(fig)
print(f"\n  Saved {OUTPUT_DIR}/perf_tangential_scaling.png")


# ---------------------------------------------------------------------------
# Section 3 — build_nfw_theory: per-call baseline
# ---------------------------------------------------------------------------

print("\n" + "=" * 65)
print("3. build_nfw_theory — per-call timing (N_bins fixed = 10)")
print("=" * 65)

nfw_med, nfw_std = time_fn(build_nfw_theory, (5e14, 0.6), n_repeats=50)
print(f"  Median: {nfw_med:.3f} ms  (std {nfw_std:.3f} ms)")


# ---------------------------------------------------------------------------
# Section 4 — compute_posterior_grid: time vs grid resolution
# ---------------------------------------------------------------------------

GRID_SIZES = [5, 10, 15, 20, 30, 40, 50]

print("\n" + "=" * 65)
print("4. compute_posterior_grid — time vs grid resolution")
print(f"   (each cell ≈ {nfw_med:.2f} ms for build_nfw_theory + lnL eval)")
print("=" * 65)

gt_data = np.full(N_BINS, 0.02)
gt_err  = np.full(N_BINS, 0.005)

pg_times    = []
pg_n_cells  = []

for g in GRID_SIZES:
    mass_grid = np.logspace(13.5, 15.5, g)
    zs_grid   = np.linspace(0.30, 1.0,  g)
    n_cells   = g * g
    n_rep     = max(1, min(5, 1000 // n_cells))
    args      = (gt_data, gt_err, mass_grid, zs_grid)
    med, _    = time_fn(compute_posterior_grid, args, n_repeats=n_rep)
    pg_times.append(med)
    pg_n_cells.append(n_cells)
    print(f"  grid {g:2d}×{g:2d} = {n_cells:4d} cells:  {med:7.1f} ms  "
          f"({med/n_cells:.2f} ms/cell)")

pg_times   = np.array(pg_times)
pg_n_cells = np.array(pg_n_cells, float)

fig, ax = plt.subplots(figsize=(7, 5))
ax.loglog(pg_n_cells, pg_times, "o-", color="#7b3294", lw=2, ms=7, label="Wall time")
ref_t_pg = nfw_med * pg_n_cells
ax.loglog(pg_n_cells, ref_t_pg, "k--", lw=0.9,
          label=f"{nfw_med:.2f} ms × N_cells (expected)")
ax.set_xlabel(r"Grid cells ($N_M \times N_{z_s}$)", fontsize=12)
ax.set_ylabel("Wall time (ms)", fontsize=12)
ax.set_title(
    "compute_posterior_grid: scaling with grid resolution\n"
    r"(double loop; one build_nfw_theory call per $(M, z_s)$ cell)",
    fontsize=11,
)
ax.legend(fontsize=9)
ax.grid(True, which="both", ls=":", alpha=0.4)
fig.tight_layout()
fig.savefig(OUTPUT_DIR / "perf_posterior_grid.png", dpi=200, bbox_inches="tight")
plt.close(fig)
print(f"\n  Saved {OUTPUT_DIR}/perf_posterior_grid.png")


# ---------------------------------------------------------------------------
# Print machine-readable summary for documentation
# ---------------------------------------------------------------------------

print("\n" + "=" * 65)
print("SUMMARY (for documentation)")
print("=" * 65)

print(f"\nEllipticity estimators at {TYPICAL_SIZE}×{TYPICAL_SIZE} stamp:")
print(f"  {'Estimator':28s}  {'ms/gal':>8}  {'gal/s':>8}  {'peak KB':>8}")
for name in estimators:
    ms  = e_time_med[name][idx_typ]
    gps = 1000.0 / ms
    kb  = e_mem_kb[name][idx_typ]
    print(f"  {LABELS[name]:28s}  {ms:8.3f}  {gps:8.0f}  {kb:8.0f}")

print(f"\nTime complexity fits (log-log slope ~ expected 2.0 for O(N²)):")
for name in estimators:
    slope = np.polyfit(np.log(sizes_arr**2), np.log(e_time_med[name]), 1)[0]
    print(f"  {LABELS[name]:28s}  slope = {slope:.2f}")

print(f"\nbuild_nfw_theory: {nfw_med:.3f} ms per call")

print(f"\ncompute_tangential_ellipticities (N_halos={N_HALOS_FIXED}):")
print(f"  {'N_sources':>10}  {'time (ms)':>10}  {'memory (MB)':>12}")
for i, n_s in enumerate(N_SOURCES_SERIES):
    print(f"  {n_s:10d}  {te_times[i]:10.2f}  {te_mem_kb[i]/1024:12.2f}")

print(f"\ncompute_posterior_grid:")
print(f"  {'Grid':>8}  {'N_cells':>8}  {'time (ms)':>10}  {'ms/cell':>9}")
for i, g in enumerate(GRID_SIZES):
    nc = pg_n_cells[i]
    print(f"  {g:2d}×{g:<2d}  {nc:8.0f}  {pg_times[i]:10.1f}  {pg_times[i]/nc:9.2f}")
