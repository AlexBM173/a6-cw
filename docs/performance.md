# Performance and Profiling

This document records measured wall-clock times and peak heap memory for the
computationally significant routines in `a6cw`. All measurements were taken on
the same machine used to run the notebooks (WSL2, Python 3.11.11, NumPy 2.x,
GalSim 2.x). Times are medians over many repeated calls; peak memory uses
Python's `tracemalloc`.

Reproduce all results and regenerate the plots with:

```bash
python profiling/run_profiling.py
```

Plots are written to `outputs/profiling/`.

---

## Optimisations applied (complete list)

All 12 identified optimisations have been implemented across two rounds.

| # | Module / file | Technique | Function(s) affected |
|---|---|---|---|
| 1 | `ellipticity` | Numba `@njit(cache=True, nogil=True)` — fused pixel loop, no intermediate arrays | `_unweighted_jit` |
| 2 | `ellipticity` | Numba `@njit(cache=True, nogil=True)` — fused centroid iteration, precomputed `inv_2s2`, convergence via `shift²<tol²` | `_weighted_jit` |
| 3 | `ellipticity` | `functools.lru_cache` on pixel-grid arrays keyed by shape | `_pixel_grids_cache` |
| 4 | `ellipticity` | `concurrent.futures.ThreadPoolExecutor` (Numba + GalSim C++ both release GIL) | `measure_all_ellipticities` |
| 5 | `ellipticity` | Single GalSim draw per HLR (noiseless stamps are deterministic) | `calibrate_weighted_response` |
| 6 | `shear` | Trig-free double-angle identities: `cos2φ=(dx²−dy²)/r²`, `sin2φ=2dx·dy/r²` | `compute_tangential_ellipticities` |
| 7 | `shear` | `np.digitize` + `np.bincount` replace O(N·B) boolean-mask loop | `mean_tangential_shear` (unweighted path) |
| 8 | `shear` | `np.bincount` vectorised weighted sums replaces per-bin loop | `mean_tangential_shear` (weighted path) |
| 9 | `shear` | `concurrent.futures.ProcessPoolExecutor` (bypasses GIL for NFW calls) | `compute_posterior_grid` |
| 10 | `shear` | `scipy.integrate.cumulative_trapezoid` replaces O(N²) list comprehension | `plot_joint_posterior`, `plot_mass_posterior_fixed_zs` |
| 11 | `nfw_theory` | Vectorised `__gamma` call: broadcast `ks` to array shape, single call over all bins | `NFWHalo_theory.get_tangential_shear` |

---

## Methodology

| Quantity | Tool | Detail |
|---|---|---|
| Wall-clock time | `time.perf_counter` | Median over 50–100 calls; one warm-up call excluded |
| Peak heap memory | `tracemalloc` | Single call; measures Python-level allocation only |
| Scaling exponent | `numpy.polyfit` on log–log data | Slope in log(time) vs log(input size) |

For the ellipticity estimators, synthetic noiseless `galsim.Exponential` stamps
are used (half-light radius ≈ stamp_size / 8 px, `g1 = 0.1`, `method='no_pixel'`).
For `compute_tangential_ellipticities`, positions and ellipticities are drawn
from a fixed RNG seed. For `compute_posterior_grid`, a flat synthetic shear
profile is used (`gt = 0.02`, `gt_err = 0.005` per bin).

---

## 1 — Ellipticity estimators vs stamp size

### Before/after at 48 × 48 (standard stamp size)

| Estimator | Time before (original) | Time after (all optimisations) | Total speedup | Memory before | Memory after | Memory reduction |
|---|---|---|---|---|---|---|
| Unweighted moments | 0.030 ms | 0.007 ms | **4×** | 127 KB | 18 KB | **7×** |
| Gaussian-weighted moments | 0.066 ms | 0.025 ms | **2.6×** | 181 KB | 18 KB | **10×** |
| HSM (GalSim) | 0.184 ms | 0.175 ms | 1.1× | 13 KB | 13 KB | unchanged |

The Numba JIT kernels fuse the pixel loops into scalar accumulators, eliminating
every intermediate NumPy array. Both memory cost and per-call overhead drop to
the input array itself (18 KB = 2 304 × 8 bytes for a 48 × 48 `float64` stamp).

### Full scaling table (after all optimisations)

| Stamp | Pixels $N^2$ | Unweighted (ms) | Weighted (ms) | HSM (ms) |
|---|---|---|---|---|
| 16 × 16 | 256 | 0.001 | 0.003 | 0.053 |
| 24 × 24 | 576 | 0.002 | 0.007 | 0.082 |
| 32 × 32 | 1 024 | 0.003 | 0.012 | 0.111 |
| **48 × 48** | **2 304** | **0.007** | **0.025** | **0.175** |
| 64 × 64 | 4 096 | 0.009 | 0.046 | 0.274 |
| 96 × 96 | 9 216 | 0.019 | 0.120 | 0.677 |
| 128 × 128 | 16 384 | 0.034 | 0.426 | 1.163 |
| 192 × 192 | 36 864 | 0.075 | 0.806 | 2.690 |

### Throughput at 48 × 48

| Estimator | ms / galaxy | Galaxies / s | Peak memory (KB) |
|---|---|---|---|
| Unweighted moments | 0.007 | 148 000 | 18 |
| Gaussian-weighted moments | 0.025 | 40 000 | 18 |
| HSM (GalSim) | 0.175 | 5 700 | 13 |

![Ellipticity scaling](../outputs/profiling/perf_ellipticity_scaling.png)

![Throughput at 48 × 48](../outputs/profiling/perf_throughput.png)

---

## 2 — `calibrate_weighted_response` (optimisation 5)

This function was the dominant startup cost in Q1: it originally drew
`n_per_hlr = 1000` noiseless stamps per HLR point and averaged their
ellipticities. Because the stamps are fully deterministic (no noise, fixed
profile), all 1000 draws for a given HLR are pixel-identical, making the
loop equivalent to drawing once and repeating the result.

### Before/after

| | Before | After | Speedup |
|---|---|---|---|
| GalSim renders | 12 HLRs × 1000 = **12 000** | 12 HLRs × 1 = **12** | **1 000×** fewer renders |
| Total time | ~14 s | **60 ms** | **~230×** |

The ~230× speedup rather than 1000× reflects fixed overhead per HLR
(cosmology initialisation, pixel convolution) not removed by the draw-count
reduction. Per-HLR cost is now ≈ 5 ms.

---

## 3 — `compute_tangential_ellipticities` vs N_sources

### Before/after at N_sources = 100 000 (N_halos = 10)

| Metric | Original | After trig-free | Improvement |
|---|---|---|---|
| Time | 83.65 ms | 38.81 ms | **2.2×** |

Trig-free identities (`cos2φ = (dx²−dy²)/r²`, `sin2φ = 2dx·dy/r²`) eliminate
two `np.arctan2` and two transcendental `cos`/`sin` calls over the full pair array.

### Scaling table

| N_sources | Time (ms) | Peak memory (MB) |
|---|---|---|
| 100 | 0.02 | 0.09 |
| 500 | 0.05 | 0.43 |
| 1 000 | 0.09 | 0.85 |
| 5 000 | 0.36 | 3.86 |
| 10 000 | 1.13 | 7.73 |
| 50 000 | 18.02 | 38.63 |
| **100 000** | **38.81** | **77.25** |

![Tangential shear scaling](../outputs/profiling/perf_tangential_scaling.png)

---

## 4 — `build_nfw_theory` (optimisation 11)

`NFWHalo_theory.get_tangential_shear` originally called the private GalSim
method `__gamma(r, ks)` once per radial bin in a Python list comprehension.
`__gamma` internally uses `np.where` and NumPy operations and accepts arrays
natively. Passing the full `rs` array (with `ks` broadcast to the same shape)
replaces 10 Python loop iterations with one vectorised C-level call.

### Before/after (per call, 10 bins fixed)

| | Before | After | Speedup |
|---|---|---|---|
| `build_nfw_theory` | 0.421 ms | **0.070 ms** | **6×** |

---

## 5 — `compute_posterior_grid` (optimisations 9 + 11 combined)

Two independent optimisations compound here:
- Opt 9: `ProcessPoolExecutor` parallelises NFW evaluations across OS processes.
- Opt 11: Vectorised `__gamma` reduces per-cell NFW cost from 0.421 ms to 0.070 ms.

### Before/after

| Grid | Cells | Original (serial, slow NFW) | After parallel + fast NFW | Total speedup |
|---|---|---|---|---|
| 5 × 5 | 25 | 11.2 ms | 2.1 ms | **5.3×** |
| 20 × 20 | 400 | 199.8 ms | 120.7 ms | **1.7×** |
| 30 × 30 | 900 | 438.6 ms | 104.5 ms | **4.2×** |
| 50 × 50 | 2 500 | 1 217.3 ms | 131.3 ms | **9.3×** |

The 9.3× total speedup at 2 500 cells comes from:
- ~6× from the vectorised NFW per cell (0.421 ms → 0.070 ms)
- ~1.6× additional from `ProcessPoolExecutor` amortisation over workers

The per-cell cost falls from a fixed 0.49 ms (serial) to 0.05 ms at
2 500 cells. The coursework grid (100 × 50 = 5 000 cells) would take
≈ **0.35 s** versus the original ≈ **2.5 s** serial — a **7× speedup**.

### Scaling table (after both optimisations)

| Grid | Cells | Time (ms) | ms / cell |
|---|---|---|---|
| 5 × 5 | 25 | 2.1 | 0.08 |
| 10 × 10 | 100 | 101.0 | 1.01 |
| 15 × 15 | 225 | 105.9 | 0.47 |
| 20 × 20 | 400 | 120.7 | 0.30 |
| 30 × 30 | 900 | 104.5 | 0.12 |
| 40 × 40 | 1 600 | 109.3 | 0.07 |
| 50 × 50 | 2 500 | 131.3 | 0.05 |

![Posterior grid scaling](../outputs/profiling/perf_posterior_grid.png)

---

## 6 — CDF computation in posterior plots (optimisation 10)

`plot_joint_posterior` and `plot_mass_posterior_fixed_zs` both computed the
cumulative distribution function for 1D marginal posteriors using a list
comprehension:

```python
cdf = np.array([np.trapezoid(p[:k+1], x[:k+1]) for k in range(len(x))])
```

Each call to `np.trapezoid(p[:k+1], …)` scans `k` elements, so the total
work is O(N²) where N is the grid resolution. Replacing with
`scipy.integrate.cumulative_trapezoid` (a single O(N) pass):

```python
cdf = np.concatenate([[0.0], cumulative_trapezoid(p, x)])
```

The fix is O(N) and more accurate (no accumulation of floating-point rounding
from repeated prefix sums). For a 100-point 1D grid the saving is small in
absolute time; for a finely spaced posterior grid (N > 200) it becomes
appreciable.

---

## Summary

### Complete before/after table

| Routine | Cost before (original) | Cost after (all optimisations) | Speedup |
|---|---|---|---|
| `unweighted_ellipticity` / 48×48 stamp | 0.030 ms | 0.007 ms | **4×** |
| `weighted_ellipticity_raw` / 48×48 stamp | 0.066 ms | 0.025 ms | **2.6×** |
| `hsm_ellipticity` / 48×48 stamp | 0.184 ms | 0.175 ms | 1.1× |
| `calibrate_weighted_response` (12 HLRs, 1000/HLR) | ~14 s | 60 ms | **~230×** |
| `compute_tangential_ellipticities` (100 k sources, 10 halos) | 84 ms | 39 ms | **2.2×** |
| `build_nfw_theory` (10 bins) | 0.421 ms | 0.070 ms | **6×** |
| `compute_posterior_grid` (2 500 cells) | 1 217 ms | 131 ms | **9.3×** |
| `credible_interval_1d` CDF | O($N^2$) | O($N$) via `cumulative_trapezoid` | algorithmic |

### Memory

| Routine | Memory before | Memory after |
|---|---|---|
| `unweighted_ellipticity` / 48×48 | 127 KB | 18 KB (7×) |
| `weighted_ellipticity_raw` / 48×48 | 181 KB | 18 KB (10×) |
| `hsm_ellipticity` / 48×48 | 13 KB | 13 KB |
