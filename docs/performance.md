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

## Optimisations applied

The package was rewritten after an initial profiling pass. The following
optimisations were made and their effects are shown in the before/after tables
below.

| Module | Technique | Functions affected |
|---|---|---|
| `ellipticity` | Numba `@njit(cache=True, nogil=True)` JIT kernel | `unweighted_ellipticity`, `weighted_ellipticity_raw` |
| `ellipticity` | `functools.lru_cache` on pixel-grid arrays (keyed by shape) | `_pixel_grids` |
| `ellipticity` | `concurrent.futures.ThreadPoolExecutor` (GIL-free Numba + GalSim C++) | `measure_all_ellipticities` |
| `shear` | Trig-free double-angle identities: `cos2φ = (dx²−dy²)/r²`, `sin2φ = 2dx·dy/r²` | `compute_tangential_ellipticities` |
| `shear` | `np.digitize` + `np.bincount` replace O(N·B) boolean-mask loop | `mean_tangential_shear` |
| `shear` | `concurrent.futures.ProcessPoolExecutor` (bypasses GIL for NFW calls) | `compute_posterior_grid` |

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

The three estimators are applied to single stamps of increasing side length $N$,
ranging from 16 to 192 pixels.

### Before/after comparison at 48 × 48 (standard stamp size)

| Estimator | Time before | Time after | Speedup | Memory before | Memory after | Memory reduction |
|---|---|---|---|---|---|---|
| Unweighted moments | 0.030 ms | 0.005 ms | **6×** | 127 KB | 18 KB | **7×** |
| Gaussian-weighted moments | 0.066 ms | 0.026 ms | **2.5×** | 181 KB | 18 KB | **10×** |
| HSM (GalSim) | 0.184 ms | 0.174 ms | 1.1× | 13 KB | 13 KB | unchanged |

The Numba JIT kernels fuse the three-pass pixel loop into a single pass with
no intermediate NumPy arrays, explaining both the speed and memory gains.
The weighted estimator also benefits from precomputing `inv_2s2 = 1/(2σ²)` to
avoid recomputing it each centroid iteration, and from replacing `sqrt` in the
convergence check with a squared-distance comparison.

HSM was already compiled C++ with negligible Python overhead, so it is unchanged.

### Full scaling tables (after optimisation)

| Stamp | Pixels $N^2$ | Unweighted (ms) | Weighted (ms) | HSM (ms) |
|---|---|---|---|---|
| 16 × 16 | 256 | 0.001 | 0.003 | 0.054 |
| 24 × 24 | 576 | 0.003 | 0.008 | 0.079 |
| 32 × 32 | 1 024 | 0.003 | 0.012 | 0.110 |
| **48 × 48** | **2 304** | **0.005** | **0.026** | **0.174** |
| 64 × 64 | 4 096 | 0.009 | 0.044 | 0.274 |
| 96 × 96 | 9 216 | 0.019 | 0.119 | 0.666 |
| 128 × 128 | 16 384 | 0.034 | 0.424 | 1.151 |
| 192 × 192 | 36 864 | 0.075 | 0.789 | 2.658 |

### Throughput at 48 × 48

| Estimator | ms / galaxy | Galaxies / s | Peak memory (KB) |
|---|---|---|---|
| Unweighted moments | 0.005 | 191 000 | 18 |
| Gaussian-weighted moments | 0.026 | 39 000 | 18 |
| HSM (GalSim) | 0.174 | 5 750 | 13 |

### Memory

The Numba JIT kernels compute all sums in scalar accumulators with no heap
allocations beyond the input array itself. At 48 × 48, both moment estimators
now show 18 KB, which is the input `float64` array (2 304 × 8 bytes = 18 KB)
with no additional intermediates.

### Scaling exponent

| Estimator | Measured slope |
|---|---|
| Unweighted moments | 0.84 |
| Gaussian-weighted moments | 1.12 |
| HSM (GalSim) | 0.79 |

The sub-unit slopes reflect fixed Numba dispatch overhead dominating at small
stamps; all estimators approach O($N^2$) asymptotically.

![Ellipticity scaling](../outputs/profiling/perf_ellipticity_scaling.png)

![Throughput at 48 × 48](../outputs/profiling/perf_throughput.png)

---

## 2 — `compute_tangential_ellipticities` vs N_sources

The trig-free identities (`cos2φ = (dx²−dy²)/r²`, `sin2φ = 2dx·dy/r²`) eliminate
two `np.arctan2` and two `np.cos`/`np.sin` calls, each of which is a transcendental
function evaluated element-wise over the full pair array.

### Before/after at N_sources = 100 000 (N_halos = 10)

| Metric | Before | After | Improvement |
|---|---|---|---|
| Time | 83.65 ms | 39.43 ms | **2.1×** |
| Peak memory | 68.67 MB | 77.25 MB | similar |

Memory is slightly higher after because the trig-free path computes `dx²` and `dy²`
as separate temporary arrays before forming `r²`, whereas the original computed
`phi` from `arctan2(dy, dx)` without materialising squared components. The
time saving dominates.

### Scaling table (after optimisation)

| N_sources | Time (ms) | Peak memory (MB) |
|---|---|---|
| 100 | 0.04 | 0.09 |
| 500 | 0.05 | 0.43 |
| 1 000 | 0.09 | 0.85 |
| 5 000 | 0.40 | 3.86 |
| 10 000 | 0.89 | 7.73 |
| 50 000 | 20.24 | 38.63 |
| **100 000** | **39.43** | **77.25** |

- **Time** scales as O($N_h \cdot N_s$); measured log–log slope ≈ 1.0.
- **Memory** scales as O($N_h \cdot N_s$); at 100 k sources and 10 halos the
  dominant arrays are $10 \times 100\,000$ `float64` = 7.6 MB each, and there
  are approximately ten such intermediates.

![Tangential shear scaling](../outputs/profiling/perf_tangential_scaling.png)

---

## 3 — `build_nfw_theory` and `compute_posterior_grid`

### `build_nfw_theory` (per-call timing)

| Metric | Value |
|---|---|
| Median time | **0.421 ms** |
| Std dev | 0.036 ms |

This function is unchanged; the cost is entirely inside `NFWHalo_theory` C code.

### `compute_posterior_grid` — before/after

The original implementation used a serial Python double `for` loop. The optimised
version uses `concurrent.futures.ProcessPoolExecutor`, which bypasses the GIL
and parallelises the NFW evaluations across OS processes.

| Grid | Cells | Time before (ms) | Time after (ms) | Speedup |
|---|---|---|---|---|
| 5 × 5 | 25 | 11.2 | 11.3 | 1.0× (startup overhead dominates) |
| 10 × 10 | 100 | 49.0 | 107.0 | – (overhead dominates) |
| 15 × 15 | 225 | 108.8 | 118.4 | 0.9× (marginal) |
| 20 × 20 | 400 | 199.8 | 127.4 | **1.6×** |
| 30 × 30 | 900 | 438.6 | 157.2 | **2.8×** |
| 40 × 40 | 1 600 | 762.2 | 211.5 | **3.6×** |
| 50 × 50 | 2 500 | 1 217.3 | 281.7 | **4.3×** |

For small grids (< 50 cells) the serial fallback is used automatically, avoiding
process-pool startup overhead. For grids of 50 × 50 and beyond, the parallelism
provides a 4× or greater speedup that scales with the number of CPU cores.

The per-cell cost drops from a fixed 0.49 ms (serial) to ~0.11 ms at 2 500 cells
(amortised across worker processes), reflecting near-ideal strong scaling on this
CPU-bound task.

The coursework grid (100 × 50 = 5 000 cells) would take ≈ **0.55 s** with the
optimised code versus ≈ **2.5 s** serial — a **4.5× speedup**.

![Posterior grid scaling](../outputs/profiling/perf_posterior_grid.png)

---

## Summary

### Before optimisation

| Routine | Cost at typical input | Complexity |
|---|---|---|
| `unweighted_ellipticity` | 0.030 ms / 48×48 stamp | O($N^2$) |
| `weighted_ellipticity_raw` | 0.066 ms / 48×48 stamp | O($K \cdot N^2$) |
| `hsm_ellipticity` | 0.184 ms / 48×48 stamp | O($N^2$) in C++ |
| `compute_tangential_ellipticities` | 84 ms (100 k sources, 10 halos) | O($N_h \cdot N_s$) |
| `build_nfw_theory` | 0.44 ms per call | O($N_\text{bins}$) |
| `compute_posterior_grid` | 0.49 ms per cell (serial) | O($N_M \cdot N_{z_s}$) |

### After optimisation

| Routine | Cost at typical input | Speedup | Complexity |
|---|---|---|---|
| `unweighted_ellipticity` | 0.005 ms / 48×48 stamp | **6×** | O($N^2$), JIT-compiled |
| `weighted_ellipticity_raw` | 0.026 ms / 48×48 stamp | **2.5×** | O($K \cdot N^2$), JIT-compiled |
| `hsm_ellipticity` | 0.174 ms / 48×48 stamp | 1.1× | O($N^2$) in C++ (unchanged) |
| `compute_tangential_ellipticities` | 39 ms (100 k sources, 10 halos) | **2.1×** | O($N_h \cdot N_s$) trig-free |
| `build_nfw_theory` | 0.421 ms per call | 1.0× | O($N_\text{bins}$) (unchanged) |
| `compute_posterior_grid` | 0.11 ms per cell (parallel) | **4.3×** at 2 500 cells | O($N_M \cdot N_{z_s}$ / cores) |

Memory for `unweighted_ellipticity` and `weighted_ellipticity_raw` at 48 × 48 dropped
from 127–181 KB to 18 KB (7–10×), since the JIT kernels accumulate in scalars
rather than materialising intermediate arrays.
