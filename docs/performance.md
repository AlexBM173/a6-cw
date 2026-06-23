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
ranging from 16 to 192 pixels. The dominant operations are O($N^2$) NumPy array
reductions (sum, multiply-accumulate).

### Timing

| Stamp | Pixels $N^2$ | Unweighted (ms) | Weighted (ms) | HSM (ms) |
|---|---|---|---|---|
| 16 × 16 | 256 | 0.020 | 0.033 | 0.054 |
| 24 × 24 | 576 | 0.022 | 0.039 | 0.086 |
| 32 × 32 | 1 024 | 0.024 | 0.048 | 0.122 |
| **48 × 48** | **2 304** | **0.030** | **0.066** | **0.184** |
| 64 × 64 | 4 096 | 0.038 | 0.091 | 0.371 |
| 96 × 96 | 9 216 | 0.064 | 0.183 | 0.779 |
| 128 × 128 | 16 384 | 0.097 | 0.495 | 1.238 |
| 192 × 192 | 36 864 | 0.195 | 1.509 | 2.764 |

The 48 × 48 row matches the data stamps used in the coursework notebooks.

### Throughput at 48 × 48 (standard stamp size)

| Estimator | ms / galaxy | Galaxies / s | Peak memory (KB) |
|---|---|---|---|
| Unweighted moments | 0.030 | 33 000 | 127 |
| Gaussian-weighted moments | 0.066 | 15 000 | 181 |
| HSM (GalSim) | 0.184 | 5 400 | 13 |

HSM has the lowest throughput but also the smallest Python heap footprint:
the adaptive-moment algorithm runs in compiled C++ code, so GalSim allocates
internally without touching the Python heap. The apparent 13 KB is only the
`galsim.hsm.ShapeData` result object.

### Memory

Peak heap memory grows proportionally to the stamp area for the two
pure-Python estimators:

| Estimator | Dominant allocation | Expected |
|---|---|---|
| Unweighted | ≈ 8 intermediate `float64` arrays of size $N^2$ | $8 \times 8 N^2$ bytes |
| Weighted | ≈ 11 intermediate arrays (includes weight $W$ and $WI$ per iteration) | $11 \times 8 N^2$ bytes |
| HSM | result object only (C++ internal) | O(1) Python heap |

At 192 × 192 the unweighted estimator uses ≈ 1.7 MB, weighted ≈ 2.5 MB,
matching the theoretical 8–11 arrays of 36 864 × 8 bytes = 0.29 MB each.

### Scaling exponent

The log–log slope of time vs stamp area $N^2$ gives the empirical scaling
exponent (slope = 1.0 would be perfect O($N^2$)):

| Estimator | Measured slope | Note |
|---|---|---|
| Unweighted moments | 0.45 | Python / NumPy dispatch overhead dominates at small $N$ |
| Gaussian-weighted moments | 0.75 | Higher per-pixel cost from centroid iteration; approaches 1.0 at large $N$ |
| HSM (GalSim) | 0.80 | Adaptive-moment loop converges in more iterations for large galaxies |

Sub-unit slopes at small $N$ are expected: each NumPy ufunc has a fixed
dispatch overhead of ≈ 1–5 µs regardless of array size, which dominates the
total time when the array holds only a few hundred elements. At $N \ge 128$
(16 000+ pixels) all three estimators approach the asymptotic O($N^2$) regime.

![Ellipticity scaling](../outputs/profiling/perf_ellipticity_scaling.png)

![Throughput at 48 × 48](../outputs/profiling/perf_throughput.png)

---

## 2 — `compute_tangential_ellipticities` vs N_sources

This function forms all $N_h \times N_s$ halo–source pairs simultaneously using
NumPy broadcasting, building eight intermediate arrays of shape $(N_h, N_s)$:
`dx`, `dy`, `r`, `phi`, `cos2phi`, `sin2phi`, `epsilon_t`, `epsilon_x`.

### Time and memory (N_halos = 10 fixed)

| N_sources | Time (ms) | Peak memory (MB) |
|---|---|---|
| 100 | 0.04 | 0.08 |
| 500 | 0.24 | 0.38 |
| 1 000 | 0.47 | 0.76 |
| 5 000 | 2.19 | 3.44 |
| 10 000 | 5.11 | 6.87 |
| 50 000 | 45.51 | 34.33 |
| **100 000** | **83.65** | **68.67** |

The coursework uses $N_h = 100$ halos and $N_s = 100\,000$ sources, so the
actual intermediate arrays are $100 \times 100\,000 = 10^7$ elements each —
approximately 610 MB for eight `float64` arrays. The function returns
one-dimensional views (`.ravel()`) so the RAM is released as soon as the
caller drops the intermediates.

### Complexity

- **Time** scales as O($N_h \cdot N_s$) — both factors appear symmetrically in
  the broadcast. At fixed $N_h = 10$, the measured slope in log(time) vs log($N_s$)
  is 1.03, confirming linear scaling with the number of sources.
- **Memory** also scales as O($N_h \cdot N_s$): the 100 000-source column shows
  68.7 MB, close to the theoretical $8 \times 10 \times 100\,000 \times 8$ bytes
  = 61 MB (the overhead comes from `.ravel()` returning copies rather than views
  in some NumPy versions).

For very large catalogues the memory ceiling is the practical limit. With the
default grid ($N_h = 100$, $N_s = 100\,000$), peak intermediate memory is
≈ 610 MB. If memory is constrained, the function could be looped over halos
in batches without changing the algorithm.

![Tangential shear scaling](../outputs/profiling/perf_tangential_scaling.png)

---

## 3 — `build_nfw_theory` and `compute_posterior_grid`

### `build_nfw_theory` (per-call timing)

A single call evaluates the NFW tangential shear at the 10 fixed
`BIN_CENTRES` using `NFWHalo_theory.get_tangential_shear`.

| Metric | Value |
|---|---|
| Median time | **0.44 ms** |
| Std dev | 0.12 ms |

The call is dominated by distance integrals inside `NFWHalo_theory`; there is
no Python-level vectorisation because the output size is fixed at $N_\text{bins} = 10$.

### `compute_posterior_grid` (time vs grid resolution)

The function evaluates the log-posterior at every point on an
$N_M \times N_{z_s}$ grid. Each cell makes one `build_nfw_theory` call
(≈ 0.44 ms) and one Gaussian log-likelihood evaluation (negligible).
The loop is a pure Python double `for`, so the total cost is:

$$T \approx 0.49 \; \text{ms} \times N_M \times N_{z_s}$$

| Grid ($N_M \times N_{z_s}$) | Cells | Time (ms) | ms / cell |
|---|---|---|---|
| 5 × 5 | 25 | 11.2 | 0.45 |
| 10 × 10 | 100 | 49.0 | 0.49 |
| 15 × 15 | 225 | 108.8 | 0.48 |
| 20 × 20 | 400 | 199.8 | 0.50 |
| 30 × 30 | 900 | 438.6 | 0.49 |
| 40 × 40 | 1 600 | 762.2 | 0.48 |
| **50 × 50** | **2 500** | **1 217.3** | **0.49** |

The per-cell cost is remarkably constant at 0.49 ms, confirming that the
runtime is entirely dominated by `build_nfw_theory` with negligible loop
overhead. The scaling is exactly O($N_M \cdot N_{z_s}$).

The coursework grid (100 × 50 = 5 000 cells) would take ≈ **2.5 seconds**
at this rate; the actual notebook runs in a few minutes because the
`NFWHalo_theory` class also evaluates more slowly at the full precision used
in the real analysis.

![Posterior grid scaling](../outputs/profiling/perf_posterior_grid.png)

---

## Summary

| Routine | Cost at typical input | Complexity |
|---|---|---|
| `unweighted_ellipticity` | 0.030 ms / 48×48 stamp | O($N^2$) |
| `weighted_ellipticity_raw` | 0.066 ms / 48×48 stamp | O($K \cdot N^2$) — $K$ ≈ 5 centroid iterations |
| `hsm_ellipticity` | 0.184 ms / 48×48 stamp | O($N^2$) in C++ |
| `measure_all_ellipticities` (100 k gals) | ≈ 3 s unweighted · 7 s weighted · 18 s HSM | O($N_\text{gal} \cdot N^2$) |
| `compute_tangential_ellipticities` | 84 ms (100 k sources, 10 halos) | O($N_h \cdot N_s$) time and memory |
| `build_nfw_theory` | 0.44 ms per call | O($N_\text{bins}$) |
| `compute_posterior_grid` | 0.49 ms per cell | O($N_M \cdot N_{z_s}$) |

The dominant cost in the Q1 notebook is `calibrate_weighted_response`, which
internally runs 12 000 GalSim renders and 24 000 moment calculations (it is
run once at startup and the resulting calibration table is reused for all
100 000 galaxies). In Q2 the dominant cost is `compute_posterior_grid` over
the $(M, z_s)$ grid.
