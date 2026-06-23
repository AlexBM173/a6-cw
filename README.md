# a6-cw — Weak Gravitational Lensing Coursework

A6 assessment: measuring galaxy shapes and inferring halo masses from weak gravitational lensing.

The analysis is split across two Jupyter notebooks (`question1.ipynb`, `question2.ipynb`). All reusable code lives in the `a6cw` Python package so the notebooks contain only top-level orchestration.

## Repository layout

```
a6cw/
    __init__.py
    ellipticity.py      # shape estimators and calibration
    shear.py            # shear pipeline, NFW theory, posterior inference
tests/
    test_ellipticity.py
    test_shear.py
docs/
    package.md          # full API reference for a6cw
    tests.md            # documentation of the test suite
question1.ipynb         # Q1: ellipticity estimation
question2.ipynb         # Q2: shear profiles and mass inference
nfw_theory.py           # NFWHalo_theory helper class (provided)
data/                   # input catalogues and galaxy stamps (not tracked)
outputs/                # figures and saved arrays (not tracked)
```

## Setup

The notebooks and package require the virtual environment at
`~/projects/a6_coursework`. Activate it before running anything:

```bash
# from the home directory
source activate_m2_work.sh
cd projects/a6-cw
```

## Running the notebooks

Open `question1.ipynb` first — it measures galaxy ellipticities and writes
`outputs/ellipticities_*.npy`, which `question2.ipynb` reads.

```bash
jupyter notebook
```

## Running the tests

```bash
python -m pytest tests/ -v
```

All 82 tests should pass in under 5 seconds.

## Documentation

- [Package reference](docs/package.md) — API for `a6cw.ellipticity` and `a6cw.shear`
- [Test documentation](docs/tests.md) — design rationale and analytic basis for every test class
