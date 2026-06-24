a6cw — Weak Gravitational Lensing
==================================

A6 coursework package for measuring galaxy shapes and inferring NFW halo masses
from weak gravitational lensing.  All reusable code lives in the ``a6cw`` Python
package so that the Jupyter notebooks contain only top-level orchestration.

.. toctree::
   :maxdepth: 2
   :caption: User guide

   package
   performance

.. toctree::
   :maxdepth: 2
   :caption: API reference

   api/index

.. toctree::
   :maxdepth: 1
   :caption: Development

   tests

Quick start
-----------

.. code-block:: bash

   # activate the venv and run the notebooks
   source activate_m2_work.sh
   cd projects/a6-cw
   jupyter notebook

   # run the test suite
   python -m pytest tests/ -v

   # regenerate performance plots
   python profiling/run_profiling.py

Package layout
--------------

.. code-block:: text

   a6cw/
       ellipticity.py   shape estimators, Numba JIT kernels, calibration
       shear.py         shear profiles, NFW theory, posterior inference
   tests/
       test_ellipticity.py
       test_shear.py
   profiling/
       run_profiling.py
   docs/               (this documentation)
   question1.ipynb     Q1: ellipticity estimation
   question2.ipynb     Q2: shear profiles and mass inference
