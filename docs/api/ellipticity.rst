a6cw.ellipticity
================

.. automodule:: a6cw.ellipticity
   :no-members:

Shape estimators
----------------

.. autofunction:: a6cw.ellipticity.unweighted_ellipticity
.. autofunction:: a6cw.ellipticity.weighted_ellipticity_raw
.. autofunction:: a6cw.ellipticity.hsm_ellipticity

Calibration
-----------

.. autofunction:: a6cw.ellipticity.calibrate_weighted_response
.. autofunction:: a6cw.ellipticity.apply_simulation_calibration

Batch processing
----------------

.. autofunction:: a6cw.ellipticity.measure_all_ellipticities

Plotting
--------

.. autofunction:: a6cw.ellipticity.plot_ellipticity_comparison

Regression
----------

.. autofunction:: a6cw.ellipticity._ols_through_origin

Internal helpers
----------------

These are implementation details; the public API above covers normal use.

.. autofunction:: a6cw.ellipticity._pixel_grids
.. autofunction:: a6cw.ellipticity._gaussian_weight
