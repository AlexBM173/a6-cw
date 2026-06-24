a6cw.shear
==========

.. automodule:: a6cw.shear
   :no-members:

Module constants
----------------

.. py:data:: a6cw.shear.N_BINS
   :value: 10

   Number of logarithmically-spaced radial bins used throughout the shear pipeline.

.. py:data:: a6cw.shear.BIN_EDGES

   Array of shape ``(N_BINS + 1,)`` — logarithmically-spaced bin edges from 100 to 3600 arcsec.

.. py:data:: a6cw.shear.BIN_CENTRES

   Array of shape ``(N_BINS,)`` — geometric means of adjacent ``BIN_EDGES``.

Data loading
------------

.. autofunction:: a6cw.shear.load_positions

Shear pipeline
--------------

.. autofunction:: a6cw.shear.compute_tangential_ellipticities
.. autofunction:: a6cw.shear.mean_tangential_shear
.. autofunction:: a6cw.shear.estimate_shape_noise_variance
.. autofunction:: a6cw.shear.optimal_weights

NFW theory
----------

.. autofunction:: a6cw.shear.build_nfw_theory

Bayesian inference
------------------

.. autofunction:: a6cw.shear.log_likelihood
.. autofunction:: a6cw.shear.compute_posterior_grid

Plotting
--------

.. autofunction:: a6cw.shear.plot_tangential_shear_profiles
.. autofunction:: a6cw.shear.plot_cross_shear_null_test
.. autofunction:: a6cw.shear.plot_joint_posterior
.. autofunction:: a6cw.shear.plot_mass_posterior_fixed_zs
.. autofunction:: a6cw.shear.plot_best_fit_overlay
