"""
a6cw — weak gravitational lensing analysis for the A6 coursework.

Submodules
----------
ellipticity
    Galaxy shape estimators (unweighted moments, Gaussian-weighted moments,
    HSM) and simulation-based shear-response calibration.
shear
    Tangential shear pipeline, NFW theory predictions, and Bayesian mass
    inference on a 2-D (mass, z_source) posterior grid.
"""

__version__ = "1.0.0"
__all__ = ["ellipticity", "shear"]
