from galsim.nfw_halo import NFWHalo
import numpy as np

class NFWHalo_theory(NFWHalo):
    """A child class of galsim.nfw_halo.NFWhalo
    which just adds a  convenience function for 
    getting the tangential shear as a function
    of angular separation. See the docstring for 
    that class here 
    https://github.com/GalSim-developers/GalSim/blob/releases/2.8/galsim/nfw_halo.py#L112"""
    def get_tangential_shear(self, z_source, sep_arcsec):
        """
        Return the tangential shear for sources at 
        z_source, for angular separations sep_arcsec in
        arcseconds
        
        Parameters
        ----------
        z_source: source redshift (float)
        sep_arcsec: separation in arcsec (numpy float array)

        Returns
        ---------
        tangential shear as numpy array
        """
        #get separation in units of the scale radius
        rs = sep_arcsec/self.rs_arcsec
        #get source redshift-dependent amplitude 
        ks = self._NFWHalo__ks(z_source)
        #call existing but private method to compute the shear
        return np.array([self._NFWHalo__gamma(r, ks) for r in rs])