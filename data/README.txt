The following data files must be present in this directory for the code to run:

  galaxy_stamps.fits
      FITS file containing postage-stamp images of all source galaxies.
      Read by question1.ipynb via a6cw.ellipticity.measure_all_ellipticities().

  halo_positions.txt
      Two-column whitespace-separated text file of halo (x, y) positions
      in arcseconds. Read by question2.ipynb via a6cw.shear.load_positions().

  source_positions.txt
      Two-column whitespace-separated text file of source galaxy (x, y)
      positions in arcseconds. Read by question2.ipynb via
      a6cw.shear.load_positions().
