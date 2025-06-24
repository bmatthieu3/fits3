import numpy as np
from astropy.io import fits
import os

data = np.load("cosmo512.npy")
data_f32 = data.astype("float32")
hdu = fits.PrimaryHDU(data=data_f32)
hdu.writeto("cosmo512.fits", overwrite=True)