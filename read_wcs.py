import pandas as pd
import numpy as np
from astropy.wcs import WCS


def is_in(wcs,ra,dec,NAXIS1 = -1,NAXIS2 = -1):
    if NAXIS1 == -1:
        NAXIS1 = wcs._naxis1
    if NAXIS2 == -1:
        NAXIS2 = wcs._naxis2

    # Convert RA and Dec to pixel coordinates
    pixel_coords = wcs.wcs_world2pix(ra, dec, 0)
    x, y = pixel_coords
    # Check if the pixel coordinates are within the image bounds
    #print(x,y)
    if  (x > 0 and x <= NAXIS1 and y > 0 and y <= NAXIS2):
        return True
    else:
        return False

WCS_pd = pd.read_csv('sky_WCS.csv')
demo_WCS_info = WCS_pd.iloc[0] # read the first row as a demo
wcs = WCS(naxis=2)
wcs.wcs.crpix = [demo_WCS_info['CRPIX1'], demo_WCS_info['CRPIX2']]
wcs.wcs.cd = [[demo_WCS_info['CD1_1'], demo_WCS_info['CD1_2']],[demo_WCS_info['CD2_1'], demo_WCS_info['CD2_2']]]
wcs.wcs.crval = [demo_WCS_info['CRVAL1'], demo_WCS_info['CRVAL2']]
wcs.wcs.ctype = [demo_WCS_info['CTYPE1'], demo_WCS_info['CTYPE2']]
NAXIS1 = demo_WCS_info["NAXIS1"]
NAXIS2 = demo_WCS_info["NAXIS2"]
print(wcs)
ra = 0
dec = 0

print(f"ra = {ra},dec = {dec} in fov:",is_in(wcs,ra,dec,NAXIS1,NAXIS2))

ra = 0
dec = 90

print(f"ra = {ra},dec = {dec} in fov:",is_in(wcs,ra,dec,NAXIS1,NAXIS2))


    