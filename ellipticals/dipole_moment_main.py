import datetime
START = datetime.datetime.now()

import os.path, warnings
import sys

import numpy as np

from astropy.table import Table
import astropy.units as u

from IO_data import extract_data, add_cols, extract_Pipe3d_data
from elliptical_virial_mass import calculate_virial_mass, calculate_dipole_moment

# bluehive 
# sys.path.insert(1, '/scratch/nravi3/RotationCurves/')




MANGA_FOLDER = '/scratch/kdougla7/data/SDSS/dr17/manga/spectro/'
IMAGE_DIR = '/scratch/nravi3/ellipticals/'
MAP_FOLDER = MANGA_FOLDER + 'analysis/v3_1_1/3.1.0/HYB10-MILESHC-MASTARSSP/'
PIPE3D_FOLDER = MANGA_FOLDER + 'pipe3d/'
DRP_FILENAME = '/scratch/nravi3/ellipticals/Elliptical_StelVelDispDAPMeanSigma_Mvir_smoothness_lt_2_dipole.fits'
NSA_FILENAME = '/scratch/kdougla7/data/NSA/nsa_v1_0_1.fits'
SAVE_FILENAME = '/scratch/nravi3/ellipticals/Elliptical_StelVelDispDAPMeanSigma_Mvir_smoothness_lt_2_dipole.fits'


START = datetime.datetime.now()

DRP_table = Table.read(DRP_FILENAME)
DRP_table = add_cols(DRP_table, ['dipole_moment'])

for i in range(len(DRP_table)):
    gal_ID = DRP_table['plateifu'][i]

    if DRP_table['mngtarg1'][i] > 0:

            ########################################################################
            # Extract the necessary data from the .fits files.
            #-----------------------------------------------------------------------

            maps = extract_data(MAP_FOLDER, gal_ID, ['flux', 'Ha'])

            if maps is None:
                print('No data for ', gal_ID, '\n')
                
                continue

            p = calculate_dipole_moment(maps['Ha_vel'],
                                        maps['Ha_vel_mask'],
                                        maps['Ha_flux'],
                                        maps['Ha_flux_ivar'],
                                        maps['mflux'])
            
            DRP_table['dipole_moment'][i] = p
            print(gal_ID, ' processed')

    else:
         print(gal_ID, ' not a galaxy target', flush=True)

    if i%10 == 0:
         DRP_table.write(SAVE_FILENAME, format='fits', overwrite=True)
         print(i)

        
DRP_table.write(SAVE_FILENAME, format='fits', overwrite=True)

FINISH = datetime.datetime.now()
print("Runtime:", FINISH - START, flush=True)