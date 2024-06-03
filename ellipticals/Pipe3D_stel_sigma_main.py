import datetime
START = datetime.datetime.now()

import os.path, warnings
import sys

import numpy as np

from astropy.table import Table
import astropy.units as u

from IO_data import extract_data, add_cols, extract_Pipe3d_data
from elliptical_virial_mass import Pipe3D_sMass_weighted_star_sigma

# bluehive 
# sys.path.insert(1, '/scratch/nravi3/RotationCurves/')

# sys.path.insert(1, '/Users/nityaravi/Documents/GitHub/RotationCurves/')
# from mapSmoothness_functions import how_smooth


# FILE_IDS = ['11011-1901','10223-9101','9877-12704','11760-9101','11953-1902',
#             '9490-9101','11830-9101','8319-6101','8592-1901','8325-3704',
#             '9878-6103','8716-1902','8143-3704','12514-6101','10837-1901',
#             '9487-12705','8133-6103','9887-1901','12067-3704','8274-6103',
#             '8134-3702','8085-6101','9089-1902','9488-6104','11953-6102',
#             '11836-3702','12678-6104','11865-9101','12620-1901','8319-1902',
#             '9039-3703','12485-6104','9090-6103','8551-6103','11950-12703',
#             '11826-6104','11836-3703','8625-3702','8614-12703','12067-3701',
#             '8093-9101','8330-3703','11828-3701','11969-9101','8248-1902',
#             '8937-3702','10840-6102','8481-9101','9862-6101','8149-12704',
#             '8488-6101','11946-3704','9512-6104','11969-9102','8330-12705',
#             '9868-6104','8566-6102','8241-3701','8612-6103','10837-6101',
#             '12073-3703','9502-12701','9186-1902','11004-9102','11011-3701',
#             '9506-6104','11024-12702','10501-3701','8158-3702','8482-12704',
#             '10511-6104','8710-1901','8655-3703','11757-6102','9875-12705',
#             '10513-1902','8598-6101','7975-3703','8939-3701','8550-1901',
#             '7959-6104','8456-6103','11828-6104','10220-1902','8725-1902',
#             '9863-3704','8710-1901','8259-1901','9002-9101','11868-12702',
#             '11748-6102','8454-12705','8310-3701','11017-1902','9502-6103',
#             '11948-12705','10495-3703','11836-12703','10498-3703','8262-6102']

FILE_IDS = []

#IMAGE_FORMAT = 'png'

################################################################################
# Paths for Nitya Macbook
################################################################################

# MANGA_FOLDER = '/Users/nityaravi/Documents/Research/RotationCurves/data/manga/'
# IMAGE_DIR = MANGA_FOLDER + 'Ellipticals_Images/'
# MAP_FOLDER = MANGA_FOLDER + 'DR17/'
# PIPE3D_FOLDER = MANGA_FOLDER +'Pipe3D/'
# DRP_FILENAME = MANGA_FOLDER + 'output_files/DR17/CURRENT_MASTER_TABLE/Elliptical_StelVelDisp.fits'
# NSA_FILENAME = '/Users/nityaravi/Documents/Research/RotationCurves/data/nsa_v1_0_1.fits'

################################################################################
# Paths for Bluehive
################################################################################

MANGA_FOLDER = '/scratch/kdougla7/data/SDSS/dr17/manga/spectro/'
IMAGE_DIR = '/scratch/nravi3/ellipticals/'
MAP_FOLDER = MANGA_FOLDER + 'analysis/v3_1_1/3.1.0/HYB10-MILESHC-MASTARSSP/'
PIPE3D_FOLDER = MANGA_FOLDER + 'pipe3d/'
DRP_FILENAME = '/scratch/nravi3/ellipticals/Elliptical_StelVelDispDAPMeanSigma_Mvir_smoothness_lt_2_dipole_vflag_comoving.fits'
NSA_FILENAME = '/scratch/kdougla7/data/NSA/nsa_v1_0_1.fits'


################################################################################
################################################################################


CLEAR_COLS = True  # zeros out columns in table
RUN_ALL = True # if true, run an all galaxies
# map_smoothness_min = 2.0

START = datetime.datetime.now()


################################################################################
# Open the DRPall file
#-------------------------------------------------------------------------------
DRP_table = Table.read(DRP_FILENAME, format='fits')

DRP_index = {}

for i in range(len(DRP_table)):
    gal_ID = DRP_table['plateifu'][i]

    DRP_index[gal_ID] = i
################################################################################
if RUN_ALL:
    FILE_IDS = DRP_table['plateifu']

if CLEAR_COLS:
    DRP_table = add_cols(DRP_table, ['Pipe3D_star_sigma', 'Pipe3D_star_sigma_err'])

    
    
for gal_ID in FILE_IDS:

    i_DRP = DRP_index[gal_ID]

    if DRP_table['Mvir'][i_DRP] > 0:


        ########################################################################
        # Extract the necessary data from the .fits files.
        #-----------------------------------------------------------------------

        maps = extract_data(MAP_FOLDER, gal_ID, ['flux', 'star_sigma', 'Ha'])

        if maps is None:
            print('\n')
            continue

        pipe3d_maps = extract_Pipe3d_data(PIPE3D_FOLDER,gal_ID,['sMass', 'star_sigma'])

        print( gal_ID, "extracted")

        start = datetime.datetime.now()

        star_sigma, star_sigma_err = Pipe3D_sMass_weighted_star_sigma(maps['star_sigma'],
                                                                           maps['star_sigma_ivar'],
                                                                           maps['star_sigma_mask'],
                                                                           pipe3d_maps['star_sigma'],
                                                                           pipe3d_maps['star_sigma_err'],
                                                                           pipe3d_maps['sMass_density'],
                                                                           pipe3d_maps['sMass_density_err'])
        

        ########################################################################
        # write to table
        ########################################################################
        DRP_table['Pipe3D_star_sigma'][i_DRP] = star_sigma
        DRP_table['Pipe3D_star_sigma_err'][i_DRP] = star_sigma_err
        calc_time = datetime.datetime.now() - start

        print(gal_ID, '\n')
        print('time: ', calc_time, '\n')
        print('star_sigma: ', star_sigma, ', star_sigma_err: ', star_sigma_err)

        DRP_table.write(DRP_FILENAME[:-5] + '_Pipe3D_sigma.fits', overwrite=True, format='fits')

DRP_table.write(DRP_FILENAME[:-5] + '_Pipe3D_sigma.fits', overwrite=True, format='fits')

TIME = datetime.datetime.now() - START
print('total time: ', TIME)