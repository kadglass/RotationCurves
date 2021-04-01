'''
Main script to extract and fit the stellar (disk) mass rotation curve in disk 
galaxies.
'''


from time import time
START = time()

import os

import numpy as np

from astropy.table import Table

from file_io import add_disk_columns, fillin_output_table

from DRP_rotation_curve import extract_data, extract_Pipe3d_data

from disk_mass import calc_mass_curve, fit_mass_curve

import sys
sys.path.insert(1, '/Users/kellydouglass/Documents/Research/Rotation_curves/Yifan_Zhang/RotationCurve/')
from rotation_curve_functions import disk_mass




################################################################################
# File format for saved images
#-------------------------------------------------------------------------------
IMAGE_FORMAT = 'eps'
################################################################################



################################################################################
# List of files (in "[MaNGA plate]-[MaNGA IFU]" format) to be run through the 
# individual galaxy version of this script.
#
# If RUN_ALL_GALAXIES is set to True, then the code will ignore what is in 
# FILE_IDS
#-------------------------------------------------------------------------------
FILE_IDS = ['7443-12705', '9678-12701']

RUN_ALL_GALAXIES = True
################################################################################



################################################################################
# 'LOCAL_PATH' should be updated depending on the file structure (e.g. if
# working in bluehive).  It is set to 'os.path.dirname(__file__)' when working 
# on a local system.
#
# ATTN: 'MANGA_FOLDER' must be manually altered according to the data release
#       being analyzed.
#-------------------------------------------------------------------------------
LOCAL_PATH = os.path.dirname(__file__)
if LOCAL_PATH == '':
    LOCAL_PATH = './'

if RUN_ALL_GALAXIES:
    IMAGE_DIR = LOCAL_PATH + 'Images/DRP-Pipe3d/'

    # Create directory if it does not already exist
    if not os.path.isdir( IMAGE_DIR):
        os.makedirs( IMAGE_DIR)
else:
    IMAGE_DIR = None
    #IMAGE_DIR = LOCAL_PATH + 'Images/DRP-Pipe3d/'


MANGA_FOLDER = '/Users/kellydouglass/Documents/Research/data/SDSS/dr16/manga/spectro/'
SDSS_FOLDER = '/Users/kellydouglass/Documents/Research/data/SDSS/'
'''
MANGA_FOLDER = '/home/kelly/Documents/Data/SDSS/dr16/manga/spectro/'
SDSS_FOLDER = '/home/kelly/Documents/Data/SDSS/'
'''
MASS_MAP_FOLDER = SDSS_FOLDER + 'dr15/manga/spectro/pipe3d/v2_4_3/2.4.3/'
VEL_MAP_FOLDER = SDSS_FOLDER + 'dr16/manga/spectro/analysis/v2_4_3/2.2.1/HYB10-GAU-MILESHC/'

MASS_CURVE_MASTER_FOLDER = LOCAL_PATH + 'Pipe3d-mass_curve_data_files/'
if not os.path.isdir(MASS_CURVE_MASTER_FOLDER):
    os.makedirs(MASS_CURVE_MASTER_FOLDER)

GALAXIES_FILENAME = 'DRP_vel_map_results_BB_smooth_lt_1p85_v2.fits'
DRP_FILENAME = MANGA_FOLDER + 'redux/v2_4_3/drpall-v2_4_3.fits'
################################################################################



################################################################################
# Open the DRPall file
#-------------------------------------------------------------------------------
DRP_table = Table.read( DRP_FILENAME, format='fits')


DRP_index = {}

for i in range(len(DRP_table)):
    gal_ID = DRP_table['plateifu'][i]

    DRP_index[gal_ID] = i
################################################################################



################################################################################
# Open the galaxies file
#-------------------------------------------------------------------------------
#galaxies_table = Table.read( GALAXIES_FILENAME, format='ascii.ecsv')
galaxies_table = Table.read(GALAXIES_FILENAME, format='fits')


galaxies_index = {}

for i in range(len(galaxies_table)):
    gal_ID = galaxies_table['plateifu'][i]

    galaxies_index[gal_ID] = i
################################################################################



################################################################################
# Create a list of galaxy IDs for which to fit the mass rotation curve.
#-------------------------------------------------------------------------------
if RUN_ALL_GALAXIES:
    
    N_files = len(galaxies_table)

    FILE_IDS = list(galaxies_index.keys())

    galaxies_table = add_disk_columns(galaxies_table)

else:

    N_files = len(FILE_IDS)
################################################################################




################################################################################
# Fit the rotation curve for the stellar mass density map for all of the 
# galaxies in the 'files' array.
#-------------------------------------------------------------------------------
for gal_ID in FILE_IDS:
    
    ############################################################################
    # Extract the necessary data from the .fits files.
    #---------------------------------------------------------------------------
    _,_, map_mask, r_band,_ = extract_data(VEL_MAP_FOLDER, gal_ID)
    sMass_density = extract_Pipe3d_data(MASS_MAP_FOLDER, gal_ID)

    if map_mask is None or r_band is None or sMass_density is None:
        print('\n')
        continue

    print( gal_ID, "extracted")
    ############################################################################


    i_gal = galaxies_index[gal_ID]
    i_DRP = DRP_index[gal_ID]


    ########################################################################
    # Extract the necessary data from the galaxies table.
    #-----------------------------------------------------------------------
    if np.isfinite(galaxies_table['phi'][i_gal]):

        axis_ratio = galaxies_table['ba'][i_gal]
        axis_ratio_err = galaxies_table['ba_err'][i_gal]

        phi_EofN_deg = galaxies_table['phi'][i_gal]
        phi_EofN_deg_err = galaxies_table['phi_err'][i_gal]

        center_x = galaxies_table['x0'][i_gal]
        center_x_err = galaxies_table['x0_err'][i_gal]

        center_y = galaxies_table['y0'][i_gal]
        center_y_err = galaxies_table['y0_err'][i_gal]

    else:

        axis_ratio = DRP_table['nsa_elpetro_ba'][i_DRP]
        axis_ratio_err = np.NaN

        phi_EofN_deg = DRP_table['nsa_elpetro_phi'][i_DRP]
        phi_EofN_deg_err = np.NaN

        center_x = None
        center_x_err = None

        center_y = None
        center_y_err = None

    z = galaxies_table['nsa_z'][i_gal]

    R90 = galaxies_table['nsa_elpetro_th90'][i_gal]
    ########################################################################
    
    
    ########################################################################
    # Extract rotation curve data for the .fits file in question and create 
    # an astropy Table containing said data.
    #-----------------------------------------------------------------------
    start = time()
    
    mass_data_table = calc_mass_curve(sMass_density, 
                                      r_band, 
                                      map_mask, 
                                      center_x,
                                      center_y,
                                      axis_ratio, 
                                      phi_EofN_deg, 
                                      z, 
                                      gal_ID, 
                                      IMAGE_DIR=IMAGE_DIR, 
                                      IMAGE_FORMAT=IMAGE_FORMAT)
                                                 
    extract_time = time() - start
    
    print(gal_ID, "mass curve calculated", extract_time)
    ########################################################################


    if len(mass_data_table) > 3:
        ####################################################################
        # Fit the stellar mass rotation curve to the disk velocity function.
        #-------------------------------------------------------------------
        start = time()

        param_outputs = fit_mass_curve(mass_data_table, 
                                       gal_ID, 
                                       IMAGE_DIR=IMAGE_DIR,
                                       IMAGE_FORMAT=IMAGE_FORMAT
                                       )

        fit_time = time() - start

        print(gal_ID, 'mass curve fit', fit_time)
        ####################################################################
        

        ####################################################################
        # Estimate the total disk mass within the galaxy
        #-------------------------------------------------------------------
        if param_outputs is not None:
            M90_disk, M90_disk_err = disk_mass(param_outputs, R90)
        ####################################################################


        if RUN_ALL_GALAXIES:

            ################################################################
            # Write the extracted mass curve to a text file in ascii format.
            #---------------------------------------------------------------
            mass_data_table.write(MASS_CURVE_MASTER_FOLDER + gal_ID + '.txt', 
                                  format='ascii.commented_header', 
                                  overwrite=True)
            ################################################################


            if param_outputs is not None:
                ############################################################
                # Write the best-fit values and calculated parameters to a 
                # text  file in ascii format.
                #-----------------------------------------------------------
                galaxies_table = fillin_output_table(galaxies_table, 
                                                     param_outputs, 
                                                     i_DRP)
                galaxies_table = fillin_output_table(galaxies_table, 
                                                     M90_disk, 
                                                     i_DRP, 
                                                     col_name='M90_disk')
                galaxies_table = fillin_output_table(galaxies_table, 
                                                     M90_disk_err, 
                                                     i_DRP, 
                                                     col_name='M90_disk_err')
                ############################################################

            print(gal_ID, "written")

        else:
            ################################################################
            # Print output to terminal if not analyzing all galaxies
            #---------------------------------------------------------------
            print(param_outputs)

            if param_outputs is not None:
                print('M90_disk:', M90_disk, '+/-', M90_disk_err)
            ################################################################


    print("\n")
################################################################################



################################################################################
# Save the output_table
#-------------------------------------------------------------------------------
if RUN_ALL_GALAXIES:

    galaxies_filename, extension = GALAXIES_FILENAME.split('.')

    galaxies_table.write(galaxies_filename + '_diskFit.fits', 
                         format='fits', #'ascii.commented_header', 
                         overwrite=True)
################################################################################



################################################################################
# Clock the program's run time to check performance.
#-------------------------------------------------------------------------------
FINISH = time()
print("Runtime:", FINISH - START)
################################################################################






