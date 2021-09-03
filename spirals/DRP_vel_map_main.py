# IMPORT MODULES
#-------------------------------------------------------------------------------
import datetime
START = datetime.datetime.now()

import os.path, warnings

import numpy as np

from astropy.table import Table
import astropy.units as u

from file_io import add_columns, fillin_output_table

from DRP_rotation_curve import extract_data

from DRP_vel_map import fit_vel_map, estimate_total_mass

import sys
sys.path.insert(1, '/Users/kellydouglass/Documents/Research/Rotation_curves/RotationCurves/')
#sys.path.insert(1, '/home/kelly/Documents/RotationCurves/')
from mapSmoothness_functions import how_smooth

warnings.simplefilter('ignore', np.RankWarning)
warnings.simplefilter('ignore', RuntimeWarning)
################################################################################



################################################################################
# File format for saved images
#-------------------------------------------------------------------------------
IMAGE_FORMAT = 'eps'
################################################################################



################################################################################
# Fitting routine restrictions
#-------------------------------------------------------------------------------
# Maximum allowed map smoothness score
map_smoothness_max = 1.85

# Velocity function to use (options are 'BB' or 'tanh')
vel_function = 'BB'
################################################################################



################################################################################
# List of files (in "[MaNGA_plate]-[MaNGA_IFU]" format) to be ran through the
# individual galaxy version of this script.
# 
# If RUN_ALL_GALAXIES is set to True, then code will ignore what is in FILE_IDS.
#-------------------------------------------------------------------------------
FILE_IDS = ['9037-12703']

RUN_ALL_GALAXIES = False
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
    IMAGE_DIR = LOCAL_PATH + 'Images/DRP/'

    # Create directory if it does not already exist
    if not os.path.isdir( IMAGE_DIR):
        os.makedirs( IMAGE_DIR)
else:
    #IMAGE_DIR = None
    IMAGE_DIR = LOCAL_PATH + 'Images/DRP/'


MANGA_FOLDER = '/Users/kellydouglass/Documents/Research/data/SDSS/dr16/manga/spectro/'
NSA_FILENAME = '/Users/kellydouglass/Documents/Drexel/Research/Data/NSA/nsa_v1_0_1.fits'
'''
MANGA_FOLDER = '/home/kelly/Documents/Data/SDSS/dr16/manga/spectro/'
NSA_FILENAME = '/home/kelly/Documents/Data/NSA/nsa_v1_0_1.fits'
'''
VEL_MAP_FOLDER = MANGA_FOLDER + 'analysis/v2_4_3/2.2.1/HYB10-GAU-MILESHC/'
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
# Open the NSA file
#-------------------------------------------------------------------------------
NSA_table = Table.read( NSA_FILENAME, format='fits')


NSA_index = {}

for i in range(len(NSA_table)):
    NSA_ID = NSA_table['NSAID'][i]

    NSA_index[NSA_ID] = i
################################################################################




################################################################################
# Create a list of galaxy IDs for which to extract a rotation curve.
#-------------------------------------------------------------------------------
if RUN_ALL_GALAXIES:
    
    N_files = len(DRP_table)

    FILE_IDS = list(DRP_index.keys())

    DRP_table = add_columns(DRP_table)

else:

    N_files = len(FILE_IDS)
################################################################################



################################################################################
# Fit the rotation curve for the H-alpha velocity map for all of the galaxies in 
# the 'files' array.
#-------------------------------------------------------------------------------
num_masked_gal = 0 # Number of completely masked galaxies
num_not_smooth = 0 # Number of galaxies which do not have smooth velocity maps

for gal_ID in FILE_IDS:

    i_DRP = DRP_index[gal_ID]

    if DRP_table['mngtarg1'][i_DRP] > 0:
    
        ########################################################################
        # Extract the necessary data from the .fits files.
        #-----------------------------------------------------------------------
        Ha_vel, Ha_vel_ivar, Ha_vel_mask, r_band, r_band_ivar, Ha_flux, Ha_flux_ivar, Ha_flux_mask, Ha_sigma, Ha_sigma_ivar, Ha_sigma_mask = extract_data( VEL_MAP_FOLDER, gal_ID)

        print( gal_ID, "extracted")
        ########################################################################


        ########################################################################
        # Calculate degree of smoothness of velocity map
        #-----------------------------------------------------------------------
        map_smoothness = how_smooth( Ha_vel, Ha_vel_mask)

        map_smoothness_5sigma = how_smooth(Ha_vel, 
                                           np.logical_or(Ha_vel_mask > 0, 
                                                         np.abs(Ha_flux*np.sqrt(Ha_flux_ivar)) < 5))
        ########################################################################


        if map_smoothness <= map_smoothness_max or map_smoothness_5sigma <= map_smoothness_max:
            ####################################################################
            # Extract the necessary data from the DRP table.
            #-------------------------------------------------------------------
            axis_ratio = DRP_table['nsa_elpetro_ba'][i_DRP]
            phi_EofN_deg = DRP_table['nsa_elpetro_phi'][i_DRP]

            z = DRP_table['nsa_z'][i_DRP]

            NSA_ID = DRP_table['nsa_nsaid'][i_DRP]
            ####################################################################
            
            
            if axis_ratio > -9999:
                ################################################################
                # Extract rotation curve data for the .fits file in question and 
                # create an astropy Table containing said data.
                #---------------------------------------------------------------
                start = datetime.datetime.now()
                
                param_outputs, num_masked_gal, fit_flag = fit_vel_map(Ha_vel, 
                                                                      Ha_vel_ivar, 
                                                                      Ha_vel_mask, 
                                                                      Ha_sigma, 
                                                                      Ha_sigma_ivar, 
                                                                      Ha_sigma_mask, 
                                                                      Ha_flux, 
                                                                      Ha_flux_ivar, 
                                                                      Ha_flux_mask, 
                                                                      r_band, 
                                                                      r_band_ivar, 
                                                                      axis_ratio, 
                                                                      phi_EofN_deg, 
                                                                      z, gal_ID, 
                                                                      vel_function, 
                                                                      #IMAGE_DIR=IMAGE_DIR, 
                                                                      #IMAGE_FORMAT=IMAGE_FORMAT, 
                                                                      num_masked_gal=num_masked_gal)
                                                             
                fit_time = datetime.datetime.now() - start
                
                print(gal_ID, "velocity map fit", fit_time)
                ####################################################################

                ####################################################################
                # Extract the necessary data from the NSA table.
                #-------------------------------------------------------------------
                i_NSA = NSA_index[NSA_ID]

                R90 = NSA_table['ELPETRO_TH90_R'][i_NSA]
                ####################################################################

                if param_outputs is not None:
                    ################################################################
                    # Estimate the total mass within the galaxy
                    #---------------------------------------------------------------
                    mass_outputs = estimate_total_mass(param_outputs['v_max'], 
                                                       param_outputs['v_max_err'], 
                                                       R90, 
                                                       z, 
                                                       gal_ID)
                    ################################################################


                if RUN_ALL_GALAXIES:
                    ################################################################
                    # Write the best-fit values and calculated parameters to a text 
                    # file in ascii format.
                    #---------------------------------------------------------------
                    DRP_table = fillin_output_table(DRP_table, 
                                                    map_smoothness, 
                                                    i_DRP, 
                                                    col_name='smoothness_score')
                                                    
                    DRP_table = fillin_output_table(DRP_table, 
                                                    R90, 
                                                    i_DRP, 
                                                    col_name='nsa_elpetro_th90')
                    
                    if param_outputs is not None:
                        DRP_table = fillin_output_table(DRP_table, 
                                                        param_outputs, 
                                                        i_DRP)
                        DRP_table = fillin_output_table(DRP_table, 
                                                        mass_outputs, 
                                                        i_DRP)
                        DRP_table = fillin_output_table(DRP_table, 
                                                        fit_flag, 
                                                        i_DRP, 
                                                        col_name='fit_flag')

                    print(gal_ID, "written")
                    ################################################################

                else:
                    ################################################################
                    # Print output to terminal if not analyzing all galaxies
                    #---------------------------------------------------------------
                    print(DRP_table[['plateifu','nsa_z','nsa_elpetro_ba','nsa_elpetro_phi']][i_DRP])
                    print('Smoothness score:', map_smoothness)
                    print(param_outputs)
                    print(mass_outputs)
                    print('Fit flag:', fit_flag)
                    ################################################################
                
            else:
                print(gal_ID, 'is missing photometric measurements.')

        else:
            print("Galaxy's map is not smooth enough to fit.")

            num_not_smooth += 1

            if RUN_ALL_GALAXIES:
                DRP_table = fillin_output_table(DRP_table, map_smoothness, i_DRP, col_name='smoothness_score')
            else:
                print('Smoothness score:', map_smoothness)
                print('5-sigma smoothness score:', map_smoothness_5sigma)

    else:
        print(gal_ID, 'is not a galaxy target.')

    print("\n")
################################################################################



################################################################################
# Save the output_table
#-------------------------------------------------------------------------------
if RUN_ALL_GALAXIES:
    DRP_table.write('DRP_vel_map_results_' + fit_function + '_smooth_lt_' + str(map_smoothness_max) + '.txt', 
                    format='ascii.commented_header', 
                    overwrite=True)
################################################################################



################################################################################
# Print number of galaxies that were completely masked
#-------------------------------------------------------------------------------
if RUN_ALL_GALAXIES:
    print('There were', num_masked_gal, 'galaxies that were completely masked.')
    print('There were', num_not_smooth, 'galaxies without smooth velocity maps.')
################################################################################


################################################################################
# Clock the program's run time to check performance.
#-------------------------------------------------------------------------------
FINISH = datetime.datetime.now()
print("Runtime:", FINISH - START)
################################################################################

