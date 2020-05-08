#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 09 2018
@author: Jacob A. Smith

Main script file for 'rotation_curve_vX_X.'
"""
import datetime
START = datetime.datetime.now()

import os.path, warnings

import numpy as np

from astropy.table import Table
import astropy.units as u

from Pipe3D_rotation_curve import extract_data, \
                                  match_to_NSA, \
                                  calc_rot_curve, \
                                  write_rot_curve, \
                                  write_master_file

warnings.simplefilter('ignore', np.RankWarning)

###############################################################################
# File format for saved images
#------------------------------------------------------------------------------
IMAGE_FORMAT = 'eps'
###############################################################################

###############################################################################
# Boolean variable to specify if the script is being run in Bluehive.
#------------------------------------------------------------------------------
WORKING_IN_BLUEHIVE = False
###############################################################################

###############################################################################
# List of files (in "[MaNGA_plate]-[MaNGA_fiberID]" format) to be ran through
#    the individual galaxy version of this script.
#------------------------------------------------------------------------------
RUN_ALL_GALAXIES = True

FILE_IDS = ['7443-12705']
###############################################################################


###############################################################################
# 'LOCAL_PATH' should be updated depending on the file structure (e.g. if
#    working in bluehive). It is set to 'os.path.dirname(__file__)' when
#    working on a local system.
#
# In addition, 'LOCAL_PATH' is altered and 'SCRATCH_PATH' is added if
#    'WORKING_IN_BLUEHIVE' is set to True. This is done because of how the data
#    folders are kept separate from the python script files in bluehive. For
#    BlueHive to run, images cannot be generated with $DISPLAY keys; therefore,
#    'matplotlib' is imported and 'Agg' is used. This must be done before
#    'matplotlib.pyplot' is imported.
#
# This block can be altered if desired, but the conditional below is tailored
#    for use with bluehive.
#
# ATTN: 'MANGA_FOLDER' must be manually altered according to the data release
#       being ran.
#------------------------------------------------------------------------------
if WORKING_IN_BLUEHIVE:
    LOCAL_PATH = '/home/jsm171/'
    SCRATCH_PATH = '/scratch/jsm171/'

    IMAGE_DIR = SCRATCH_PATH + 'images/'
    MANGA_FOLDER = SCRATCH_PATH + 'manga_files/dr15/'
    ROT_CURVE_MASTER_FOLDER = SCRATCH_PATH + 'rot_curve_data_files/'
    MASTER_FILENAME = 'master_file.txt'

else:
    LOCAL_PATH = os.path.dirname(__file__)
    if LOCAL_PATH == '':
        LOCAL_PATH = './'

    if RUN_ALL_GALAXIES:
        IMAGE_DIR = LOCAL_PATH + 'Images/Pipe3D/'
    else:
        IMAGE_DIR = None
    
    MANGA_FOLDER = LOCAL_PATH + '../data/MaNGA/MaNGA_DR15/pipe3d/'
    ROT_CURVE_MASTER_FOLDER = LOCAL_PATH + 'rot_curve_data_files/'
    MASTER_FILENAME = 'master_file_vflag_10.txt'


# Create output directory if it does not already exist
if (IMAGE_DIR is not None) and (not os.path.isdir( IMAGE_DIR)):
    os.makedirs( IMAGE_DIR)
if not os.path.isdir( ROT_CURVE_MASTER_FOLDER):
    os.makedirs( ROT_CURVE_MASTER_FOLDER)
###############################################################################



###############################################################################
# Open the master file
#------------------------------------------------------------------------------
master_table = Table.read( MASTER_FILENAME, format='ascii.ecsv')


master_index = {}

for i in range(len(master_table)):
    plate = master_table['MaNGA_plate'][i]
    IFU = master_table['MaNGA_fiberID'][i]

    master_index[str(plate) + '-' + str(IFU)] = i
###############################################################################



################################################################################
# Create list of galaxy IDs for which to extract a rotation curve.
#-------------------------------------------------------------------------------
if RUN_ALL_GALAXIES:

    N_files = len(master_table)

    FILE_IDS = list(master_index.keys())

else:

    N_files = len(FILE_IDS)
################################################################################



###############################################################################
# This for loop runs through the necessary calculations to calculate and write
# the rotation curve for all of the galaxies in the 'files' array.
#------------------------------------------------------------------------------
num_masked_gal = 0 # Number of completely masked galaxies

for gal_ID in FILE_IDS:
    
    ###########################################################################
    # gal_id is a simplified string that identifies each galaxy that is run
    # through the algorithm.  The gal_id name scheme is [PLATE]-[FIBER ID].
    #--------------------------------------------------------------------------
    manga_plate, manga_IFU = gal_ID.split('-')

    file_name = MANGA_FOLDER + manga_plate + '/manga-' + gal_ID + '.Pipe3D.cube.fits.gz'
    ###########################################################################


    ###########################################################################
    # Extract the necessary data from the .fits file.
    #--------------------------------------------------------------------------
    galaxy_target, good_galaxy, Ha_vel, Ha_vel_error, v_band, v_band_err, \
    sMass_density, gal_ra, gal_dec = extract_data( file_name)
    print( gal_ID, " EXTRACTED")
    ###########################################################################
    

    ############################################################################
    # Extract data from the master table.
    #---------------------------------------------------------------------------
    i_master = master_index[gal_ID]

    axis_ratio = master_table['NSA_ba'][ i_master]
    phi_EofN_deg = master_table['NSA_phi'][ i_master] * u.degree
    z = master_table['NSA_redshift'][ i_master]
    ############################################################################
    
    
    if galaxy_target and good_galaxy:
        ########################################################################
        # Extract rotation curve data for the .fits file in question and create 
        # an astropy Table containing said data.
        #-----------------------------------------------------------------------
        rot_data_table, gal_stat_table, num_masked_gal = calc_rot_curve( Ha_vel, 
                                                                         Ha_vel_error, 
                                                                         v_band, 
                                                                         v_band_err, 
                                                                         sMass_density, 
                                                                         axis_ratio, 
                                                                         phi_EofN_deg, 
                                                                         z, gal_ID, 
                                                                         IMAGE_DIR=IMAGE_DIR, 
                                                                         num_masked_gal=num_masked_gal)
        print(gal_ID, " ROT CURVE CALCULATED")
        ########################################################################
    
    
        ########################################################################
        # Write the rotation curve data to a text file in ascii format.
        #
        # IMPORTANT: rot_curve_main.py writes the data files into the default
        #            folder 'rot_curve_data_files'. It also saves the file with 
        #            the default extension '_rot_curve_data'.
        #-----------------------------------------------------------------------
        write_rot_curve( rot_data_table, gal_stat_table, gal_ID, ROT_CURVE_MASTER_FOLDER)

        print(gal_ID, " WRITTEN")
        ########################################################################

    print("\n")
###############################################################################


###############################################################################
# Print number of galaxies that were completely masked
#------------------------------------------------------------------------------
if RUN_ALL_GALAXIES:
    print('There were', num_masked_gal, 'galaxies that were completely masked.')
###############################################################################


###############################################################################
# Clock the program's run time to check performance.
#------------------------------------------------------------------------------
FINISH = datetime.datetime.now()
print("Runtime:", FINISH - START)
###############################################################################

