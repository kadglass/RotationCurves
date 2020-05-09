#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 20 2018
@author: Jacob A. Smith
@version: 1.2

Main script file for dark_matter_mass_vX_X.
"""
import datetime
START = datetime.datetime.now()

import glob, os.path
import numpy as np

from astropy.table import QTable

import astropy.units as u

from dark_matter_mass import fit_rot_curve, estimate_dark_matter


###############################################################################
# Number of times to try to fit the data within 'scipy.optimize.curve_fit().'
#------------------------------------------------------------------------------
TRY_N = 100000
###############################################################################


###############################################################################
# Boolean variables to specify if the script is being run in Bluehive and if
#    all galaxies are to be ran through the script and saved to the
#    'master_file' or not.
#------------------------------------------------------------------------------
WORKING_IN_BLUEHIVE = False
RUN_ALL_GALAXIES = False
###############################################################################


###############################################################################
# List of files (in "[MaNGA_plate]-[MaNGA_fiberID]" format) to be ran through
#    the individual galaxy version of this script.
#------------------------------------------------------------------------------
FILE_IDS = ['7443-6102']
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
    import matplotlib
    matplotlib.use('Agg')

    LOCAL_PATH = '/home/jsm171'
    SCRATCH_PATH = '/scratch/jsm171'

    IMAGE_DIR = SCRATCH_PATH + '/images'
    ROT_CURVE_MASTER_FOLDER = SCRATCH_PATH + '/rot_curve_data_files'

else:
    LOCAL_PATH = os.path.dirname(__file__)
    if LOCAL_PATH == '':
        LOCAL_PATH = './'

    IMAGE_DIR = LOCAL_PATH + 'Images/DRP/'
    ROT_CURVE_MASTER_FOLDER = LOCAL_PATH + 'DRP-rot_curve_data_files/'


MASTER_FILENAME = LOCAL_PATH + 'DRPall-master_file.txt'

# Create output directories if they do not already exist
if not os.path.isdir( IMAGE_DIR):
    os.makedirs( IMAGE_DIR)
if not os.path.isdir( ROT_CURVE_MASTER_FOLDER):
    os.makedirs( ROT_CURVE_MASTER_FOLDER)
###############################################################################


if RUN_ALL_GALAXIES:
    ###########################################################################
    # Read in the 'master_file' and extract its length for use later in the
    # program.
    #--------------------------------------------------------------------------
    master_table = QTable.read( MASTER_FILENAME, format='ascii.ecsv')
    N_galaxies = len( master_table)
    ###########################################################################


    ###########################################################################
    # Master arrays initialized to contain memory-holding values.
    #--------------------------------------------------------------------------
    master_table['center_flux'] = -1. * (u.erg / (u.cm * u.cm * u.s))
    master_table['center_flux_error'] = -1. * (u.erg / (u.cm * u.cm * u.s))
    master_table['frac_masked_spaxels'] = -1.

    master_table['avg_v_max'] = -1. * (u.km / u.s)
    master_table['avg_r_turn'] = -1. * (u.kpc)
    master_table['avg_alpha'] = -1. #* np.ones( N_galaxies)

    master_table['avg_v_max_sigma'] = -1. * (u.km / u.s)
    master_table['avg_r_turn_sigma'] = -1. * (u.kpc)
    master_table['avg_alpha_sigma'] = -1. #* np.ones( N_galaxies)
    master_table['avg_chi_square_rot'] = -1. #* np.ones( N_galaxies)
    master_table['avg_chi_square_ndf'] = -1.

    master_table['pos_v_max'] = -1. * (u.km / u.s)
    master_table['pos_r_turn'] = -1. * (u.kpc)
    master_table['pos_alpha'] = -1. #* np.ones( N_galaxies)

    master_table['pos_v_max_sigma'] = -1. * (u.km / u.s)
    master_table['pos_r_turn_sigma'] = -1. * (u.kpc)
    master_table['pos_alpha_sigma'] = -1. #* np.ones( N_galaxies)
    master_table['pos_chi_square_rot'] = -1. #* np.ones( N_galaxies)
    master_table['pos_chi_square_ndf'] = -1.

    master_table['neg_v_max'] = -1. * (u.km / u.s)
    master_table['neg_r_turn'] = -1. * (u.kpc)
    master_table['neg_alpha'] = -1. #* np.ones( N_galaxies)

    master_table['neg_v_max_sigma'] = -1. * (u.km / u.s)
    master_table['neg_r_turn_sigma'] = -1. * (u.kpc)
    master_table['neg_alpha_sigma'] = -1. #* np.ones( N_galaxies)
    master_table['neg_chi_square_rot'] = -1. #* np.ones( N_galaxies)
    master_table['neg_chi_square_ndf'] = -1.


    master_table['Mtot'] = -1. * u.M_sun
    master_table['Mtot_error'] = -1. * u.M_sun
    master_table['Mdark'] = -1. * u.M_sun
    master_table['Mdark_error'] = -1. * u.M_sun
    master_table['Mstar'] = -1. * u.M_sun
    master_table['Mdark_Mstar_ratio'] = -1. #* np.ones( N_galaxies)
    master_table['Mdark_Mstar_ratio_error'] = -1. #* np.ones( N_galaxies)

    master_table['curve_used'] = '    '
    master_table['points_cut'] = 0 #np.zeros( N_galaxies)
    ###########################################################################
    

    ###########################################################################
    # For all of the galaxies in the 'master_table'
    #--------------------------------------------------------------------------
    for i in range( N_galaxies):

        #######################################################################
        # Build the file names for the rotation curve and galaxy statistic data
        #    from the 'MaNGA_plate' and 'MaGNA_fiberID' columns of the
        #    'master_file.'
        #----------------------------------------------------------------------
        plate = master_table['MaNGA_plate'][i]
        IFU = master_table['MaNGA_IFU'][i]
        gal_ID = str(plate) + '-' + str(IFU)
        print("gal_ID MAIN:", gal_ID)

        rot_curve_filename = ROT_CURVE_MASTER_FOLDER + gal_ID + '_rot_curve_data.txt'
        gal_stat_filename = ROT_CURVE_MASTER_FOLDER + gal_ID + '_gal_stat_data.txt'
        #######################################################################


        #######################################################################
        # Set of functions to run the set of rotation curve and of galaxy
        # statistic files through.
        #----------------------------------------------------------------------
        param_outputs = fit_rot_curve( rot_curve_filename, gal_stat_filename,
                                       TRY_N)

        mass_outputs = estimate_dark_matter( param_outputs, rot_curve_filename, 
                                                            gal_stat_filename)

        for col_name in mass_outputs:
            master_table[ col_name][i] = mass_outputs[ col_name]
        #######################################################################


    ###########################################################################
    # Save the 'master_table.'
    #--------------------------------------------------------------------------
    master_table.write( MASTER_FILENAME[:-4] + '_30.txt', format='ascii.ecsv', 
                        overwrite=True)
    ###########################################################################


else:
    ###########################################################################
    # For the galaxies contained within the 'FILE_IDS' array...
    #--------------------------------------------------------------------------
    for i in range( len( FILE_IDS)):
        print("gal_ID MAIN:", FILE_IDS[i])

        #######################################################################
        # Build the file names for the rotation curve and galaxy statistic data
        #    from the 'FILE_IDS' array.
        #----------------------------------------------------------------------
        rot_curve_filename = ROT_CURVE_MASTER_FOLDER + '/' + FILE_IDS[i] \
                                            + '_rot_curve_data.txt'
        gal_stat_filename = ROT_CURVE_MASTER_FOLDER + '/' + FILE_IDS[i] \
                                            + '_gal_stat_data.txt'
        #######################################################################


        #######################################################################
        # Set of functions to run the set of rotation curve and of galaxy
        #    statistic files through.
        #----------------------------------------------------------------------
        param_outputs = fit_rot_curve( rot_curve_filename, gal_stat_filename,
                                       TRY_N)

        mass_outputs = estimate_dark_matter( param_outputs, rot_curve_filename, 
                                             gal_stat_filename)
        #######################################################################


        #######################################################################
        # Print the'mass_outputs' dictionary.
        #----------------------------------------------------------------------
        print("mass_outputs:", mass_outputs)
        #######################################################################


###############################################################################
# Clock the program's run time to check performance.
#------------------------------------------------------------------------------
FINISH = datetime.datetime.now()
print("Runtime (COMPLETED):", FINISH - START)
###############################################################################