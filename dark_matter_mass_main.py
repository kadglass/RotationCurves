#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 20 2018
@author: Jacob A. Smith
@version: 1.1

Main script file for dark_matter_mass_vX_X.
"""
import datetime
START = datetime.datetime.now()

import glob, os.path
import numpy as np
from astropy.table import Table

###############################################################################
# File format for saved images
#------------------------------------------------------------------------------
IMAGE_FORMAT = 'eps'
###############################################################################

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
WORKING_IN_BLUEHIVE = True
RUN_ALL_GALAXIES = True
###############################################################################

###############################################################################
# List of files (in "[MaNGA_plate]-[MaNGA_fiberID]" format) to be ran through
#    the individual galaxy version of this script.
#------------------------------------------------------------------------------
FILE_IDS = ['7495-9101']
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

    CROSS_REF_FILE_NAMES = [
        SCRATCH_PATH + '/kias1033_5_P-MJD-F_MPAJHU_ZdustOS_stellarMass_BPT' + \
                      '_SFR_NSA_correctVflag.txt',
        SCRATCH_PATH + '/updated_vflag_data_files' + \
                     '/vflag_not_classified_RECLASS.txt',
        SCRATCH_PATH + '/updated_vflag_data_files' + \
                     '/vflag_not_found_RECLASS.txt',
        SCRATCH_PATH + '/updated_vflag_data_files' + \
                     '/void_reclassification_RECLASS.txt',
        SCRATCH_PATH + '/updated_vflag_data_files' + \
                     '/wall_reclassification_RECLASS.txt']

else:
    LOCAL_PATH = os.path.dirname(__file__)
    if LOCAL_PATH == '':
        LOCAL_PATH = '.'

    IMAGE_DIR = LOCAL_PATH + '/images'
    ROT_CURVE_MASTER_FOLDER = LOCAL_PATH + '/rot_curve_data_files'

    CROSS_REF_FILE_NAMES = [
        LOCAL_PATH + '/kias1033_5_P-MJD-F_MPAJHU_ZdustOS_stellarMass_BPT' + \
                      '_SFR_NSA_correctVflag.txt',
        LOCAL_PATH + '/updated_vflag_data_files' + \
                     '/vflag_not_classified_RECLASS.txt',
        LOCAL_PATH + '/updated_vflag_data_files' + \
                     '/vflag_not_found_RECLASS.txt',
        LOCAL_PATH + '/updated_vflag_data_files' + \
                     '/void_reclassification_RECLASS.txt',
        LOCAL_PATH + '/updated_vflag_data_files' + \
                     '/wall_reclassification_RECLASS.txt']

MASTER_FILE_NAME = LOCAL_PATH + '/master_file.txt'

# Create output directories if they do not already exist
if not os.path.isdir( IMAGE_DIR):
    os.makedirs( IMAGE_DIR)
if not os.path.isdir( ROT_CURVE_MASTER_FOLDER):
    os.makedirs( ROT_CURVE_MASTER_FOLDER)
###############################################################################


###############################################################################
# Import functions from 'dark_matter_mass_vX_X.'
#------------------------------------------------------------------------------
from dark_matter_mass_v1_1 import initialize_master_table, \
                                pull_matched_data, \
                                fit_rot_curve, \
                                estimate_dark_matter, \
                                plot_mass_ratios, \
                                analyze_rot_curve_discrep, \
                                analyze_chi_square
###############################################################################

if RUN_ALL_GALAXIES:
    ###########################################################################
    # Read in the 'master_file' and extract its length for use later in the
    #    program.
    #--------------------------------------------------------------------------
    master_table = Table.read( MASTER_FILE_NAME, format='ascii.ecsv')
    N_galaxies = len( master_table)
    ###########################################################################


    ###########################################################################
    # Master arrays initialized to contain memory-holding values.
    #--------------------------------------------------------------------------
    master_table['center_flux'] = -1 * np.ones( N_galaxies)
    master_table['center_flux_error'] = -1 * np.ones( N_galaxies)
    master_table['sMass_processed'] = -1 * np.ones( N_galaxies)


    master_table['v_max_best'] = -1 * np.ones( N_galaxies)
    master_table['r_turn_best'] = -1 * np.ones( N_galaxies)
    master_table['alpha_best'] = -1 * np.ones( N_galaxies)

    master_table['v_max_sigma'] = -1 * np.ones( N_galaxies)
    master_table['r_turn_sigma'] = -1 * np.ones( N_galaxies)
    master_table['alpha_sigma'] = -1 * np.ones( N_galaxies)
    master_table['chi_square_rot'] = -1 * np.ones( N_galaxies)

    master_table['pos_v_max_best'] = -1 * np.ones( N_galaxies)
    master_table['pos_r_turn_best'] = -1 * np.ones( N_galaxies)
    master_table['pos_alpha_best'] = -1 * np.ones( N_galaxies)

    master_table['pos_v_max_sigma'] = -1 * np.ones( N_galaxies)
    master_table['pos_r_turn_sigma'] = -1 * np.ones( N_galaxies)
    master_table['pos_alpha_sigma'] = -1 * np.ones( N_galaxies)
    master_table['pos_chi_square_rot'] = -1 * np.ones( N_galaxies)

    master_table['neg_v_max_best'] = -1 * np.ones( N_galaxies)
    master_table['neg_r_turn_best'] = -1 * np.ones( N_galaxies)
    master_table['neg_alpha_best'] = -1 * np.ones( N_galaxies)

    master_table['neg_v_max_sigma'] = -1 * np.ones( N_galaxies)
    master_table['neg_r_turn_sigma'] = -1 * np.ones( N_galaxies)
    master_table['neg_alpha_sigma'] = -1 * np.ones( N_galaxies)
    master_table['neg_chi_square_rot'] = -1 * np.ones( N_galaxies)


    master_table['total_mass'] = -1 * np.ones( N_galaxies)
    master_table['total_mass_error'] = -1 * np.ones( N_galaxies)
    master_table['dmMass'] = np.zeros( N_galaxies)
    master_table['dmMass_error'] = -1 * np.ones( N_galaxies)
    master_table['sMass'] = -1 * np.ones( N_galaxies)
    master_table['dmMass_to_sMass_ratio'] = np.zeros( N_galaxies)
    master_table['dmMass_to_sMass_ratio_error'] = -1 * np.ones( N_galaxies)
    ###########################################################################


    ###########################################################################
    # For all of the galaxies in the 'master_table'...
    # ~   ~   ~   ~   ~   ~   ~   ~   ~   ~   ~   ~   ~   ~   ~   ~   ~   ~   ~
    for i in range( N_galaxies):
        #######################################################################
        # Build the file names for the rotation curve and galaxy statistic data
        #    from the 'MaNGA_plate' and 'MaGNA_fiberID' columns of the
        #    'master_file.'
        #----------------------------------------------------------------------
        plate = master_table['MaNGA_plate'][i]
        fiberID = master_table['MaNGA_fiberID'][i]
        gal_ID = str(plate) + '-' + str(fiberID)
        print("gal_ID MAIN:", gal_ID)

        rot_curve_filename = ROT_CURVE_MASTER_FOLDER + '/' + gal_ID \
                                        + '_rot_curve_data.txt'
        gal_stat_filename = ROT_CURVE_MASTER_FOLDER + '/' + gal_ID \
                                        + '_gal_stat_data.txt'
        #######################################################################


        #######################################################################
        # Set of functions to run the set of rotation curve and of galaxy
        #    statistic files through.
        #----------------------------------------------------------------------
        param_outputs = fit_rot_curve( rot_curve_filename,
                                            gal_stat_filename,
                                            TRY_N)

        for col_name in param_outputs:
            master_table[ col_name][i] = param_outputs[ col_name]

        mass_outputs = estimate_dark_matter( param_outputs, rot_curve_filename)

        for col_name in mass_outputs:
            master_table[ col_name][i] = mass_outputs[ col_name]
        #######################################################################
        # ~   ~   ~   ~   ~   ~   ~   ~   ~   ~   ~   ~   ~   ~   ~   ~   ~   ~     ~


    ###########################################################################
    # Save the 'master_table.'
    #--------------------------------------------------------------------------
    master_table.write( MASTER_FILE_NAME, format='ascii.commented_header',
                       overwrite=True)
    ###########################################################################

else:
    ###########################################################################
    # For the galaxies contained within the 'FILE_IDS' array...
    # ~   ~   ~   ~   ~   ~   ~   ~   ~   ~   ~   ~   ~   ~   ~   ~   ~   ~   ~
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
        # Print the list of file names.
        #----------------------------------------------------------------------
#        print("rot_curve_files:", rot_curve_files)
#        print("gal_stat_files:", gal_stat_files)
        #######################################################################


        #######################################################################
        # Set of functions to run the set of rotation curve and of galaxy
        #    statistic files through.
        #----------------------------------------------------------------------
        param_outputs = fit_rot_curve( rot_curve_filename,
                                            gal_stat_filename,
                                            TRY_N)

        mass_outputs = estimate_dark_matter( param_outputs, rot_curve_filename)
        #######################################################################


        #######################################################################
        # Print the 'param_outputs' and 'mass_outputs' dictionaries.
        #----------------------------------------------------------------------
        print("param_outputs:", param_outputs)
        print("mass_outputs:", mass_outputs)
        #######################################################################
    # ~   ~   ~   ~   ~   ~   ~   ~   ~   ~   ~   ~   ~   ~   ~   ~   ~   ~   ~


'''
###############################################################################
# Create a list of tuples with the column names along with the associated units
#    for each column.
#
# NOTE: If a column has an associated unit of 'Unit(1),' the column is
#       dimensionless
#------------------------------------------------------------------------------
col_names_with_units = [('vflag', Unit(1)),
                        ('v_max_best', Unit('km/s')),
                        ('v_max_sigma', Unit('km/s')),
                        ('turnover_rad_best', Unit('kpc')),
                        ('turnover_rad_sigma', Unit('kpc')),
                        ('alpha_best', Unit(1)),
                        ('alpha_sigma', Unit(1)),
                        ('chi_square_rot', Unit(1)),
                        ('pos_v_max_best', Unit('km/s')),
                        ('pos_v_max_sigma', Unit('km/s')),
                        ('pos_turnover_rad_best', Unit('kpc')),
                        ('pos_turnover_rad_sigma', Unit('kpc')),
                        ('pos_alpha_best', Unit(1)),
                        ('pos_alpha_sigma', Unit(1)),
                        ('pos_chi_square_rot', Unit(1)),
                        ('neg_v_max_best', Unit('km/s')),
                        ('neg_v_max_sigma', Unit('km/s')),
                        ('neg_turnover_rad_best', Unit('kpc')),
                        ('neg_turnover_rad_sigma', Unit('kpc')),
                        ('neg_alpha_best', Unit(1)),
                        ('neg_alpha_sigma', Unit(1)),
                        ('neg_chi_square_rot', Unit(1)),
                        ('center_flux', Unit('erg / (cm2 s)')),
                        ('center_flux_error', Unit('erg / (cm2 s)')),
                        ('total_mass', Unit('solMass')),
                        ('total_mass_error', Unit('solMass')),
                        ('dmMass', Unit('solMass')),
                        ('dmMass_error', Unit('solMass')),
                        ('sMass', Unit('solMass')),
                        ('dmMass_to_sMass_ratio', Unit(1)),
                        ('dmMass_to_sMass_ratio_error', Unit(1))]
###############################################################################
'''
'''
###############################################################################
# Initialize the 'master_table' to have -1's in all of the columns listed in
#    the 'col_names' array.
#
# WARNING: If this function is ran, the data contained in the columns with
#          names specified in 'col_names_with_units' are overwritten with -1s.
#------------------------------------------------------------------------------
master_table = initialize_master_table( master_table, col_names_with_units)
###############################################################################
'''
'''
###############################################################################
# Initialize the data fields to pull in matching for each call to
#    'pull_matched_data().'
#------------------------------------------------------------------------------
vflag_pulls = col_names_with_units[ 0 : 1]
best_param_pulls = col_names_with_units[ 1 : 24]
mass_estimate_pulls = col_names_with_units[ 24 : ]
###############################################################################
'''
'''
###############################################################################
# Set of functions to run the set of rotation curves and set of galaxy
# statistics through.
#
# NOTE: After each function is ran and the data is matched to the
#       'master_table,' the 'master_table' is written so data is not lost if
#       one function crashes the program.
#------------------------------------------------------------------------------
vflag_ref_table = build_vflag_ref_table( CROSS_REF_FILE_NAMES)
master_table = pull_matched_data( master_table, vflag_ref_table, vflag_pulls)

#ascii.write( master_table, MASTER_FILE_NAME, format = 'ecsv', overwrite = True)
# -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -
master_table = fit_rot_curve_files( master_table,
                                   rot_curve_files, gal_stat_files,
                                   TRY_N, ROT_CURVE_MASTER_FOLDER,
                                   IMAGE_DIR)

#ascii.write( master_table, MASTER_FILE_NAME, format = 'ecsv', overwrite = True)
# -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -
#mass_estimate_table = estimate_dark_matter( best_fit_param_table,
#                                           ROT_CURVE_MASTER_FOLDER,
#                                           IMAGE_FORMAT, IMAGE_DIR)
#master_table = pull_matched_data( master_table, mass_estimate_table,
#                                 mass_estimate_pulls)
#
#ascii.write( master_table, MASTER_FILE_NAME, format = 'ecsv', overwrite = True)
# -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -
#plot_mass_ratios( master_table, IMAGE_FORMAT, IMAGE_DIR)
#------------------------------------------------------------------------------
#     DIAGNOSTIC FUNCTIONS
# ~   ~   ~   ~   ~   ~   ~   ~   ~   ~   ~   ~   ~   ~   ~   ~   ~   ~   ~   ~
#analyze_rot_curve_discrep( master_table, IMAGE_FORMAT, IMAGE_DIR)
#analyze_chi_square( master_table, IMAGE_FORMAT, IMAGE_DIR)
###############################################################################
'''

###############################################################################
# Clock the program's run time to check performance.
#------------------------------------------------------------------------------
FINISH = datetime.datetime.now()
print("Runtime (COMPLETED):", FINISH - START)
###############################################################################