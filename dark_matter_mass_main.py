#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 20 2018
@author: Jacob A. Smith
@version: 1.1

Main script file to fit rotation curve data files
(see rotation_curve_vX_X output) from MaNGA .fits files.
"""
import datetime
START = datetime.datetime.now()

import glob, os.path
import astropy.io.ascii as ascii

###############################################################################
# File format for saved images
#------------------------------------------------------------------------------
IMAGE_FORMAT = 'eps'
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
WORKING_IN_BLUEHIVE = True

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
TRY_N = 100000      # number of times to try line of best fit within
                    #    scipy.optimize.curve_fit

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
                                build_vflag_ref_table, \
                                fit_rot_curve_files, \
                                estimate_dark_matter, \
                                plot_mass_ratios, \
                                analyze_rot_curve_discrep, \
                                analyze_chi_square
###############################################################################


###############################################################################
# Create the file name lists of the rotation curve and galaxy statistic
# files to be ran.
#------------------------------------------------------------------------------
rot_curve_files = glob.glob( ROT_CURVE_MASTER_FOLDER + \
                              '/*_rot_curve_data.txt')
gal_stat_files = glob.glob( ROT_CURVE_MASTER_FOLDER + \
                              '/*_gal_stat_data.txt')
###############################################################################


###############################################################################
# Code to isolate files and run it through all of the functions from
# fit_rotation_curve_vX_X.
# ~   ~   ~   ~   ~   ~   ~   ~   ~   ~   ~   ~   ~   ~   ~   ~   ~   ~   ~   ~
FILE_IDS = ['7495-9101']

rot_curve_files = []
gal_stat_files = []

for file_name in FILE_IDS:
    rot_curve_files.append( ROT_CURVE_MASTER_FOLDER + '/' + file_name \
                           + '_rot_curve_data.txt')
    gal_stat_files.append( ROT_CURVE_MASTER_FOLDER + '/' + file_name  \
                           + '_gal_stat_data.txt')

print("rot_curve_files:", rot_curve_files)
print("gal_stat_files:", gal_stat_files)
###############################################################################


###############################################################################
# Read in the master file.
#------------------------------------------------------------------------------
master_table = ascii.read( MASTER_FILE_NAME, format = 'ecsv')
###############################################################################


###############################################################################
# Initialize the 'master_table' to have -1's in all of the columns listed in
#    the 'col_names' array.
#------------------------------------------------------------------------------
col_names = ['vflag',
             'v_max_best',
             'v_max_sigma',
             'turnover_rad_best',
             'turnover_rad_sigma',
             'alpha_best',
             'alpha_sigma',
             'chi_square_rot',
             'pos_v_max_best',
             'pos_v_max_sigma',
             'pos_turnover_rad_best',
             'pos_turnover_rad_sigma',
             'pos_alpha_best',
             'pos_alpha_sigma',
             'pos_chi_square_rot',
             'neg_v_max_best',
             'neg_v_max_sigma',
             'neg_turnover_rad_best',
             'neg_turnover_rad_sigma',
             'neg_alpha_best',
             'neg_alpha_sigma',
             'neg_chi_square_rot',
             'center_luminosity',
             'center_luminosity_err',
             'sMass_processed',
             'total_mass',
             'total_mass_error',
             'dmMass',
             'dmMass_error',
             'sMass',
             'dmMass_to_sMass_ratio',
             'dmMass_to_sMass_ratio_error']

master_table = initialize_master_table( master_table, col_names)
###############################################################################


###############################################################################
# Initialize the data fields to pull in matching for each call to
#    'pull_matched_data().'
#------------------------------------------------------------------------------
vflag_pulls = col_names[ 0 : 1]
best_param_pulls = col_names[ 1 : 24]
mass_estimate_pulls = col_names[ 24 : ]
###############################################################################


###############################################################################
# Set of functions to run the set of rotation curves and set of galaxy
# statistics through.
#------------------------------------------------------------------------------
vflag_ref_table = build_vflag_ref_table( CROSS_REF_FILE_NAMES)
master_table = pull_matched_data( master_table, vflag_ref_table, vflag_pulls)
# -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -
best_fit_param_table = fit_rot_curve_files( rot_curve_files, gal_stat_files,
                                      TRY_N, ROT_CURVE_MASTER_FOLDER,
                                      IMAGE_DIR)
master_table = pull_matched_data( master_table, best_fit_param_table,
                                 best_param_pulls)
# -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -
mass_estimate_table = estimate_dark_matter( master_table,
                                           IMAGE_FORMAT, IMAGE_DIR)
master_table = pull_matched_data( master_table, mass_estimate_table,
                                 mass_estimate_pulls)
# -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -
plot_mass_ratios( master_table, IMAGE_FORMAT, IMAGE_DIR)
#------------------------------------------------------------------------------
#     DIAGNOSTIC FUNCTIONS
# ~   ~   ~   ~   ~   ~   ~   ~   ~   ~   ~   ~   ~   ~   ~   ~   ~   ~   ~   ~
analyze_rot_curve_discrep( master_table, IMAGE_FORMAT, IMAGE_DIR)
analyze_chi_square( master_table, IMAGE_FORMAT, IMAGE_DIR)
###############################################################################


###############################################################################
# Write the 'master_table'.
#------------------------------------------------------------------------------
ascii.write( master_table, MASTER_FILE_NAME, format='ecsv', overwrite = True)
###############################################################################



###############################################################################
# Clock the program's run time to check performance.
#------------------------------------------------------------------------------
FINISH = datetime.datetime.now()
print("Runtime (COMPLETED):", FINISH - START)
###############################################################################