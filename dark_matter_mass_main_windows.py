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

###############################################################################
# File format for saved images
#------------------------------------------------------------------------------
image_format = 'eps'
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
from dark_matter_mass_v1_1 import extract_matched_data, \
                                write_matched_data, \
                                fit_rot_curve_files, \
                                write_best_params, \
                                estimate_dark_matter, \
                                plot_mass_ratios, \
                                write_mass_estimates, \
                                analyze_rot_curve_discrep, \
                                analyze_chi_square
###############################################################################


###############################################################################
# Create the file name lists of the rotation curve and galaxy statistic
# files to be ran.
#------------------------------------------------------------------------------
#rot_curve_files = glob.glob( ROT_CURVE_MASTER_FOLDER + \
#                              '/*_rot_curve_data.txt')
#gal_stat_files = glob.glob( ROT_CURVE_MASTER_FOLDER + \
#                              '/*_gal_stat_data.txt')
###############################################################################


###############################################################################
# Code to isolate files and run it through all of the functions from
# fit_rotation_curve_vX_X.
# ~   ~   ~   ~   ~   ~   ~   ~   ~   ~   ~   ~   ~   ~   ~   ~   ~   ~   ~   ~
FILE_IDS = ['7495-12705']

rot_curve_files = []
gal_stat_files = []

for file_name in FILE_IDS:
    rot_curve_files.append( ROT_CURVE_MASTER_FOLDER + file_name \
                           + '_rot_curve_data.txt')
    gal_stat_files.append( ROT_CURVE_MASTER_FOLDER + file_name  \
                           + '_gal_stat_data.txt')

#print("rot_curve_files:", rot_curve_files)
#print("gal_stat_files:", gal_stat_files)
###############################################################################


###############################################################################
# Set of functions to run the set of rotation curves and set of galaxy
# statistics through.
#------------------------------------------------------------------------------
vflag_list, mstar_NSA_list = extract_matched_data( MASTER_FILE_NAME,
                                          CROSS_REF_FILE_NAMES)
write_matched_data( vflag_list, mstar_NSA_list, MASTER_FILE_NAME)
# -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -
best_fit_param_table = fit_rot_curve_files( rot_curve_files, gal_stat_files,
                                      TRY_N, ROT_CURVE_MASTER_FOLDER,
                                      IMAGE_DIR)
write_best_params( best_fit_param_table, gal_stat_files,
                  MASTER_FILE_NAME, ROT_CURVE_MASTER_FOLDER)
# -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -
mass_estimate_table = estimate_dark_matter( best_fit_param_table,
                                           rot_curve_files, gal_stat_files,
                                           IMAGE_DIR)
write_mass_estimates( mass_estimate_table, MASTER_FILE_NAME)
plot_mass_ratios( mass_estimate_table, MASTER_FILE_NAME, IMAGE_DIR)
#------------------------------------------------------------------------------
#     DIAGNOSTIC FUNCTIONS
# ~   ~   ~   ~   ~   ~   ~   ~   ~   ~   ~   ~   ~   ~   ~   ~   ~   ~   ~   ~
analyze_rot_curve_discrep( rot_curve_files, gal_stat_files,
                   MASTER_FILE_NAME, ROT_CURVE_MASTER_FOLDER, IMAGE_DIR)

analyze_chi_square( MASTER_FILE_NAME, IMAGE_DIR)
###############################################################################



###############################################################################
# Clock the program's run time to check performance.
###############################################################################
FINISH = datetime.datetime.now()
print("Runtime (COMPLETED):", FINISH - START)