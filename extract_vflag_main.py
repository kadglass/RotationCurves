# -*- coding: utf-8 -*-
"""Created on Wed Feb 13 2019
@author: Jacob A. Smith
@version: 1.0

Main script file for extract_vflag_vX_X.
"""
import datetime
START = datetime.datetime.now()

import os.path
from astropy.table import Table

###############################################################################
# Boolean variables to specify if the script is being run in Bluehive.
#------------------------------------------------------------------------------
WORKING_IN_BLUEHIVE = True
###############################################################################


###############################################################################
# Array of string representations of the criteria by which to match the
#    catalogs.
#------------------------------------------------------------------------------
match_criteria = ['MaNGA_plate', 'MaNGA_fiberID']
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
###############################################################################


###############################################################################
# Import functions from 'extract_vflag_vX_X.'
#------------------------------------------------------------------------------
from extract_vflag_v1 import build_ref_table, match_vflag
###############################################################################


###############################################################################
# Read in the 'master_table.'
#------------------------------------------------------------------------------
master_table = Table.read( MASTER_FILE_NAME, format='ascii.ecsv')
###############################################################################


###############################################################################
# Functions to build the 'vflag_ref_table' from various files and then match
#    to the 'master_table' according to 'MaNGA_plate' and 'MaNGA_fiberID.'
#------------------------------------------------------------------------------
vflag_ref_table = build_ref_table( CROSS_REF_FILE_NAMES)
master_table = match_vflag( master_table, vflag_ref_table, match_criteria)
###############################################################################


###############################################################################
# Write the 'master_table.'
#------------------------------------------------------------------------------
master_table.write( MASTER_FILE_NAME, format='ascii.ecsv', overwrite=True)
###############################################################################