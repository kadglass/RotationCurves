'''
Main script file for extract_index_vX_X.
'''
import datetime
START = datetime.datetime.now()

import os.path
from astropy.table import Table

###############################################################################
# Boolean variables to specify if the script is being run in Bluehive.
#------------------------------------------------------------------------------
WORKING_IN_BLUEHIVE = False
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
INDEX_FILE_NAME = '/Users/kellydouglass/Documents/Drexel/Research/Data/kias1033_5_P-MJD-F_MPAJHU_ZdustOS_stellarMass_BPT_SFR_NSA_correctVflag.txt'

MASTER_FILE_NAME = 'master_file_vflag_10.txt'
###############################################################################


###############################################################################
# Import functions from 'extract_vflag_vX_X.'
#------------------------------------------------------------------------------
from extract_index_v1 import match_index
###############################################################################


###############################################################################
# Read in the 'master_table.'
#------------------------------------------------------------------------------
master_table = Table.read( MASTER_FILE_NAME, format='ascii.ecsv')
###############################################################################


###############################################################################
# Read in the 'abundance_table.'
#------------------------------------------------------------------------------
index_table = Table.read( INDEX_FILE_NAME, format='ascii.commented_header')
###############################################################################


###############################################################################
# Match to the 'master_table' according to 'NSA_plate', 'NSA_MJD', and 
# 'NSA_fiberID'
#------------------------------------------------------------------------------
master_table = match_index( master_table, index_table)
###############################################################################


###############################################################################
# Write the 'master_table.'
#------------------------------------------------------------------------------
master_table.write( MASTER_FILE_NAME, format='ascii.ecsv', overwrite=True)
###############################################################################



###############################################################################
# Clock the program's run time to check performance.
#------------------------------------------------------------------------------
FINISH = datetime.datetime.now()
print("Runtime (COMPLETED):", FINISH - START)
###############################################################################