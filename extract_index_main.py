'''
Main script file for extract_index_vX.
'''
import datetime
START = datetime.datetime.now()

#import os.path
from astropy.table import Table,QTable

from extract_index_v1 import match_index


###############################################################################
# User inputs
#------------------------------------------------------------------------------
INDEX_FILE_NAME = '/Users/kellydouglass/Documents/Drexel/Research/Data/kias1033_5_P-MJD-F_MPAJHU_ZdustOS_stellarMass_BPT_SFR_NSA_correctVflag.txt'

MASTER_FILE_NAME = 'master_file_vflag_10.txt'
master_file_format = 'ascii.ecsv'
###############################################################################


###############################################################################
# Read in the 'master_table.'
#------------------------------------------------------------------------------
if master_file_format == 'ascii.ecsv':
    master_table = QTable.read( MASTER_FILE_NAME, format=master_file_format)
else:
    master_table = Table.read( MASTER_FILE_NAME, format=master_file_format)
###############################################################################


###############################################################################
# Read in the table with KIAS indices
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
master_table.write( MASTER_FILE_NAME, format=master_file_format, overwrite=True)
###############################################################################



###############################################################################
# Clock the program's run time to check performance.
#------------------------------------------------------------------------------
FINISH = datetime.datetime.now()
print("Runtime (COMPLETED):", FINISH - START)
###############################################################################