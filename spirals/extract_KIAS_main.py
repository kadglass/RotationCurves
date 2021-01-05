'''
Matches galaxies to the KIAS-VAGC by their P-MJD-F.
'''

import datetime
START = datetime.datetime.now()

#import os.path
from astropy.table import Table,QTable

from extract_KIAS_functions import PMJDF_match


###############################################################################
# User inputs
#------------------------------------------------------------------------------
INDEX_FILE_NAME = '/Users/kellydouglass/Documents/Research/data/SDSS/kias1033_5_MPAJHU_ZdustOS.txt'

#MASTER_FILE_NAME = 'master_file_vflag_10.txt'
MASTER_FILE_NAME = 'DRPall-master_file.txt'
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
columns_to_add = ['imc', 'aimc']

master_table = PMJDF_match( master_table, index_table, columns_to_add)
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