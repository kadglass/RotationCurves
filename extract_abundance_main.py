'''
Main script file for extract_abundance_vX.
'''
import datetime
START = datetime.datetime.now()

import os.path
from astropy.table import Table, QTable

from extract_abundance_v1 import match_abundance


###############################################################################
# User inputs
#------------------------------------------------------------------------------
method = 'O3N2'

#ABUNDANCE_FILE_NAME = '/Users/kellydouglass/Documents/Drexel/Research/Data/kias1033_5_P-MJD-F_MPAJHU_ZdustOS_stellarMass_BPT_SFR_NSA_correctVflag.txt'
ABUNDANCE_FILE_NAME = '/Users/kellydouglass/Documents/Drexel/Research/Programs/MartiniMethods/comp_Z_Martini_' + method + '_kias1033_5_Martini_MPAJHU_flux_oii_dustCorr.txt'

MASTER_FILE_NAME = 'master_file_vflag_10.txt'
###############################################################################


###############################################################################
# Read in the 'master_table.'
#------------------------------------------------------------------------------
master_table = QTable.read( MASTER_FILE_NAME, format='ascii.ecsv')
###############################################################################


###############################################################################
# Read in the 'abundance_table.'
#------------------------------------------------------------------------------
abundance_table = Table.read( ABUNDANCE_FILE_NAME, format='ascii.commented_header')
###############################################################################


###############################################################################
# Match to the 'master_table' according to 'index'
#------------------------------------------------------------------------------
master_table = match_abundance( master_table, abundance_table, method)
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