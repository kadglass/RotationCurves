
import datetime
START = datetime.datetime.now()

import os.path
from astropy.table import Table, QTable

from extract_CMD_v1 import match_CMD


###############################################################################
# User inputs
#------------------------------------------------------------------------------
CMD_FILE_NAME = '/Users/kellydouglass/Documents/Drexel/Research/Data/kias1033_5_P-MJD-F_MPAJHU_ZdustOS_stellarMass_BPT_SFR_NSA_correctVflag_Voronoi_CMD.txt'

MASTER_FILE_NAME = 'Pipe3D-master_file_vflag_10_smooth2p27.txt'
###############################################################################


###############################################################################
# Read in the 'master_table.'
#------------------------------------------------------------------------------
master_table = QTable.read( MASTER_FILE_NAME, format='ascii.ecsv')
###############################################################################


###############################################################################
# Read in the 'CMD_table.'
#------------------------------------------------------------------------------
CMD_table = Table.read( CMD_FILE_NAME, format='ascii.commented_header')
###############################################################################


###############################################################################
# Match to the 'master_table' according to 'index'
#------------------------------------------------------------------------------
master_table = match_CMD( master_table, CMD_table)
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