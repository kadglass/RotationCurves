
from astropy.table import Table, QTable

from extract_CMD_v1 import match_CMD


###############################################################################
# User inputs
#------------------------------------------------------------------------------
#CMD_FILE_NAME = '/Users/kellydouglass/Documents/Drexel/Research/Data/kias1033_5_P-MJD-F_MPAJHU_ZdustOS_stellarMass_BPT_SFR_NSA_correctVflag_Voronoi_CMD.txt'
CMD_FILE_NAME = '/Users/kellydouglass/Documents/Research/data/kias1033_5_MPAJHU_ZdustOS_HI100_NSAv012_CMDJan2020.txt'

#MASTER_FILE_NAME = 'Pipe3D-master_file_vflag_BB_10_smooth2p27.txt'
#MASTER_FILE_NAME = 'Pipe3D-master_file_vflag_BB_minimize_chi10_smooth2p27_mapFit_N2O2_noWords.txt'
MASTER_FILE_NAME = 'Pipe3D-master_file_vflag_BB_minimize_chi10_smooth2p27_mapFit_N2O2_HIdr2_noWords_v5.txt'
###############################################################################


###############################################################################
# Read in the 'master_table.'
#------------------------------------------------------------------------------
#master_table = QTable.read( MASTER_FILE_NAME, format='ascii.ecsv')
master_table = Table.read(MASTER_FILE_NAME, format='ascii.commented_header')
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


#print(master_table.colnames)


###############################################################################
# Write the 'master_table.'
#------------------------------------------------------------------------------
#master_table.write( MASTER_FILE_NAME, format='ascii.ecsv', overwrite=True)
master_table.write(MASTER_FILE_NAME, 
                   format='ascii.commented_header', 
                   overwrite=True)
###############################################################################
