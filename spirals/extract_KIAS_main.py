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
INDEX_FILE_NAME = '/Users/kellydouglass/Documents/Research/data/kias1033_5_MPAJHU_ZdustOS.txt'

#MASTER_FILE_NAME = 'master_file_vflag_10.txt'
#MASTER_FILE_NAME = 'DRPall-master_file.txt'
#MASTER_FILE_NAME = 'Pipe3D-master_file_vflag_BB_minimize_chi10_smooth2p27_mapFit_N2O2_HIdr2_noWords_v5.txt'
# MASTER_FILE_NAME = 'DRP-dr17_vflag_BB_smooth2_mapFit_AJLaBarca.txt'
MASTER_FILE_NAME = '../../Nitya_Ravi/master_table_H_alpha_BB_HI_H2_MxCG_R90_CMD_ZPG16R_SFR_MZ.txt'

master_file_format = 'ascii.commented_header'
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
columns_to_add = ['index', 'imc', 'aimc', 'cd', 'u_r']

master_table = PMJDF_match( master_table, index_table, columns_to_add)

master_table['u_r'].name = 'u_r_KIAS'
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