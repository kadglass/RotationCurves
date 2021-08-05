'''
Match to the KIAS-VAGC by the galaxies' unique KIAS index number.
'''


import datetime
START = datetime.datetime.now()

from astropy.table import Table, QTable

from extract_KIAS2_functions import match_by_index



###############################################################################
# Data columns to pull from the KIAS-VAGC and add to the other data file
#------------------------------------------------------------------------------
columns_to_add = ['u_r']
#column_units = ['']
###############################################################################



###############################################################################
# File names
#------------------------------------------------------------------------------
KIAS_FILE_NAME = '/Users/kellydouglass/Documents/Research/data/kias1033_5_MPAJHU_ZdustOS.txt'

#MASTER_FILE_NAME = 'Pipe3D-master_file_vflag_BB_10_smooth2p27.txt'
#MASTER_FILE_NAME = '/Users/kellydouglass/Desktop/Pipe3D-master_file_vflag_10_smooth2p27_N2O2_noWords.txt'
MASTER_FILE_NAME = 'Pipe3D-master_file_vflag_BB_minimize_chi10_smooth2p27_mapFit_N2O2_HIdr2_noWords_v5.txt'

master_file_format = 'ascii.commented_header'
###############################################################################



###############################################################################
# Read in data files
#------------------------------------------------------------------------------
if master_file_format == 'ascii.ecsv':
    master_table = QTable.read( MASTER_FILE_NAME, format='ascii.ecsv')
else:
    master_table = Table.read( MASTER_FILE_NAME, format=master_file_format)

KIAS_table = Table.read( KIAS_FILE_NAME, format='ascii.commented_header')
###############################################################################



###############################################################################
# Match to the 'master_table' according to 'index'
#------------------------------------------------------------------------------
master_table = match_by_index( master_table, 
                               KIAS_table, 
                               columns_to_add#, 
                               #column_units=column_units
                               )
###############################################################################



###############################################################################
# Write the 'master_table.'
#------------------------------------------------------------------------------
if master_file_format == 'ascii.ecsv':
    master_table.write( MASTER_FILE_NAME, format='ascii.ecsv', overwrite=True)
else:
    master_table.write( MASTER_FILE_NAME, format=master_file_format, overwrite=True)
###############################################################################



###############################################################################
# Clock the program's run time to check performance.
#------------------------------------------------------------------------------
FINISH = datetime.datetime.now()
print("Runtime (COMPLETED):", FINISH - START)
###############################################################################