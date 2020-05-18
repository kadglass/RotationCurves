
from astropy.table import QTable

from extract_Mturn_v1 import extract_Mturn



################################################################################
# User inputs
#-------------------------------------------------------------------------------
DATA_PIPELINE = 'Pipe3D'
#DATA_PIPELINE = 'DRP'

MASTER_FILE_NAME = 'Pipe3D-master_file_vflag_10_smooth2p27.txt'
################################################################################



################################################################################
# Read in the master data table
#-------------------------------------------------------------------------------
master_table = QTable.read( MASTER_FILE_NAME, format='ascii.ecsv')
################################################################################



################################################################################
# Calculate M*(Rturn) and add to master table
#-------------------------------------------------------------------------------
master_table = extract_Mturn( master_table, DATA_PIPELINE)
################################################################################



################################################################################
# Write the master table
#-------------------------------------------------------------------------------
master_table.write( MASTER_FILE_NAME, format='ascii.ecsv', overwrite=True)
################################################################################