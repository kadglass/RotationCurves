'''
Extract additional data values to add to master data file.
'''


from astropy.table import QTable
import astropy.units as u

from extract_Vmax_v1 import match_Vmax
from extract_HI_functions import match_HI
from extract_DRP_functions import match_DRP



###############################################################################
# User inputs
#------------------------------------------------------------------------------
#DATA_PIPELINE = 'Pipe3D'
DATA_PIPELINE = 'DRP'

#MASTER_FILE_NAME = 'Pipe3D-master_file_vflag_BB_minimize_chi10_smooth2p27.txt'
MASTER_FILE_NAME = 'DRPall-master_file.txt'
###############################################################################


###############################################################################
# Read in the 'master_table.'
#------------------------------------------------------------------------------
master_table = QTable.read( MASTER_FILE_NAME, format='ascii.ecsv')
###############################################################################


###############################################################################
# Match to the 'master_table'
#------------------------------------------------------------------------------
#master_table = match_Vmax( master_table, DATA_PIPELINE)
#master_table = match_HI( master_table)
master_table = match_DRP( master_table, ['nsa_elpetro_th50_r'], [u.arcsec])
###############################################################################


###############################################################################
# Write the 'master_table.'
#------------------------------------------------------------------------------
master_table.write( MASTER_FILE_NAME, format='ascii.ecsv', overwrite=True)
###############################################################################