'''
Determine the maximum deprojected radius from the best-fit velocity map.
'''


from astropy.table import Table #QTable

from extract_Rmax_v1 import match_Rmax_map


###############################################################################
# User inputs
#------------------------------------------------------------------------------
#DATA_PIPELINE = 'Pipe3D'
#DATA_PIPELINE = 'DRP'

MANGA_FOLDER = '/Users/kellydouglass/Documents/Research/data/SDSS/dr16/manga/spectro/'
VEL_MAP_FOLDER = MANGA_FOLDER + 'analysis/v2_4_3/2.2.1/HYB10-GAU-MILESHC/'

#MASTER_FILE_NAME = 'Pipe3D-master_file_vflag_10_smooth2p27.txt'
MASTER_FILE_NAME = 'DRP-master_file_vflag_BB_smooth1p85_mapFit_N2O2_HIdr2_morph_v6.txt'
###############################################################################


###############################################################################
# Read in the 'master_table.'
#------------------------------------------------------------------------------
#master_table = QTable.read( MASTER_FILE_NAME, format='ascii.ecsv')
master_table = Table.read(MASTER_FILE_NAME, format='ascii.commented_header')
###############################################################################


###############################################################################
# Find the Rmax value for each object in the master_table
#------------------------------------------------------------------------------
#master_table = match_Rmax( master_table, DATA_PIPELINE)
master_table = match_Rmax_map(master_table, VEL_MAP_FOLDER)
###############################################################################


###############################################################################
# Write the 'master_table.'
#------------------------------------------------------------------------------
master_table.write(MASTER_FILE_NAME, 
                   format='ascii.commented_header',#'ascii.ecsv', 
                   overwrite=True)
###############################################################################
