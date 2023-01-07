'''
Extract the H2 data for the MaNGA galaxies.
'''



from astropy.table import QTable, Table

from extract_H2_functions import match_H2_MASCOT, match_H2_ALMaQUEST, match_H2_xCOLDGASS



################################################################################
# User inputs
#-------------------------------------------------------------------------------
#DATA_PIPELINE = 'Pipe3D'
DATA_PIPELINE = 'DRP'

#MASTER_FILE_NAME = 'DRP-master_file_vflag_BB_smooth1p85_mapFit_N2O2_HIdr2_morph_SK_v6.txt'
MASTER_FILE_NAME = 'DRP-dr17_vflag_BB_smooth2_mapFit_morph_AJLaBarca.txt'
################################################################################


################################################################################
# Read in the 'master_table.'
#-------------------------------------------------------------------------------
master_table = Table.read(MASTER_FILE_NAME, format='ascii.commented_header')
################################################################################


################################################################################
# Match to the 'master_table'
#-------------------------------------------------------------------------------
master_table = match_H2_xCOLDGASS(master_table)
master_table = match_H2_ALMaQUEST(master_table)
master_table = match_H2_MASCOT(master_table)
################################################################################


################################################################################
# Write the 'master_table.'
#-------------------------------------------------------------------------------
master_table.write(MASTER_FILE_NAME[:-4] + '_H2.txt', 
                   format='ascii.commented_header', 
                   overwrite=True)
################################################################################