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
MANGA_FOLDER = '/Users/nityaravi/Documents/Research/RotationCurves/data/manga/'
MASTER_FILE_NAME = MANGA_FOLDER + 'output_files/DR17/CURRENT_MASTER_TABLE/H_alpha_HIvel_BB_extinction_H2_R90_v3p5.fits'

MASCOT_FILE_NAME = MANGA_FOLDER + 'MASCOT.fits'
COLD_GASS_FILE_NAME = MANGA_FOLDER + 'xCOLDGASS_PubCat.fits'

################################################################################


################################################################################
# Read in the 'master_table.'
#-------------------------------------------------------------------------------
#master_table = Table.read(MASTER_FILE_NAME, format='ascii.commented_header')
master_table = Table.read(MASTER_FILE_NAME, format='fits')

################################################################################


################################################################################
# Match to the 'master_table'
#-------------------------------------------------------------------------------
master_table = match_H2_xCOLDGASS(master_table, COLD_GASS_FILE_NAME)
#master_table = match_H2_ALMaQUEST(master_table)
master_table = match_H2_MASCOT(master_table, MASCOT_FILE_NAME)
################################################################################


################################################################################
# Write the 'master_table.'
#-------------------------------------------------------------------------------
#master_table.write(MASTER_FILE_NAME[:-4] + '_H2.txt', 
#                   format='ascii.commented_header', 
#                   overwrite=True)

master_table.write(MANGA_FOLDER + 'output_files/DR17/CURRENT_MASTER_TABLE/H_alpha_HIvel_BB_extinction_H2_MxCG_R90_v3p5.fits', 
                   format='fits', 
                   overwrite=True)
################################################################################