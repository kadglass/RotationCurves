'''
Extract the morphology data for the MaNGA galaxies from the MaNGA Visual 
Morphology Catalog 
(https://www.sdss.org/dr16/data_access/value-added-catalogs/?vac_id=manga-visual-morphologies-from-sdss-and-desi-images)
'''


from astropy.table import Table, QTable

from extract_morphology_functions import match_morph_visual, \
                                         match_morph_gz, \
                                         match_morph_dl



################################################################################
# File names
#-------------------------------------------------------------------------------
DATA_PIPELINE = 'DRP'

#MASTER_FILENAME = 'DRP-master_file_vflag_BB_smooth1p85_mapFit_N2O2_HIdr2_v5.txt'
MASTER_FILENAME = 'DRP-dr17_vflag_BB_smooth2_mapFit_AJLaBarca.txt'
################################################################################




################################################################################
# Read in the master table
#-------------------------------------------------------------------------------
#master_table = QTable.read(MASTER_FILENAME, format='ascii.ecsv')
master_table = Table.read(MASTER_FILENAME, format='ascii.commented_header')
################################################################################




################################################################################
# Match to the master table
#-------------------------------------------------------------------------------
# Visual morphologies
master_table = match_morph_visual(master_table)

# Galaxy Zoo
master_table = match_morph_gz(master_table)

# Deep learning
master_table = match_morph_dl(master_table)
################################################################################




################################################################################
# Save the master table
#-------------------------------------------------------------------------------
master_table.write(MASTER_FILENAME[:-13] + 'morph_' + MASTER_FILENAME[-13:], 
                   #format='ascii.ecsv', 
                   format='ascii.commented_header', 
                   overwrite=True)
################################################################################