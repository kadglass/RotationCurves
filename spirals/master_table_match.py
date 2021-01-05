'''
Extract information from master table and insert into another data table.
'''


import numpy as np

from astropy.table import Table, QTable


################################################################################
# Added column parameters
#-------------------------------------------------------------------------------
columns_to_extract = ['nsa_elpetro_th50_r']

default_values = [-99.]
################################################################################



################################################################################
# Data files
#-------------------------------------------------------------------------------
# Master table to use
master_filename = 'DRPall-master_file.txt'

# Data file
galaxy_filename = '/Users/kellydouglass/Desktop/Pipe3D-master_file_vflag_10_smooth2p27_N2O2_noWords.txt'
################################################################################



################################################################################
# Import data
#-------------------------------------------------------------------------------
master = QTable.read(master_filename, format='ascii.ecsv')

galaxies = Table.read(galaxy_filename, format='ascii.commented_header')
################################################################################



################################################################################
# Build look-up dictionary for row indices
#-------------------------------------------------------------------------------
index_dict = {}

for i in range(len(master)):
    gal_ID = (master['MaNGA_plate'][i], master['MaNGA_IFU'][i])
    index_dict[gal_ID] = i
################################################################################



################################################################################
# Initialize new columns in galaxies table
#-------------------------------------------------------------------------------
for i in range(len(columns_to_extract)):
    galaxies[columns_to_extract[i]] = default_values[i]
################################################################################



################################################################################
# Match and extract data
#-------------------------------------------------------------------------------
for i in range(len(galaxies)):

    gal_ID = (galaxies['MaNGA_plate'][i], galaxies['MaNGA_IFU'][i])

    if gal_ID in index_dict:

        j = index_dict[gal_ID]

        for column in columns_to_extract:

            galaxies[column][i] = master[column][j].value
################################################################################



################################################################################
# Save updated table
#-------------------------------------------------------------------------------
galaxies.write(galaxy_filename, format='ascii.commented_header', overwrite=True)
################################################################################


