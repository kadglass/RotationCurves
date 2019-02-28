'''
Extract various data values from the NSA catalog.
'''


################################################################################
#
#   IMPORT MODULES
#
#-------------------------------------------------------------------------------


from astropy.table import Table
from astropy.io import fits

import numpy as np


################################################################################
#
#   FILE NAMES
#
#-------------------------------------------------------------------------------


# File name of data to be matched
data_filename = 'master_file.txt'

# File name of NSA catalog
NSA_filename = '/Users/kellydouglass/Documents/Drexel/Research/Data/nsa_v0_1_2.fits'


################################################################################
#
#   IMPORT DATA
#
#-------------------------------------------------------------------------------


# Data table of galaxies to be matched
data_table = Table.read(data_filename, format='ascii.ecsv')

N = len(data_table) # Number of galaxies


# Data table of NSA catalog
NSA_table = fits.open(NSA_filename)
NSA_data = NSA_table[1].data


################################################################################
#
#   INITIALIZE NEW COLUMNS
#
#-------------------------------------------------------------------------------


rabsmag_col = np.zeros(N)


################################################################################
#
#   EXTRACT GALAXY INFO
#
#-------------------------------------------------------------------------------
# Match via NSAID (unique ID number in NSA catalog)


for i in range(len(data_table)):

    # Find galaxy in NSA catalog
    boolean = NSA_data['NSAID'] == data_table['NSA_index'][i]

    # Array of absolute magnitudes for this galaxy (FNugriz)
    absmag_array = NSA_data['ABSMAG'][boolean]

    rabsmag_col[i] = absmag_array[0][4] # SDSS r-band


################################################################################
#
#   UPDATE & SAVE DATA TABLE
#
#-------------------------------------------------------------------------------


# Add new column(s) to table
data_table['rabsmag'] = rabsmag_col

data_table.write(data_filename, format='ascii.ecsv', overwrite=True)