'''
Extract various data values from the NSA catalog.
'''


################################################################################
#
#   IMPORT MODULES
#
#-------------------------------------------------------------------------------


from astropy.table import QTable
from astropy.io import fits

import numpy as np

from extract_NSA_functions import galaxies_dict


################################################################################
#
#   FILE NAMES
#
#-------------------------------------------------------------------------------


# File name of data to be matched
data_filename = 'master_file_vflag_10.txt'

# File name of NSA catalog
NSA_filename = '/Users/kellydouglass/Documents/Drexel/Research/Data/nsa_v1_0_1.fits'


################################################################################
#
#   IMPORT DATA
#
#-------------------------------------------------------------------------------


# Data table of galaxies to be matched
data_table = QTable.read(data_filename, format='ascii.ecsv')

N = len(data_table) # Number of galaxies


# Data table of NSA catalog
NSA_table = fits.open(NSA_filename)
NSA_data = NSA_table[1].data


################################################################################
#
#   INITIALIZE NEW COLUMNS
#
#-------------------------------------------------------------------------------


data_table['rabsmag'] = np.zeros(N)



################################################################################
#
#   REFERENCE DICTIONARY
#
#-------------------------------------------------------------------------------
# Build dictionary of tuples for storing galaxies with KIAS-VAGC indices


ref_dict = galaxies_dict(NSA_data)



################################################################################
#
#   EXTRACT GALAXY INFO
#
#-------------------------------------------------------------------------------
# Match via NSAID (unique ID number in NSA catalog)


N_missing = 0

for i in range(len(data_table)):

    index = data_table['NSA_index'][i]
    galaxy_ID = (index)

    if galaxy_ID in ref_dict.keys():

        # Array of absolute magnitudes for this galaxy (FNugriz)
        '''
        absmag_array = NSA_data['ABSMAG'][ref_dict[galaxy_ID]]
        data_table['rabsmag'][i] = absmag_array[0][4] # SDSS r-band
        '''
        absmag_array = NSA_data['ELPETRO_ABSMAG'][ref_dict[galaxy_ID]]
        data_table['rabsmag'][i] = absmag_array[4] # SDSS r-band
    else:
        N_missing += 1


################################################################################
#
#   UPDATE & SAVE DATA TABLE
#
#-------------------------------------------------------------------------------


data_table.write(data_filename, format='ascii.ecsv', overwrite=True)

