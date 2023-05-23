'''
Extract various data values from the NSA catalog.
'''


################################################################################
# IMPORT MODULES
#-------------------------------------------------------------------------------
from astropy.table import QTable, Table

import numpy as np
################################################################################





################################################################################
################################################################################
def galaxies_dict(ref_table):
    '''
    Build a dictionary of the galaxies with NSA IDs

    Parameters:
    ===========

    ref_table : astropy table
        Galaxies with NSA IDs


    Returns:
    ========

    ref_dict : dictionary
        Keys are the NSA ID, value is the index number in the NSA catalog
    '''

    # Initialize dictionary of cell IDs with at least one galaxy in them
    ref_dict = {}

    for idx in range(len(ref_table)):

        galaxy_ID = (ref_table['NSAID'][idx])

        ref_dict[galaxy_ID] = idx

    return ref_dict
################################################################################
################################################################################




################################################################################
# FILE NAMES
#-------------------------------------------------------------------------------
# File name of data to be matched
#data_filename = 'master_file_vflag_10.txt'
#data_filename = 'master_file_vflag_10_smooth2-27.txt'
#data_filename = 'DRPall-master_file.txt'
#data_filename = 'DRP-master_file_vflag_BB_smooth1p85_mapFit_N2O2_HIdr2_morph_v6.txt'
data_filename = 'DRP-dr17_vflag_BB_smooth2_mapFit_AJLaBarca.txt'

# File name of NSA catalog
NSA_filename = '/Users/kellydouglass/Documents/Drexel/Research/Data/NSA/nsa_v1_0_1.fits'
################################################################################




################################################################################
# IMPORT DATA
#-------------------------------------------------------------------------------
# Data table of galaxies to be matched
#data_table = QTable.read(data_filename, format='ascii.ecsv')
data_table = Table.read(data_filename, format='ascii.commented_header')

N = len(data_table) # Number of galaxies


# Data table of NSA catalog
NSA_data = Table.read(NSA_filename, format='fits')
################################################################################




################################################################################
# INITIALIZE NEW COLUMNS
#-------------------------------------------------------------------------------
#data_table['rabsmag'] = np.zeros(N)
data_table['u_r'] = -99.*np.ones(N)
data_table['NSA_plate'] = np.zeros(N, dtype=int)
data_table['NSA_MJD'] = np.zeros(N, dtype=int)
data_table['NSA_fiberID'] = np.zeros(N, dtype=int)
#data_table['NSA_elpetro_th50'] = np.zeros(N)
################################################################################




################################################################################
# REFERENCE DICTIONARY
#
# Build dictionary of tuples for storing galaxies with KIAS-VAGC indices
#-------------------------------------------------------------------------------
ref_dict = galaxies_dict(NSA_data)
################################################################################




################################################################################
# EXTRACT GALAXY INFO
#
# Match via NSAID (unique ID number in NSA catalog)
#-------------------------------------------------------------------------------
N_missing = 0

for i in range(N):

    #index = data_table['NSA_index'][i]
    index = data_table['NSAID'][i]
    galaxy_ID = (index)

    if galaxy_ID in ref_dict:

        # Array of absolute magnitudes for this galaxy (FNugriz)
        '''
        absmag_array = NSA_data['ABSMAG'][ref_dict[galaxy_ID]]
        data_table['rabsmag'][i] = absmag_array[0][4] # SDSS r-band
        '''
        absmag_array = NSA_data['ELPETRO_ABSMAG'][ref_dict[galaxy_ID]]
        #data_table['rabsmag'][i] = absmag_array[4] # SDSS r-band
        data_table['u_r'][i] = absmag_array[2] - absmag_array[4]

        '''
        # Array of petrosian flux values for this galaxy (FNugriz)
        flux_array = NSA_data['ELPETRO_FLUX'][ref_dict[galaxy_ID]]
        data_table['u_r'][i] = -2.5*np.log10(flux_array[2]/flux_array[4])
        '''
        
        data_table['NSA_plate'][i] = NSA_data['PLATE'][ref_dict[galaxy_ID]]
        data_table['NSA_MJD'][i] = NSA_data['MJD'][ref_dict[galaxy_ID]]
        data_table['NSA_fiberID'][i] = NSA_data['FIBERID'][ref_dict[galaxy_ID]]
        '''
        data_table['NSA_elpetro_th50'][i] = NSA_data['ELPETRO_TH50_R'][ref_dict[galaxy_ID]]
        '''
    else:
        N_missing += 1
################################################################################




################################################################################
# UPDATE & SAVE DATA TABLE
#-------------------------------------------------------------------------------
#data_table.write(data_filename, format='ascii.ecsv', overwrite=True)
data_table.write(data_filename, format='ascii.commented_header', overwrite=True)
################################################################################





