'''
Extract the HI data for the MaNGA galaxies, and correct the velocities by their 
angles of inclination.
'''



from astropy.table import QTable, Table

from extract_HI_functions import match_HI, match_HI_dr2, match_HI_dr3, extract_HI_mass



###############################################################################
# User inputs
#------------------------------------------------------------------------------
#DATA_PIPELINE = 'Pipe3D'
# DATA_PIPELINE = 'DRP'

#MASTER_FILE_NAME = '/Users/kellydouglass/Desktop/Pipe3D-master_file_vflag_10_smooth2p27_N2O2_noWords.txt'
#MASTER_FILE_NAME = 'Pipe3D-master_file_vflag_BB_chi10_alpha10_smooth2p27.txt'
#MASTER_FILE_NAME = 'DRPall-master_file.txt'
#MASTER_FILE_NAME = 'Pipe3D-master_file_vflag_BB_minimize_chi10_smooth2p27_mapFit.txt'
#MASTER_FILE_NAME = 'Pipe3D-master_file_vflag_BB_minimize_chi10_smooth2p27_mapFit_N2O2_v3.txt'
# MASTER_FILE_NAME = '/Users/nityaravi/Documents/Research/RotationCurves/data/manga/output_files/DR17/disk_masses.fits'

DATA_DIRECTORY = '/Users/nityaravi/Documents/Research/RotationCurves/data/manga/'
MASTER_FILE_NAME = DATA_DIRECTORY + 'output_files/DR17/CURRENT_MASTER_TABLE/Elliptical_sphdisk_refitspirals_BPT_v7.fits'
dr3 = DATA_DIRECTORY + '/mangahiall.fits'
dr4 = DATA_DIRECTORY + '/himanga_dr4.fits'
###############################################################################


###############################################################################
# Read in the 'master_table.'
#------------------------------------------------------------------------------
#master_table = QTable.read( MASTER_FILE_NAME, format='ascii.ecsv')
#master_table = Table.read( MASTER_FILE_NAME, format='ascii.commented_header')
master_table = Table.read(MASTER_FILE_NAME, format='fits')
###############################################################################


###############################################################################
# Match to the 'master_table'
#------------------------------------------------------------------------------
#master_table = match_HI(master_table)
#master_table = match_HI_dr2(master_table)
# master_table = match_HI_dr3(master_table)
###############################################################################

#### get hi-manga dr3 data
master_table = extract_HI_mass(master_table, dr3, v=3)
master_table = extract_HI_mass(master_table, dr4, v=4)


###############################################################################
# Write the 'master_table.'
#------------------------------------------------------------------------------
#master_table.write(MASTER_FILE_NAME, format='ascii.ecsv', overwrite=True)
#master_table.write(MASTER_FILE_NAME[:-4] + '_HIdr2.txt', 
#                   format='ascii.commented_header', 
#                   overwrite=True)
# master_table.write(MASTER_FILE_NAME[:-5] + '_HIdr3.fits', format='fits', overwrite=True)

master_table.write(DATA_DIRECTORY + 'output_files/DR17/CURRENT_MASTER_TABLE/Elliptical_sphdisk_refitspirals_BPT_v8.fits', format='fits', overwrite=True)
###############################################################################