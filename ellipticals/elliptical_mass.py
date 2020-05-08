'''
Estimate the mass of elliptical galaxies
'''


################################################################################
# IMPORT MODULES
#-------------------------------------------------------------------------------

from elliptical_functions import FaberJackson, elliptical_masses

################################################################################





################################################################################
# DATA FILENAMES
#-------------------------------------------------------------------------------

filename_master = '../spirals/master_file_30.txt'

data_directory = '/Users/kellydouglass/Documents/Research/Rotation_Curves/RotationCurves/data/MaNGA/MaNGA_DR16/HYB10-GAU-MILESHC/'

################################################################################





################################################################################
# GALAXIES TO ANALYZE
#
# Select the individual galaxy to analyze.  If the galaxy ID number is set to 
# all, then all elliptical galaxies will be analyzed.
#-------------------------------------------------------------------------------

galaxy_ID = '9031-3701'
#galaxy_ID = 'all'

################################################################################





################################################################################
# ANALYZE GALAXIES
#-------------------------------------------------------------------------------

#FaberJackson(galaxy_ID, data_directory, filename_master, 'median', save=True)
elliptical_masses(galaxy_ID, data_directory, filename_master)

################################################################################