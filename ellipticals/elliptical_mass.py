'''
Estimate the mass of elliptical galaxies
'''


################################################################################
# IMPORT MODULES
#-------------------------------------------------------------------------------

from elliptical_functions import FundamentalPlane, FaberJackson, elliptical_masses

################################################################################





################################################################################
# DATA FILENAMES
#-------------------------------------------------------------------------------
filename_master = '../spirals/DRPall-master_file.txt'

data_directory = '/Users/kellydouglass/Documents/Research/data/SDSS/dr16/manga/spectro/analysis/v2_4_3/2.2.1/HYB10-GAU-MILESHC/'
################################################################################





################################################################################
# GALAXIES TO ANALYZE
#
# Select the individual galaxy to analyze.  If the galaxy ID number is set to 
# all, then all elliptical galaxies will be analyzed.
#-------------------------------------------------------------------------------

#galaxy_ID = '9031-3701'
galaxy_ID = 'all'

################################################################################





################################################################################
# ANALYZE GALAXIES
#-------------------------------------------------------------------------------

#FundamentalPlane(galaxy_ID, data_directory, filename_master, 'median', save_fig=False)
FaberJackson(galaxy_ID, data_directory, filename_master, 'median', save_fig=False)
#elliptical_masses(galaxy_ID, data_directory, filename_master)

################################################################################