'''
Estimate the mass of elliptical galaxies
'''


################################################################################
#
# IMPORT MODULES
#
#-------------------------------------------------------------------------------

from read_data import read_master_file

################################################################################



################################################################################
#
# DATA FILENAMES
#
#-------------------------------------------------------------------------------

filename_master = 'master_file_vflag_10_smooth2-27_N2O2_noWords.txt'

data_directory = '/Users/kellydouglass/Documents/Research/Rotation_Curves/RotationCurves/manga_files/MaNGA_DR16/HYB10-GAU-MILESHC/'

################################################################################



################################################################################
#
# GALAXIES TO ANALYZE
#
# Select the individual galaxy to analyze.  If the galaxy ID number is set to 
# all, then all elliptical galaxies will be analyzed.
#-------------------------------------------------------------------------------

galaxy_ID = 'all'

################################################################################



################################################################################
#
# ANALYZE GALAXIES
#
#-------------------------------------------------------------------------------

elliptical_masses(filename_master, data_directory, galaxy_ID)