
from astropy.table import QTable
from astropy.io import fits

import sys
sys.path.insert(1, '/Users/kellydouglass/Documents/Research/Rotation_curves/RotationCurves/spirals/')
from DRP_rotation_curve import extract_data, extract_Pipe3d_data
from DRP_rotation_curve_functions import build_mask

from mapSmoothness_functions import how_smooth



data_directory = '/Users/kellydouglass/Documents/Research/data/SDSS/MaNGA/MaNGA_DR16/HYB10-GAU-MILESHC/'
PIPE3D_folder = '/Users/kellydouglass/Documents/Research/data/SDSS/MaNGA/MaNGA_DR15/pipe3d/'

################################################################################
# Galaxy data
#-------------------------------------------------------------------------------
galaxies_filename = 'spirals/DRPall-master_file.txt'

galaxies = QTable.read(galaxies_filename, format='ascii.ecsv')

galaxies['smoothness_score'] = -1.
################################################################################



################################################################################
# Calculate smoothness score
#-------------------------------------------------------------------------------
for i in range(len(galaxies)):

    ############################################################################
    # Build galaxy file name
    #---------------------------------------------------------------------------
    plate = galaxies['MaNGA_plate'][i]
    IFU = galaxies['MaNGA_IFU'][i]

    gal_ID = str(plate) + '-' + str(IFU)

    galaxy_fits_filename = data_directory + str(plate) + '/manga-' + gal_ID + '-MAPS-HYB10-GAU-MILESHC.fits.gz'
    ############################################################################


    ############################################################################
    # Read in data
    #---------------------------------------------------------------------------
    Ha_vel, _, Ha_vel_mask, _, _ = extract_data( galaxy_fits_filename)
    sMass_density = extract_Pipe3d_data( PIPE3D_folder, gal_ID)

    data_mask = build_mask( Ha_vel_mask, sMass_density)
    ############################################################################


    ############################################################################
    # Calculate degree of smoothness of velocity map
    #---------------------------------------------------------------------------
    galaxies['smoothness_score'][i] = how_smooth( Ha_vel, data_mask)
    ############################################################################
################################################################################



################################################################################
# Save results
#-------------------------------------------------------------------------------
galaxies.write(galaxies_filename, format='ascii.ecsv', overwrite=True)
################################################################################