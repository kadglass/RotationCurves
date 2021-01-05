
from astropy.table import QTable
from astropy.io import fits

import sys
sys.path.insert(1, '/Users/kellydouglass/Documents/Research/Rotation_curves/RotationCurves/spirals/')
from Pipe3D_rotation_curve import extract_data
from Pipe3D_rotation_curve_functions import build_mask

from mapSmoothness_functions import how_smooth


################################################################################
# Galaxy data
#-------------------------------------------------------------------------------
galaxies_filename = 'spirals/Pipe3D-master_file_vflag_BB_10.txt'

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

    galaxy_fits_filename = 'data/MaNGA/MaNGA_DR15/pipe3d/' + str(plate) + '/manga-' + str(plate) + '-' + str(IFU) + '.Pipe3D.cube.fits.gz'
    ############################################################################


    ############################################################################
    # Read in data
    #---------------------------------------------------------------------------
    _,_, Ha_vel, Ha_vel_err, v_band, v_band_err, sMass_density,_,_ = extract_data( galaxy_fits_filename)

    data_mask = build_mask( Ha_vel_err, v_band, v_band_err, sMass_density)
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
galaxies.write(galaxies_filename[:-4] + '_smooth.txt', format='ascii.ecsv', 
               overwrite=True)
################################################################################