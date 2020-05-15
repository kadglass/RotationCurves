
import os.path

import numpy as np
import numpy.ma as ma

import astropy.units as u
import astropy.constants as const

import math


################################################################################
# Constants
#-------------------------------------------------------------------------------
H_0 = 100 * (u.km / ( u.s * u.Mpc))  # Hubble's Constant in units of km /s /Mpc

MANGA_SPAXEL_SIZE = 0.5*(1/60)*(1/60)*(np.pi/180) # spaxel size (0.5") in radians
################################################################################



################################################################################
################################################################################

def find_Mturn_Pipe3D( gal_ID, Rturn, phi_EofN_deg, axis_ratio, z):
    '''
    Read in the galaxy's stellar mass map and extract the mass within an ellipse 
    equal to the unprojected circle with radius Rturn concentric with the 
    galaxy's center.


    PARAMETERS
    ==========

    gal_ID : string
        MaNGA galaxy ID: <plate> - <IFU>

    Rturn : float
        Fitted Rturn parameter to the rotation curve (proxy for the bulge's 
        radius).  Units are kpc.

    phi_EofN_deg : float
        Angle (east of north) of rotation in the 2D observational plane
        NOTE: East is 'left' per astronomy convention.

    axis_ratio : float
        b/a axis ratio for galaxy

    z : float
        Galaxy redshift


    RETURNS
    =======

    Mturn : float with units
        Stellar mass contained within the radius Rturn in the galaxy.  Units are 
        log10( solar masses).
    '''

    ############################################################################
    # Import data processing functions
    #---------------------------------------------------------------------------
    from Pipe3D_rotation_curve import extract_data
    from Pipe3D_rotation_curve_functions import build_mask, calc_stellar_mass

    LOCAL_PATH = os.path.dirname(__file__)
    if LOCAL_PATH == '':
        LOCAL_PATH = './'
    else:
        LOCAL_PATH = LOCAL_PATH + '/'

    MANGA_FOLDER = LOCAL_PATH + '../data/MaNGA/MaNGA_DR15/pipe3d/'
    ############################################################################


    ###########################################################################
    # The gal_id name scheme is [PLATE]-[FIBER ID].
    #--------------------------------------------------------------------------
    manga_plate, manga_IFU = gal_ID.split('-')

    file_name = MANGA_FOLDER + manga_plate + '/manga-' + gal_ID + '.Pipe3D.cube.fits.gz'
    ###########################################################################


    ############################################################################
    # Read in galaxy data maps
    #---------------------------------------------------------------------------
    _, _, _, Ha_vel_err, v_band, v_band_err, sMass_density, _, _ = extract_data( file_name)
    ############################################################################


    ############################################################################
    # Create data mask and mask stellar mass map
    #---------------------------------------------------------------------------
    data_mask = build_mask( Ha_vel_err, v_band, v_band_err, sMass_density)

    masked_sMass_density = ma.masked_where( data_mask, sMass_density)
    masked_v_band = ma.masked_where( data_mask, v_band)
    ############################################################################


    ############################################################################
    # Determine optical center via the max luminosity in the visual band.
    #---------------------------------------------------------------------------
    optical_center = np.argwhere( masked_v_band.max() == masked_v_band)
    ############################################################################


    ############################################################################
    # Convert pixel distance to physical distances in units of both
    # kiloparsecs and centimeters.
    #---------------------------------------------------------------------------
    dist_to_galaxy_kpc = ( z * const.c.to('km/s') / H_0).to('kpc')

    pix_scale_factor = dist_to_galaxy_kpc * np.tan( MANGA_SPAXEL_SIZE)
    ############################################################################


    ############################################################################
    # Create a meshgrid for all coordinate points based on the dimensions of
    # the stellar mass numpy array.
    #---------------------------------------------------------------------------
    array_length = masked_sMass_density.shape[0]  # y-coordinate distance
    array_width = masked_sMass_density.shape[1]  # x-coordinate distance

    X_RANGE = np.arange(0, array_width, 1)
    Y_RANGE = np.arange(0, array_length, 1)
    X_COORD, Y_COORD = np.meshgrid( X_RANGE, Y_RANGE)
    ############################################################################


    ############################################################################
    # Initialization code to draw the elliptical annuli and to normalize the
    #    2D-arrays for the max and min velocity so as to check for anomalous
    #    data.
    #---------------------------------------------------------------------------
    phi_elip = math.radians( 90 - ( phi_EofN_deg / u.deg)) * u.rad

    x_diff = X_COORD - optical_center[0][1]
    y_diff = Y_COORD - optical_center[0][0]

    ellipse = ( x_diff*np.cos( phi_elip) - y_diff*np.sin( phi_elip))**2 + \
              ( x_diff*np.sin( phi_elip) + y_diff*np.cos( phi_elip))**2 / \
              ( axis_ratio)**2
    ############################################################################


    ############################################################################
    # Locate points in map that are within ellipse bounds
    #---------------------------------------------------------------------------
    # We need to deproject the turn radius to match the pixel distances
    R = Rturn / pix_scale_factor

    pix_within_ellipse = ellipse < R
    ############################################################################


    ############################################################################
    # Calculate mass within ellipse
    #---------------------------------------------------------------------------
    Mturn = calc_stellar_mass(0*u.M_sun, masked_sMass_density, pix_within_ellipse)
    ############################################################################


    return Mturn
################################################################################
################################################################################






################################################################################
################################################################################

def find_Mturn_DRP( gal_ID, Rturn):
    '''
    Read in the galaxy's stellar mass map and extract the mass within an ellipse 
    equal to the unprojected circle with radius Rturn concentric with the 
    galaxy's center.


    PARAMETERS
    ==========

    gal_ID : string
        MaNGA galaxy ID: <plate> - <IFU>

    Rturn : float
        Fitted Rturn parameter to the rotation curve (proxy for the bulge's 
        radius).  Units are kpc.


    RETURNS
    =======

    Mturn : float with units
        Stellar mass contained within the radius Rturn in the galaxy.  Units are 
        log10( solar masses).
    '''

    ############################################################################
    # Import correct data processing functions for corresponding data pipeline
    #---------------------------------------------------------------------------
    from DRP_rotation_curve import extract_data, extract_Pipe3D_data
    from DRP_rotation_curve_functions import build_mask, calc_stellar_mass

    LOCAL_PATH = os.path.dirname(__file__)
    if LOCAL_PATH == '':
        LOCAL_PATH = './'

    MANGA_FOLDER = LOCAL_PATH + '../data/MaNGA/MaNGA_DR16/HYB10-GAU-MILESHC/'
    PIPE3D_folder = LOCAL_PATH + '../data/MaNGA/MaNGA_DR15/pipe3d/'
    ############################################################################


    ###########################################################################
    # The gal_id name scheme is [PLATE]-[FIBER ID].
    #--------------------------------------------------------------------------
    manga_plate, manga_IFU = gal_ID.split('-')

    file_name = MANGA_FOLDER + manga_plate + '/manga-' + gal_ID + '-MAPS-HYB10-GAU-MILESHC.fits.gz'
    ###########################################################################


    ############################################################################
    # Read in galaxy data maps
    #---------------------------------------------------------------------------
    _, _, Ha_vel_mask, r_band, _ = extract_data( file_name)
    sMass_density = extract_Pipe3d_data( PIPE3D_folder, gal_ID)
    ############################################################################


    ############################################################################
    # Create data mask and mask stellar mass map
    #---------------------------------------------------------------------------
    data_mask = build_mask( Ha_vel_mask, sMass_density)

    masked_sMass_density = ma.array( sMass_density, mask=data_mask)
    mr_band = ma.array( r_band, mask=data_mask)
    ############################################################################


    ############################################################################
    # Determine optical center via the max luminosity in the r-band.
    #---------------------------------------------------------------------------
    optical_center = np.unravel_index( ma.argmax(mr_band, axis=None), mr_band.shape)
    ############################################################################


    ############################################################################
    # Convert pixel distance to physical distances in units of both
    # kiloparsecs and centimeters.
    #---------------------------------------------------------------------------
    dist_to_galaxy_kpc = ( z * const.c.to('km/s') / H_0).to('kpc')

    pix_scale_factor = dist_to_galaxy_kpc * np.tan( MANGA_SPAXEL_SIZE)
    ############################################################################


    ############################################################################
    # Create a meshgrid for all coordinate points based on the dimensions of
    # the stellar mass numpy array.
    #---------------------------------------------------------------------------
    array_length = masked_sMass_density.shape[0]  # y-coordinate distance
    array_width = masked_sMass_density.shape[1]  # x-coordinate distance

    X_RANGE = np.arange(0, array_width, 1)
    Y_RANGE = np.arange(0, array_length, 1)
    X_COORD, Y_COORD = np.meshgrid( X_RANGE, Y_RANGE)
    ############################################################################


    ############################################################################
    # Initialization code to draw the elliptical annuli and to normalize the
    #    2D-arrays for the max and min velocity so as to check for anomalous
    #    data.
    #---------------------------------------------------------------------------
    phi_elip = math.radians( 90 - ( phi_EofN_deg / u.deg)) * u.rad

    x_diff = X_COORD - optical_center[1]
    y_diff = Y_COORD - optical_center[0]

    ellipse = ( x_diff*np.cos( phi_elip) - y_diff*np.sin( phi_elip))**2 + \
              ( x_diff*np.sin( phi_elip) + y_diff*np.cos( phi_elip))**2 / \
              ( axis_ratio)**2
    ############################################################################


    ############################################################################
    # Locate points in map that are within ellipse bounds
    #---------------------------------------------------------------------------
    # We need to deproject the turn radius to match the pixel distances
    R = Rturn / pix_scale_factor

    pix_within_ellipse = ellipse < R
    ############################################################################


    ############################################################################
    # Calculate mass within ellipse
    #---------------------------------------------------------------------------
    Mturn = calc_stellar_mass(0*u.M_sun, masked_sMass_density, pix_within_ellipse)
    ############################################################################


    return Mturn
################################################################################
################################################################################






################################################################################
################################################################################

def extract_Mturn( master_table, DATA_PIPELINE):
    '''
    Locate the stellar mass within the Rturn radius for each galaxy.


    PARAMETERS
    ==========

    master_table : astropy QTable
        Data table with N rows, each row containing one MaNGA galaxy for which 
        the rotation curve has been measured.

    DATA_PIPELINE : string
        Name of the data pipeline used in the analysis.


    RETURNS
    =======

    master_table : astropy QTable
        Same as the input master_table object, but with the additional Mturn 
        columns containing the stellar mass contained within the radius Rturn 
        for each of the three rotation curves (positive, negative, and average).  
        Each column is in units of solar masses.
    '''

    ############################################################################
    # Initialize Mturn columns in master_table
    #---------------------------------------------------------------------------
    master_table['Mstar_turn'] = -1*np.ones(len(master_table), dtype=float) * u.M_sun
    ############################################################################


    for i in range(len(master_table)):

        ########################################################################
        # Build galaxy ID
        #-----------------------------------------------------------------------
        plate = master_table['MaNGA_plate'][i]
        IFU = master_table['MaNGA_IFU'][i]

        gal_ID = str(plate) + '-' + str(IFU)
        ########################################################################


        ########################################################################
        # Find Mturn for the galaxy
        #-----------------------------------------------------------------------
        curve = master_table['curve_used'][i]

        if curve != 'none' and curve != 'non':

            # Galaxy properties from the master table
            Rturn = master_table[curve + '_r_turn'][i]
            phi = master_table['NSA_phi'][i]
            axis_ratio = master_table['NSA_ba'][i]
            z = master_table['NSA_redshift'][i]

            if master_table['frac_masked_spaxels'][i] < 1 and Rturn.value != -1:
                if DATA_PIPELINE == 'Pipe3D':
                    master_table['Mstar_turn'][i] = find_Mturn_Pipe3D( gal_ID, 
                                                                       Rturn, 
                                                                       phi, 
                                                                       axis_ratio, 
                                                                       z)
                elif DATA_PIPELINE == 'DRP':
                    master_table['Mstar_turn'][i] = find_Mturn_DRP( gal_ID, 
                                                                    Rturn, 
                                                                    phi, 
                                                                    axis_ratio, 
                                                                    z)
                else:
                    print('Data pipeline uknown.')
                    exit()
        ########################################################################

    return master_table
################################################################################
################################################################################