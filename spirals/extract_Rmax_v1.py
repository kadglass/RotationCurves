
import numpy as np
import numpy.ma as ma

import astropy.units as u
from astropy.table import QTable

from DRP_vel_map_functions import build_map_mask, deproject_spaxel

from DRP_rotation_curve import extract_data


################################################################################
# Constants
#-------------------------------------------------------------------------------
H_0 = 100      # Hubble's Constant in units of h km/s/Mpc
c = 299792.458 # Speed of light in units of km/s

MANGA_SPAXEL_SIZE = 0.5*(1/60)*(1/60)*(np.pi/180)  # spaxel size (0.5") in radians
################################################################################




################################################################################
################################################################################
def find_Rmax( gal_ID, DATA_PIPELINE):
    '''
    Read in the data file built during the MaNGA rotation curve analysis, and 
    extract the last deprojected radius analyzed.


    PARAMETERS
    ==========

    gal_ID : string
        MaNGA galaxy ID: <plate> - <IFU>

    DATA_PIPELINE : string
        Name of the data pipeline used in the analysis.


    RETURNS
    =======

    Rmax : float with units
        Maximum analyzed deprojected radius for the given galaxy.  Units are 
        kpc.
    '''


    ############################################################################
    # Read in galaxy rotation curve data file
    #---------------------------------------------------------------------------
    rot_curve_data_filename = DATA_PIPELINE + '-rot_curve_data_files/' + \
                              gal_ID + '_rot_curve_data.txt'

    rot_curve_data = QTable.read(rot_curve_data_filename, format='ascii.ecsv')
    ############################################################################


    ############################################################################
    # Find last measured deprojected radius analyzed
    #---------------------------------------------------------------------------
    Rmax = rot_curve_data['deprojected_distance'][-1]
    ############################################################################


    return Rmax
################################################################################
################################################################################







################################################################################
################################################################################
def find_Rmax_map(gal_ID, gal_info, VEL_MAP_FOLDER):
    '''
    Read in the data file built during the MaNGA rotation curve analysis, and 
    extract the last deprojected radius analyzed.


    PARAMETERS
    ==========

    gal_ID : string
        MaNGA galaxy ID: <plate> - <IFU>

    gal_info : row of astropy table
        Contains all of the best-fit values and metadata for the current galaxy

    VEL_MAP_FOLDER : string
        Name of the data pipeline used in the analysis.


    RETURNS
    =======

    Rmax : float with units
        Maximum analyzed deprojected radius for the given galaxy.  Units are 
        kpc.
    '''


    ############################################################################
    # Read in necessary data maps
    #---------------------------------------------------------------------------
    Ha_vel, _, Ha_vel_mask, _, _, Ha_flux, Ha_flux_ivar, Ha_flux_mask, Ha_sigma, _, Ha_sigma_mask = extract_data( VEL_MAP_FOLDER, gal_ID)
    ############################################################################


    ############################################################################
    # Build galaxy map mask
    #---------------------------------------------------------------------------
    map_mask = build_map_mask(gal_ID, 
                              gal_info['map_fit_flag'], 
                              ma.array(Ha_vel, mask=Ha_vel_mask), 
                              ma.array(Ha_flux, mask=Ha_flux_mask), 
                              ma.array(Ha_flux_ivar, mask=Ha_flux_mask), 
                              ma.array(Ha_sigma, mask=Ha_sigma_mask))
    ############################################################################


    ############################################################################
    # Convert pixel distance to physical distances.
    #---------------------------------------------------------------------------
    dist_to_galaxy_Mpc = c*gal_info['NSA_redshift']/H_0
    dist_to_galaxy_kpc = dist_to_galaxy_Mpc*1000

    pix_scale_factor = dist_to_galaxy_kpc*np.tan(MANGA_SPAXEL_SIZE)
    ############################################################################


    ############################################################################
    # Deproject all data values in the given velocity map
    #---------------------------------------------------------------------------
    vel_array_shape = map_mask.shape

    r_deproj = np.zeros(vel_array_shape)

    for i in range(vel_array_shape[0]):
        for j in range(vel_array_shape[1]):

            r_deproj[i,j], _ = deproject_spaxel((i,j), 
                                                (gal_info['x0_map'], gal_info['y0_map']), 
                                                gal_info['phi_map']*np.pi/180, 
                                                np.arccos(gal_info['ba_map']))

    # Scale radii to convert from spaxels to kpc
    r_deproj *= pix_scale_factor

    # Find maximum deprojected radius
    Rmax = ma.max(ma.array(r_deproj, mask=map_mask))
    ############################################################################


    return Rmax
################################################################################
################################################################################







################################################################################
################################################################################
def match_Rmax( master_table):
    '''
    Locate the maximum deprojected data radius in the velocity map for each 
    galaxy.


    PARAMETERS
    ==========

    master_table : astropy QTable
        Data table with N rows, each row containing one MaNGA galaxy for which 
        the velocity map has been fit.

    DATA_PIPELINE : string
        Name of the data pipeline used in the analysis.


    RETURNS
    =======

    master_table : astropy QTable
        Same as the input master_table object, but with the additional Rmax 
        column containing the maximum deprojected radius analyzed for each 
        velocity map.
    '''


    ############################################################################
    # Initialize Rmax column in master_table
    #---------------------------------------------------------------------------
    master_table['Rmax'] = np.zeros(len(master_table), dtype=float) * u.kpc
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
        # Find Rmax for the galaxy
        #-----------------------------------------------------------------------
        master_table['Rmax_map'][i] = find_Rmax_map(gal_ID, DATA_PIPELINE)
        ########################################################################

    return master_table
################################################################################
################################################################################







################################################################################
################################################################################
def match_Rmax_map( master_table, VEL_MAP_FOLDER):
    '''
    Locate the maximum deprojected data radius in the velocity map for each 
    galaxy.


    PARAMETERS
    ==========

    master_table : astropy QTable
        Data table with N rows, each row containing one MaNGA galaxy for which 
        the velocity map has been fit.

    VEL_MAP_FOLDER : string
        Directory of the velocity maps


    RETURNS
    =======

    master_table : astropy QTable
        Same as the input master_table object, but with the additional Rmax 
        column containing the maximum deprojected radius analyzed for each 
        velocity map.
    '''


    ############################################################################
    # Initialize Rmax column in master_table
    #---------------------------------------------------------------------------
    master_table['Rmax_map'] = np.zeros(len(master_table), dtype=float) #* u.kpc
    ############################################################################


    for i in range(len(master_table)):

        if master_table['M90_map'][i] > 0:
            ####################################################################
            # Build galaxy ID
            #-------------------------------------------------------------------
            plate = master_table['MaNGA_plate'][i]
            IFU = master_table['MaNGA_IFU'][i]

            gal_ID = str(plate) + '-' + str(IFU)
            ####################################################################


            ####################################################################
            # Find Rmax for the galaxy
            #-------------------------------------------------------------------
            master_table['Rmax_map'][i] = find_Rmax_map(gal_ID, 
                                                        master_table[i], 
                                                        VEL_MAP_FOLDER)
            ####################################################################

    return master_table
################################################################################
################################################################################







