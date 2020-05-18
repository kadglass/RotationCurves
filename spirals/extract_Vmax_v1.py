
import numpy as np

import astropy.units as u
from astropy.table import QTable



################################################################################
################################################################################
def find_Vmax( gal_ID, DATA_PIPELINE):
    '''
    Read in the data file built during the MaNGA rotation curve analysis, and 
    extract the maximum velocity measured.


    PARAMETERS
    ==========

    gal_ID : string
        MaNGA galaxy ID: <plate> - <IFU>

    DATA_PIPELINE : string
        Name of the data pipeline used in the analysis.


    RETURNS
    =======

    pos_Vmax : float with units
        Maximum measured velocity in the positive rotation curve for the given 
        galaxy.  Units are km/s.

    avg_Vmax : float with units
        Maximum measured velocity in the average rotation curve for the given 
        galaxy.  Units are km/s.

    neg_Vmax : float with units
        Maximum measured velocity in the negative rotation curve for the given 
        galaxy.  Units are km/s.
    '''


    ############################################################################
    # Read in galaxy rotation curve data file
    #---------------------------------------------------------------------------
    rot_curve_data_filename = DATA_PIPELINE + '-rot_curve_data_files/' + \
                              gal_ID + '_rot_curve_data.txt'

    rot_curve_data = QTable.read(rot_curve_data_filename, format='ascii.ecsv')
    ############################################################################


    ############################################################################
    # Find maximum measured velocities
    #---------------------------------------------------------------------------
    pos_Vmax = rot_curve_data['max_velocity'].max()
    avg_Vmax = rot_curve_data['rot_vel_avg'].max()
    neg_Vmax = rot_curve_data['min_velocity'].max()
    ############################################################################


    return pos_Vmax, avg_Vmax, neg_Vmax
################################################################################
################################################################################







################################################################################
################################################################################
def match_Vmax( master_table, DATA_PIPELINE):
    '''
    Locate the maximum measured velocity in each rotation curve for each galaxy.


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
        Same as the input master_table object, but with the additional Vmax 
        columns containing the maximum radii analyzed for each rotation curve.
    '''


    ############################################################################
    # Initialize Vmax columns in master_table
    #---------------------------------------------------------------------------
    master_table['pos_Vmax_data'] = -1*np.ones(len(master_table), dtype=float) * (u.km / u.s)
    master_table['avg_Vmax_data'] = -1*np.ones(len(master_table), dtype=float) * (u.km / u.s)
    master_table['neg_Vmax_data'] = -1*np.ones(len(master_table), dtype=float) * (u.km / u.s)
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
        master_table['pos_Vmax_data'][i], master_table['avg_Vmax_data'][i], master_table['neg_Vmax_data'][i] = find_Vmax( gal_ID, DATA_PIPELINE)
        ########################################################################

    return master_table
################################################################################
################################################################################







