
import numpy as np

import astropy.units as u
from astropy.table import QTable



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
def match_Rmax( master_table, DATA_PIPELINE):
    '''
    Locate the maximum data radius in the rotation curve for each galaxy.


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
        Same as the input master_table object, but with the additional Rmax 
        column containing the maximum radius analyzed for each rotation curve.
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
        master_table['Rmax'][i] = find_Rmax( gal_ID, DATA_PIPELINE)
        ########################################################################

    return master_table
################################################################################
################################################################################







