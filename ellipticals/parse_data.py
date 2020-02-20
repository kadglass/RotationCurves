
################################################################################
# Import modules
#-------------------------------------------------------------------------------
import numpy as np

from IO_data import read_master_file
################################################################################



################################################################################
#-------------------------------------------------------------------------------
def build_galaxy_IDs(galaxy_ID, master_filename):
    '''
    Build a list of galaxy ID tuples to be analayzed.


    PARAMETERS
    ==========

    galaxy_ID : string
        Identifies what is to be analyzed.  If 'all', then analyze all 
        elliptical galaxies.  If not 'all', then is a single 'plate-fiberID' 
        galaxy combination.

    master_filename : string
        File name of master data table containing all MaNGA galaxies, their 
        NSA data, and additional calculated parameters.


    RETURNS
    =======

    galaxy_ID_list : length-N list of tuples
        List of (plate, fiberID) combinations for the galaxies to be analyzed. 
    '''


    if galaxy_ID is 'all':
        # Analyze all the elliptical galaxies in MaNGA

        elliptical_IDs = find_ellipticals(master_filename)

    else:
        # Analyze the single elliptical galaxy identified in galaxy_ID

        plate, fiberID = galaxy_ID.split('-')

        elliptical_IDs = [(plate, fiberID)]


    return elliptical_IDs
################################################################################






################################################################################
#-------------------------------------------------------------------------------
def remove_plate(master_table, master_boolean, plate, IFU_keep=-1, IFU_remove=-1):
    '''
    Remove objects from plate
    
    These objects were not analyzed with the DRP pipeline, but were analyzed in 
    the pipe3d pipeline.


    PARAMETERS
    ==========

    master_table : astropy table of length N
        Table of galaxy data

    master_boolean : numpy boolean array of shape (N,)
        Array of boolean values marking galaxies to be analayzed.  True values
        correspond to galaxies that will eventually be analyzed.

    plate : integer
        MaNGA plate number to remove from sample

    IFU_keep : integer
        MaNGA IFU number of plate to keep.  If IFU_keep == -1 (default), then 
        remove all objects on that plate.

    IFU_remove : integer
        MaNGA IFU number of plate to remove.  If IFU_remove == -1 (default), 
        then remove all objects on that plate.


    RETURNS
    =======

    analyze_boolean : numpy boolean array of shape (N,)
        master_boolean with those galaxies matching plate but NOT IFU_keep set 
        to False (so that they will not be analyzed).
    '''


    ############################################################################
    # Only remove one galaxy
    #---------------------------------------------------------------------------
    if IFU_remove > 0:

        plate_boolean = master_table['MaNGA_plate'] == plate
        IFU_boolean = master_table['MaNGA_fiberID'] == IFU_remove

        remove_boolean = np.logical_and(plate_boolean, IFU_boolean)
        remove_boolean = np.logical_not(remove_boolean)
    ############################################################################

    else:

        ########################################################################
        # Remove entire plate
        #-----------------------------------------------------------------------
        remove_boolean = master_table['MaNGA_plate'] != plate
        ########################################################################


        ########################################################################
        # Keep one of the objects in the plate
        #-----------------------------------------------------------------------
        if IFU_keep > 0:

            # Invert plate_boolean array
            plate_boolean = np.logical_not(remove_boolean)

            IFU_boolean = master_table['MaNGA_fiberID'] == IFU_keep

            # Locate which galaxy (plate-IFU_keep) to keep
            keep_plate_boolean = np.logical_and(plate_boolean, IFU_boolean)

            # switch plate-IFU_keep galaxy to True (so that it will be analyzed)
            remove_boolean[keep_plate_boolean] = True
        ########################################################################


    ############################################################################
    # Update master boolean
    #---------------------------------------------------------------------------
    analyze_boolean = np.logical_and(master_boolean, remove_boolean)
    ############################################################################

    return analyze_boolean
################################################################################






################################################################################
#-------------------------------------------------------------------------------
def find_ellipticals(master_filename):
    '''
    Find the elliptical galaxies in the master data table.  Elliptical galaxies 
    are defined as those with smoothness scores greater than 2.27 and/or 
    chi^2 > 10


    PARAMETERS
    ==========

    master_filename : string
        File name of master data table containing all MaNGA galaxies, their 
        NSA data, and additional calculated parameters.


    RETURNS
    =======
    
    elliptical_IDs : list of length-2 tuples
        All (plate, fiberID) combinations for the elliptical galaxies
    '''

    ############################################################################
    # Read in master file
    #---------------------------------------------------------------------------
    master_table = read_master_file(master_filename)
    ############################################################################


    ############################################################################
    # Locate ellipticals
    #---------------------------------------------------------------------------
    smooth_boolean = master_table['smoothness_score'] > 2.27

    pos_chi2_boolean = master_table['avg_chi_square_rot'] > 10
    neg_chi2_boolean = master_table['neg_chi_square_rot'] > 10
    avg_chi2_boolean = master_table['avg_chi_square_rot'] > 10
    chi2_boolean = np.logical_and.reduce([pos_chi2_boolean, 
                                          neg_chi2_boolean, 
                                          avg_chi2_boolean])

    elliptical_boolean = np.logical_and(smooth_boolean, chi2_boolean)
    ############################################################################


    ############################################################################
    # Remove objects that were not analyzed with the DRP pipeline, but were 
    # analyzed in the pipe3d pipeline.
    #---------------------------------------------------------------------------
    elliptical_boolean = remove_plate(master_table, elliptical_boolean, 7443, IFU_remove=3703)
    elliptical_boolean = remove_plate(master_table, elliptical_boolean, 7444)
    elliptical_boolean = remove_plate(master_table, elliptical_boolean, 8140, IFU_remove=6101)
    elliptical_boolean = remove_plate(master_table, elliptical_boolean, 8479, IFU_keep=3703)
    elliptical_boolean = remove_plate(master_table, elliptical_boolean, 8480, IFU_keep=3701)
    elliptical_boolean = remove_plate(master_table, elliptical_boolean, 8953, IFU_keep=3702)
    elliptical_boolean = remove_plate(master_table, elliptical_boolean, 8993, IFU_remove=1901)
    elliptical_boolean = remove_plate(master_table, elliptical_boolean, 9051, IFU_keep=6103)
    elliptical_boolean = remove_plate(master_table, elliptical_boolean, 9888, IFU_remove=9102)
    ############################################################################


    ############################################################################
    # Build elliptical IDs
    #---------------------------------------------------------------------------
    elliptical_plates = master_table['MaNGA_plate'][elliptical_boolean]
    elliptical_fiberIDs = master_table['MaNGA_fiberID'][elliptical_boolean]

    elliptical_IDs = list(zip(elliptical_plates.astype('str'), 
                              elliptical_fiberIDs.astype('str')))
    ############################################################################

    return elliptical_IDs
################################################################################





################################################################################
#-------------------------------------------------------------------------------
def find_data_DRPall(data_table, ID, field_name):
    '''
    Find field_name value in data table


    PARAMETERS
    ==========

    data_table : fits table
        Data table with stellar mass values, plate-ifu IDs

    ID : length-2 tuple
        (plate, fiberID) of querying galaxy

    field_name : string
        Data field from which to extract value


    RETURNS
    =======

    data_value : float
        field_name value of galaxy
    '''


    ############################################################################
    # Find galaxy in table
    #---------------------------------------------------------------------------
    idx_boolean = data_table['plateifu'] == '-'.join(ID)
    ############################################################################


    ############################################################################
    # Extract field_name value from table
    #---------------------------------------------------------------------------
    data_value = data_table[field_name][idx_boolean]
    ############################################################################

    return data_value
################################################################################