
################################################################################
# Import modules
#-------------------------------------------------------------------------------
from read_data import read_master_file
################################################################################



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
    # Build elliptical IDs
    #---------------------------------------------------------------------------
    elliptical_plates = master_table['MaNGA_plate'][elliptical_boolean]
    elliptical_fiberIDs = master_table['MaNGA_fiberID'][elliptical_boolean]

    elliptical_IDs = list(zip(elliptical_plates.astype('str'), 
                              elliptical_fiberIDs.astype('str')))
    ############################################################################

    return elliptical_IDs

