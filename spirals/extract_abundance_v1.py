import numpy as np



###############################################################################
###############################################################################
###############################################################################



def match_abundance( master_table, Z_ref_table, method=None):
    '''
    Matches the Z_ref_table to the master_table via NSA_plate, NSA_MJD, and
    NSA_fiberID, then extracts the metallicity from the Z_ref_table and assigns 
    it to the corresponding entry in the master_table.

    Parameters:
    ===========

    master_table : astropy QTable
        Contains all summary information for MaNGA galaxies

    Z_ref_table : astropy table
        Contains metallicity information

    method : string
        Abundance method used.  Default is None (field name will be 'Z12logOH').


    Returns:
    ========

        master_table : astropy QTable
            Updated version of master_table containing the extracted metallicity
    '''

    ###########################################################################
    # Initialize output columns
    #--------------------------------------------------------------------------
    abund_field = 'Z12logOH'

    if method is not None:
        abund_field = 'Z12logOH_' + method

    master_table[abund_field] = np.nan
    #--------------------------------------------------------------------------


    ###########################################################################
    # Build dictionary of tuples for storing galaxies with KIAS-VAGC indices
    #--------------------------------------------------------------------------
    ref_dict = galaxies_dict(Z_ref_table)
    #--------------------------------------------------------------------------


    ###########################################################################
    # Match galaxies with metallicities
    #--------------------------------------------------------------------------
    for i in range( len( master_table)):
        
        index = master_table['index'][i]

        galaxy_ID = (index)

        if galaxy_ID in ref_dict.keys():
            master_table[abund_field][i] = Z_ref_table['Z12logOH'][ref_dict[galaxy_ID]]
        else:
            print("NO MATCHES FOUND FOR GALAXY",
                  str( master_table['MaNGA_plate'][i]) + '-' + str( master_table['MaNGA_IFU'][i]))
    #--------------------------------------------------------------------------

    return master_table




###############################################################################
###############################################################################
###############################################################################




def galaxies_dict(ref_table):
    '''
    Build a dictionary of the galaxies with KIAS-VAGC indices

    Parameters:
    ===========

    ref_table : astropy table
        Galaxies with KIAS-VAGC indices


    Returns:
    ========

    ref_dict : dictionary
        Tuples of (plate, MJD, fiber) for those galaxies with metallicities
    '''

    # Initialize dictionary of cell IDs with at least one galaxy in them
    ref_dict = {}

    for idx in range(len(ref_table)):

        galaxy_ID = (ref_table['index'][idx])

        ref_dict[galaxy_ID] = idx

    return ref_dict



