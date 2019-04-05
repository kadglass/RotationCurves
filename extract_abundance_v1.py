import numpy as np



###############################################################################
###############################################################################
###############################################################################



def match_abundance( master_table, Z_ref_table):
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


    Returns:
    ========

        master_table : astropy QTable
            Updated version of master_table containing the extracted metallicity
    '''

    ###########################################################################
    # Initialize output columns
    #--------------------------------------------------------------------------
    master_table['t3'] = np.nan
    master_table['BPT'] = np.nan
    master_table['Z12logOH'] = np.nan
    master_table['logNO'] = np.nan
    #--------------------------------------------------------------------------


    ###########################################################################
    # Build dictionary of tuples for storing galaxies with metallicities
    #--------------------------------------------------------------------------
    Z_ref_dict = Z_galaxies_dict(Z_ref_table)
    #--------------------------------------------------------------------------


    ###########################################################################
    # Match galaxies with metallicities
    #--------------------------------------------------------------------------
    for i in range( len( master_table)):
        
        plate = master_table['NSA_plate'][i]
        MJD = master_table['NSA_MJD'][i]
        fiberID = master_table['NSA_fiberID'][i]

        galaxy_ID = (plate, MJD, fiberID)

        if galaxy_ID in Z_ref_dict.keys():
            master_table['t3'][i] = Z_ref_table['t3'][Z_ref_dict[galaxy_ID]]
            master_table['BPT'][i] = Z_ref_table['BPTclass'][Z_ref_dict[galaxy_ID]]
            master_table['Z12logOH'][i] = Z_ref_table['Z12logOH'][Z_ref_dict[galaxy_ID]]
            master_table['logNO'][i] = Z_ref_table['logNO'][Z_ref_dict[galaxy_ID]]
        else:
            print("NO MATCHES FOUND FOR GALAXY",
                  str( master_table['MaNGA_plate'][i]) + '-' + str( master_table['MaNGA_fiberID'][i]))
    #--------------------------------------------------------------------------

    return master_table





###############################################################################
###############################################################################
###############################################################################




def Z_galaxies_dict(Z_table):
    '''
    Build a dictionary of the galaxies with metallicities

    Parameters:
    ===========

    Z_table : astropy table
        Galaxies with metallicities


    Returns:
    ========

    ref_dict : dictionary
        Tuples of (plate, MJD, fiber) for those galaxies with metallicities
    '''

    # Initialize dictionary of cell IDs with at least one galaxy in them
    ref_dict = {}

    for idx in range(len(Z_table)):

        plate = Z_table['plate'][idx]
        MJD = Z_table['MJD'][idx]
        fiberID = Z_table['fiberID'][idx]

        galaxy_ID = (plate, MJD, fiberID)

        ref_dict[galaxy_ID] = idx

    return ref_dict