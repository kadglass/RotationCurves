import numpy as np



###############################################################################
###############################################################################
###############################################################################



def match_index( master_table, ref_table):
    '''
    Matches the ref_table to the master_table via NSA_plate, NSA_MJD, and
    NSA_fiberID, then extracts the KIAS-VAGC index from the ref_table and 
    assigns it to the corresponding entry in the master_table.

    Parameters:
    ===========

        master_table : astropy QTable
            Contains all summary information for MaNGA galaxies

        ref_table : astropy table
            Contains KIAS-VAGC information


    Returns:
    ========

        master_table : astropy QTable
            Updated version of master_table containing the extracted KIAS-VAGC 
            index
    '''

    ###########################################################################
    # Initialize output columns
    #--------------------------------------------------------------------------
    master_table['index'] = -1
    #--------------------------------------------------------------------------


    ###########################################################################
    # Build dictionary of tuples for storing galaxies with KIAS-VAGC indices
    #--------------------------------------------------------------------------
    ref_dict = galaxies_dict(ref_table)
    #--------------------------------------------------------------------------


    ###########################################################################
    # Match galaxies with metallicities
    #--------------------------------------------------------------------------
    for i in range( len( master_table)):
        
        plate = master_table['NSA_plate'][i]
        MJD = master_table['NSA_MJD'][i]
        fiberID = master_table['NSA_fiberID'][i]

        galaxy_ID = (plate, MJD, fiberID)

        if galaxy_ID in ref_dict.keys():
            master_table['index'][i] = ref_table['index'][ref_dict[galaxy_ID]]
        else:
            print("NO MATCH FOUND FOR GALAXY",
                  str( master_table['MaNGA_plate'][i]) + '-' + str( master_table['MaNGA_fiberID'][i]))
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

        plate = ref_table['plate'][idx]
        MJD = ref_table['MJD'][idx]
        fiberID = ref_table['fiberID'][idx]

        galaxy_ID = (plate, MJD, fiberID)

        ref_dict[galaxy_ID] = idx

    return ref_dict