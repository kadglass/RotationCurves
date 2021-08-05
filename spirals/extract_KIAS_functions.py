

import numpy as np



###############################################################################
###############################################################################
###############################################################################



def PMJDF_match( master_table, ref_table, column_names):
    '''
    Match the ref_table to the master_table via plate, MJD, and fiberID, then 
    extract the KIAS-VAGC index and other given values from the ref_table and 
    assigns it to the corresponding entry in the master_table.

    Parameters:
    ===========

        master_table : astropy QTable
            Contains all summary information for MaNGA galaxies

        ref_table : astropy table
            Contains KIAS-VAGC information

        column_names : list
            Column names to add to master_table


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

    for col_name in column_names:
        master_table[col_name] = -1.
    #--------------------------------------------------------------------------


    ###########################################################################
    # Build dictionary of tuples for storing galaxies with KIAS-VAGC indices
    #--------------------------------------------------------------------------
    ref_dict = galaxies_dict(ref_table)
    #--------------------------------------------------------------------------


    ###########################################################################
    # Match galaxies in KIAS-VAGC
    #--------------------------------------------------------------------------
    for i in range( len( master_table)):
        
        plate = master_table['NSA_plate'][i]
        MJD = master_table['NSA_MJD'][i]
        fiberID = master_table['NSA_fiberID'][i]
        '''
        plate = master_table['plate'][i]
        MJD = master_table['MJD'][i]
        fiberID = master_table['fiberID'][i]
        '''
        galaxy_ID = (plate, MJD, fiberID)

        if galaxy_ID in ref_dict.keys():
            master_table['index'][i] = ref_table['index'][ref_dict[galaxy_ID]]

            for col_name in column_names:
                master_table[col_name][i] = ref_table[col_name][ref_dict[galaxy_ID]]
        else:
            print("NO MATCH FOUND FOR GALAXY",
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

        plate = ref_table['plate'][idx]
        MJD = ref_table['MJD'][idx]
        fiberID = ref_table['fiberID'][idx]

        galaxy_ID = (plate, MJD, fiberID)

        ref_dict[galaxy_ID] = idx

    return ref_dict