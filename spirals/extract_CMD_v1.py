
import numpy as np



###############################################################################
###############################################################################
###############################################################################



def match_CMD( master_table, CMD_ref_table):
    '''
    Matches the CMD_ref_table to the master_table via the galaxy index number, 
    then extracts the color-magnitude classification from the CMD_ref_table and 
    assigns it to the corresponding column in the master_table.


    Parameters:
    ===========

    master_table : astropy QTable
        Contains all summary information for MaNGA galaxies

    CMD_ref_table : astropy table
        Contains color-magnitude classifications


    Returns:
    ========

    master_table : astropy QTable
        Updated version of master_table containing the extracted 
        color-magnitude classifications 
    '''

    ###########################################################################
    # Initialize output column
    #--------------------------------------------------------------------------
    master_table['CMD_class'] = -1*np.ones(len(master_table), dtype=int)
    #--------------------------------------------------------------------------


    ###########################################################################
    # Build dictionary of tuples for storing galaxies with KIAS-VAGC indices
    #--------------------------------------------------------------------------
    ref_dict = galaxies_dict(CMD_ref_table)
    #--------------------------------------------------------------------------


    ###########################################################################
    # Match galaxies with CMD classifications
    #--------------------------------------------------------------------------
    for i in range( len( master_table)):
        
        index = master_table['index'][i]

        galaxy_ID = (index)

        if galaxy_ID in ref_dict.keys():
            master_table['CMD_class'][i] = CMD_ref_table['CMD_class'][ref_dict[galaxy_ID]]
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
        Tuples of (index) for those galaxies with CMD classifications
    '''

    # Initialize dictionary of cell IDs with at least one galaxy in them
    ref_dict = {}

    for idx in range(len(ref_table)):

        galaxy_ID = (ref_table['index'][idx])

        ref_dict[galaxy_ID] = idx

    return ref_dict



