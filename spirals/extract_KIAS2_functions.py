
import numpy as np

import astropy.units as u



###############################################################################
###############################################################################
###############################################################################



def match_by_index( master_table, ref_table, columns_to_add, column_units=None):
    '''
    Matches the ref_table to the master_table via the galaxy index number, 
    then extracts the data in the columns_to_add list from the ref_table and 
    assigns it to the corresponding column in the master_table.


    Parameters:
    ===========

    master_table : astropy QTable
        Contains all summary information for MaNGA galaxies

    ref_table : astropy table
        KIAS-VAGC data table

    columns_to_add : list
        Column names of data to pull from ref_table and add to master_table

    column_units : list
        String representation of the units to assign to each column (necessary 
        for QTable).  Empty strings represent unitless quantities.  Default 
        value is None (no units - table is only Table).


    Returns:
    ========

    master_table : astropy QTable
        Updated version of master_table containing the extracted data
    '''


    ###########################################################################
    # Initialize output columns
    #--------------------------------------------------------------------------
    unit_dict = {}
    cunit_string = ''

    for i in range(len(columns_to_add)):
        #----------------------------------------------------------------------
        # Determine column unit
        #----------------------------------------------------------------------
        if column_units is not None:
            cunit_string = column_units[i]

        if cunit_string == '':
            cunit = 1
        elif cunit_string == 'Msun/yr':
            cunit = u.dex( u.M_sun / u.yr)
        elif cunit_string == '/yr':
            cunit = u.dex( 1/u.yr)
        else:
            print('Column unit', cunit_string, 'unknown.')
            exit()
        #----------------------------------------------------------------------
        # Build unit reference dictionary
        #----------------------------------------------------------------------
        unit_dict[columns_to_add[i]] = cunit
        #----------------------------------------------------------------------
        master_table[columns_to_add[i]] = -99*np.ones(len(master_table), dtype=float) * cunit
    ###########################################################################



    ###########################################################################
    # Build dictionary of tuples for storing galaxies with KIAS-VAGC indices
    #--------------------------------------------------------------------------
    ref_dict = galaxies_dict(ref_table)
    ###########################################################################



    ###########################################################################
    # Match galaxies and extract new data
    #--------------------------------------------------------------------------
    for i in range( len( master_table)):
        
        index = master_table['index'][i]

        if index in ref_dict:
            for name in columns_to_add:
                master_table[name][i] = ref_table[name][ref_dict[index]] * unit_dict[name]
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
        Keys are KIAS-VAGC index numbers for those galaxies in the KIAS-VAGC
    '''

    # Initialize dictionary of cell IDs with at least one galaxy in them
    ref_dict = {}

    for idx in range(len(ref_table)):

        galaxy_ID = ref_table['index'][idx]

        ref_dict[galaxy_ID] = idx

    return ref_dict



