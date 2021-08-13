
from astropy.table import Table

import numpy as np


################################################################################
################################################################################

def galaxies_dict( ref_table):
    '''
    Built dictionary of plate-IFU keys that refer to the galaxy's row index in 
    the ref_table.


    PARAMETERS
    ==========

    ref_table : astropy table
        Data table with column plateifu (string of <plate>-<IFU>)


    RETURNS
    =======

    ref_dict : dictionary
        Dictionary with keys plate-IFU and values are the row index in 
        ref_table
    '''


    # Initialize dictionary to store plate-ifu and row index
    ref_dict = {}


    for i in range(len(ref_table)):

        galaxy_ID = ref_table['plateifu'][i]

        ref_dict[galaxy_ID] = i


    return ref_dict


################################################################################
################################################################################

def match_DRP( master_table, columns_to_add, column_dtypes, column_units=None):
    '''
    Locate the requested data in the DRP file for each galaxy in the 
    master_table.


    PARAMETERS
    ==========

    master_table : astropy quantity table
        Contains all objects to match to

    columns_to_add : list of strings
        List of column names from DRP file to add to the master_table

    column_dtypes : list of data types
        List of column data types (ith data type of the ith quantity in 
        columns_to_add)

    column_units : list of units
        List of column units (ith unit of ith quantity in columns_to_add)
        Default value is None - no units for the table


    RETURNS
    =======

    master_table : astropy quantity table
        Same as original table, but with additional columns specified by 
        columns_to_add from DRP file
    '''


    ############################################################################
    # Read in DRP file
    #---------------------------------------------------------------------------
    DRP_filename = '/Users/kellydouglass/Documents/Research/data/SDSS/dr16/manga/spectro/redux/v2_4_3/drpall-v2_4_3.fits'

    DRP = Table.read(DRP_filename, format='fits')

    #print(DRP.colnames)
    ############################################################################


    ############################################################################
    # Build galaxy reference dictionary
    #---------------------------------------------------------------------------
    DRP_table_dict = galaxies_dict(DRP)
    ############################################################################


    ############################################################################
    # Add empty columns to master_table
    #---------------------------------------------------------------------------
    for i in range(len(columns_to_add)):

        if column_units is not None:

            master_table[columns_to_add[i]] = np.empty(len(master_table), 
                                                       dtype=column_dtypes[i])*column_units[i]

        else:

            master_table[columns_to_add[i]] = np.empty(len(master_table), 
                                                       dtype=column_dtypes[i])
    ############################################################################


    ############################################################################
    # Insert DRP data into master_table
    #---------------------------------------------------------------------------
    for i in range(len(master_table)):

        # Build galaxy ID
        gal_ID = str(master_table['MaNGA_plate'][i]) + '-' + str(master_table['MaNGA_IFU'][i])

        if gal_ID in DRP_table_dict:
            # Retrieve galaxy's row index number for DRP table
            DRP_idx = DRP_table_dict[gal_ID]

            # Insert data
            for j in range(len(columns_to_add)):

                if column_units is not None:

                    master_table[columns_to_add[j]][i] = DRP[columns_to_add[j]][DRP_idx]*column_units[j]

                else:
                    master_table[columns_to_add[j]][i] = DRP[columns_to_add[j]][DRP_idx]
    ############################################################################


    return master_table




