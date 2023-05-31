
import numpy as np




################################################################################
# Initialize output table
#-------------------------------------------------------------------------------
def add_columns(data_table, fit_function):
    '''
    Add additional columns to the table.


    PARAMETERS
    ==========

    data_table : astropy table
        Table of galaxies with various data already included.

    fit_function : string
        Determines which function to use for the velocity.  Options are 'BB' and 
        'tanh'.


    RETURNS
    =======

    data_table : astropy table
        Original data_table with additional columns
    '''

    N = len(data_table)

    data_table['v_sys'] = np.nan
    data_table['v_sys_err'] = np.nan

    data_table['ba'] = np.nan
    data_table['ba_err'] = np.nan

    data_table['x0'] = np.nan
    data_table['x0_err'] = np.nan

    data_table['y0'] = np.nan
    data_table['y0_err'] = np.nan

    data_table['phi'] = np.nan
    data_table['phi_err'] = np.nan

    data_table['r_turn'] = np.nan*np.ones(N, dtype=float)
    data_table['r_turn_err'] = np.nan*np.ones(N, dtype=float)

    data_table['v_max'] = np.nan*np.ones(N, dtype=float)
    data_table['v_max_err'] = np.nan*np.ones(N, dtype=float)
    
    data_table['chi2'] = np.nan

    if fit_function == 'BB':
        data_table['alpha'] = np.nan*np.ones(N, dtype=float)
        data_table['alpha_err'] = np.nan*np.ones(N, dtype=float)
    elif fit_function != 'tanh':
        print('This fit_function is not known.  Please update add_columns function.')

    data_table['nsa_elpetro_th90'] = np.nan*np.ones(N, dtype=float)

    data_table['fit_flag'] = np.nan*np.ones(N, dtype=float)


    return data_table



def add_smoothness_columns(data_table):
    N = len(data_table)

    data_table['smoothness_score'] = np.nan
    data_table['can_fit'] = np.nan

    return data_table


################################################################################
# Initialize stellar mass output table
#-------------------------------------------------------------------------------
def add_disk_columns(data_table):
    '''
    Add additional columns to the table.


    PARAMETERS
    ==========

    data_table : astropy table
        Table of galaxies with various data already included.


    RETURNS
    =======

    data_table : astropy table
        Original data_table with additional columns
    '''

    N = len(data_table)

    data_table['Sigma_disk'] = np.nan
    data_table['Sigma_disk_err'] = np.nan

    data_table['R_disk'] = np.nan
    data_table['R_disk_err'] = np.nan

    data_table['rho_bulge'] = np.nan
    data_table['rho_bulge_err'] = np.nan

    data_table['R_bulge'] = np.nan
    data_table['R_bulge_err'] = np.nan

    data_table['M90_disk'] = np.nan
    data_table['M90_disk_err'] = np.nan

    data_table['M_disk'] = np.nan
    data_table['M_disk_err'] = np.nan



    return data_table




################################################################################
################################################################################
################################################################################




def fillin_output_table(output_table, data_to_add, row_index, col_name=None):
    '''
    Add data values to table.


    PARAMETERS
    ==========

    output_table : astropy table
        Table to add data values to.  Note that the columns should already exist 
        in the table for these values!

    data_to_add : dictionary or float
        Data to add to the table.  If a dictionary, field names must match the 
        corresponding column name in output_table.

    row_index : integer
        Row in output_table in which to add the data

    col_name : string
        Column name in which to insert data_to_add, if data_to_add is only a 
        float (and not a dictionary).

        Default value is None - data_to_add is a dictionary.


    RETURNS
    =======

    output_table : astropy table
        Same as input output_table, but with values in row_index replaces with 
        provided data.
    '''


    if col_name is None:

        #print(data_to_add)
        #print(output_table.colnames)

        for field in data_to_add:

            if field not in output_table.colnames:
                output_table[field] = np.nan

            output_table[field][row_index] = data_to_add[field]

    else:

        if col_name not in output_table.colnames:
            output_table[col_name] = np.nan
            
        output_table[col_name][row_index] = data_to_add


    return output_table













