import numpy as np


def calculate_shift(void, wall, field, err_field):
    '''
    Calculate the average and median shifts between the void and wall 
    populations.
    
    
    Parameters:
    ===========
    
    void : astropy table of length n_void
        Table containing void galaxy parameters
        
    wall : astropy table of length n_wall
        Table containing wall galaxy parameters
        
    field : string
        Name of the field in the void and wall tables that refers to the 
        characteristic currently being analyzed.

    err_field : string
        Name of the field containing the error of the characteristic 
        currently being analyzed.

    '''


    #######################################################################
    # Calculate averages, shift between voids and walls
    #----------------------------------------------------------------------
    v_mean = np.mean(void[field])
    w_mean = np.mean(wall[field])

    v_median = np.median(void[field])
    w_median = np.median(wall[field])

    mean_diff = v_mean - w_mean
    median_diff = v_median - w_median
    #######################################################################


    #######################################################################
    # Calculate uncertainties in the averages and shifts
    #----------------------------------------------------------------------
    # Preserve only finite elements for error calculation
    v_finite = void[err_field][np.isfinite(void[err_field])]
    w_finite = wall[err_field][np.isfinite(wall[err_field])]

    # Uncertainties in the mean
    v_mean_err = np.sqrt(np.sum(v_finite**2))/len(v_finite)
    w_mean_err = np.sqrt(np.sum(w_finite**2))/len(w_finite)

    mean_diff_err = np.sqrt(v_mean_err**2 + w_mean_err**2)

    # Uncertainties in the median
    v_median_err = v_mean_err*np.sqrt(np.pi*len(v_finite)/(4*0.5*(len(v_finite) - 1)))
    w_median_err = w_mean_err*np.sqrt(np.pi*len(w_finite)/(4*0.5*(len(w_finite) - 1)))

    median_diff_err = np.sqrt(v_median_err**2 + w_median_err**2)
    #######################################################################


    print('There are', len(void), 'void galaxies and', len(wall), 'wall galaxies in this sample.')
    print('The average ratio for voids is', v_mean, 'pm', v_mean_err, 'and for walls is', w_mean, 'pm', w_mean_err)
    print('The average difference between the two populations is', mean_diff, 'pm', mean_diff_err)
    print('The median ratio for voids is', v_median, 'pm', v_median_err, 'and for walls is', w_median, 'pm', w_median_err)
    print('The median difference between the two populations is', median_diff, 'pm', median_diff_err)


