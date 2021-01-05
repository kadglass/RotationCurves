import numpy

from scipy.optimize import least_squares

import matplotlib.pyplot as plt


def residuals_function(a, x, y, w):
    
    return w*(a[0] + a[1]*x - y)
    


def smooth(x, y, span=0.25):
    """
    Description
    ===========
    
    Using a sliding window over an input data set (x,y), perform weighted least
    squares regression on each window to calculate a "smoothed" y value at each
    input x value.
    
    At either edge, the window size will grow up to the selected span percentage
    and shrink back down to 0 as the window intersects the edges of the data.
    Windows are centered on the current x value as this function iterates
    through the data.
    
    Parameters
    ==========
    
    x : np.ndarray of length (N,)
        the x values of the data
        
    y : np.ndarray of length (N,)
        the y values of the data
        
    span : float in [0.0, 1.0)
        percentage of the x-span to use for a window length
        
    Returns
    =======

    x : np.ndarray of length (N,)
        the sorted x values of the data
    
    out_y : np.ndarray of length (N,)
        the smoothed y values
    """
    
    ################################################################################
    # First sort the data to make iterating through and finding the window indices
    # much easier
    ################################################################################
    sort_order = x.argsort()
    
    x = x[sort_order]
    
    y = y[sort_order]
    
    out_y = numpy.empty(y.shape[0], dtype=y.dtype)
    
    ################################################################################
    # Using span percentage, calculate in x-space units the length of half the
    # window
    #
    # Initialize some start and end index values, and some space for our 
    # y = a0 + a1*x model fitting
    ################################################################################
    x_min = x.min()
    
    x_max = x.max()
    
    x_range = x_max - x_min
    
    middle_half_window_dist = span*x_range/2.0
    
    start_idx = 0
    
    end_idx = 1
    
    num_pts = x.shape[0]
    
    #print("Num pts: ", num_pts)
    
    a_vals = numpy.ones(2)
    
    ################################################################################
    # Slide the window through the available x points
    ################################################################################
    for idx in range(num_pts):
        
        if idx == 0 or idx == num_pts - 1:
            out_y[idx] = y[idx]
            continue
        
        curr_x = x[idx]
        
        curr_span = (curr_x - x_min)/x_range
        
        if curr_span < span/2.0:
            
            half_window_dist = curr_span*x_range
            
        elif curr_span > (1.0 - span/2.0):
            
            half_window_dist = (1.0 - curr_span)*x_range
            
        else:
            half_window_dist = middle_half_window_dist
        
        left_edge = x[idx] - half_window_dist
        
        right_edge = x[idx] + half_window_dist
        
        while x[start_idx] < left_edge:
            start_idx += 1
            
        while x[end_idx] < right_edge and end_idx < (num_pts-1):
            end_idx += 1
            
        window_x = x[start_idx:end_idx]
            
        window_y = y[start_idx:end_idx]
        
        window_std = numpy.std(window_y)
        
        window_mean = numpy.mean(window_y)
        
        diffs = window_y - window_mean
        
        weights = window_std/numpy.abs(diffs)
        
        weights[weights < 0.166] = 0 #clamp anything more than 6 sigma (1/6th = 0.166) to 0
        
        ################################################################################
        # weighted least squares regression for simple model
        # y = a0 + a1*x find the a0 and a1 coefficients
        # to get the 'smooth' value at the current x[idx] location
        ################################################################################
        
        a_vals.fill(1.0) #re-init the a vals

        res_lsq = least_squares(residuals_function, a_vals, args=(window_x, window_y, weights))
        
        a_solved = res_lsq.x
        
        out_y[idx] = a_solved[0] + a_solved[1]*x[idx]
        
    return x, out_y