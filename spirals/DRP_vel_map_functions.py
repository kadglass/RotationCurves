
import numpy as np
import numpy.ma as ma

import numdifftools as ndt

from skimage.filters import gaussian

from scipy.optimize import minimize

from dark_matter_mass_v1 import rot_fit_BB, rot_fit_tanh

import matplotlib.pyplot as plt





################################################################################
# Constants
#-------------------------------------------------------------------------------
H_0 = 100      # Hubble's Constant in units of h km/s/Mpc
c = 299792.458 # Speed of light in units of km/s
G = 4.30091E-3 # Gravitation constant in units of (km/s)^2 pc/Msun

q0 = 0.2

MANGA_FIBER_DIAMETER = 2*(1/60)*(1/60)*(np.pi/180) # angular fiber diameter (2") in radians
MANGA_SPAXEL_SIZE = 0.5*(1/60)*(1/60)*(np.pi/180)  # spaxel size (0.5") in radians
################################################################################



################################################################################
# Fitting function options
#-------------------------------------------------------------------------------
fit_options = {'BB': rot_fit_BB,
               'tanh': rot_fit_tanh
              }
################################################################################





################################################################################
################################################################################
################################################################################


def build_map_mask(gal_ID, 
                   fit_flag, 
                   mHa_vel, 
                   mHa_flux, 
                   mHa_flux_ivar, 
                   mHa_sigma):
    '''
    Build mask for galaxy based on the fit method used.


    Parameters
    ==========

    gal_ID : string
        <MaNGA plate> - <MaNGA IFU>

    fit_flag : float
        Identifies the fit method used to model the velocity map.
        - 1 : original (only Ha_vel_mask)
        - 2 : residual (only Ha_vel_mask)
        - 3 : continuous (Ha_vel_mask + spaxels which are separated from others)
        - 4 : non-AGN (Ha_vel_mask + spaxels with high Ha velocity dispersion)
        - 5 : S/N > 5 (Ha_vel_mask + spaxels with S/N < 5)

    mHa_vel : numpy ndarray of shape (n,n)
        masked H-alpha velocity map

    mHa_flux : numpy ndarray of shape (n,n)
        H-alpha flux map

    mHa_flux_ivar : numpy ndarray of shape (n,n)
        masked inverse variance map of the H-alpha flux

    mHa_sigma : numpy ndarray of shape (n,n)
        masked H-alpha velocity width map


    Returns:
    ========

    mask : numpy boolean ndarray of shape (n,n)
        Mask map where true values are to be masked.
    '''


    ############################################################################
    # By default, the mask will just be the Ha_vel_mask
    #---------------------------------------------------------------------------
    bitmask = mHa_vel.mask
    ############################################################################


    if fit_flag == -3:
        ########################################################################
        # Mask spaxels that have velocities which are not continuous
        #-----------------------------------------------------------------------
        _,_, bitmask = find_vel_bounds(mHa_vel, gal_ID)
        ########################################################################

    elif fit_flag == -4:
        ########################################################################
        # Mask spaxels that have large velocity widths
        #-----------------------------------------------------------------------
        _, bitmask = find_sigma_bounds(mHa_sigma, gal_ID)
        ########################################################################

    elif fit_flag > 0:
        ########################################################################
        # Mask spaxels that have S/N smaller than fit_flag
        #-----------------------------------------------------------------------
        SN = ma.abs(mHa_flux*ma.sqrt(mHa_flux_ivar))

        bitmask = np.logical_or(mHa_vel.mask + mHa_flux.mask, SN < fit_flag)
        ########################################################################


    ############################################################################
    # Convert mask into boolean array
    #---------------------------------------------------------------------------
    mask = bitmask > 0
    ############################################################################

    return mask



################################################################################
################################################################################
################################################################################
def find_sigma_bounds(mHa_sigma, gal_ID):
    '''
    Locate the maximum line width of Ha that is continuously linked to the 
    smallest line width (normally around 0) found in the line width map.


    PARAMETERS
    ==========

    mHa_sigma : masked ndarray of shape (n,n)
        Masked H-alpha line width map, in units of km/s

    gal_ID : string
        <MaNGA plate>-<MaNGA IFU>


    RETURNS
    =======

    max_sigma : float
        The maximum line width of this continous distribution (in units of 
        km/s).

    modified_mask : boolean ndarray of shape (n,n)
        Values of true correspond to masked locations (either already masked, or 
        with line widths larger than the determined line width maximum).
    '''

    bin_width = 10 # Bin the line widths in bins of 10 km/s

    if gal_ID in ['8601-1902', '9037-12703', '8724-12701', '9088-12702', 
                  '8551-12703', '8452-12703', '7960-12704', '8481-12705', 
                  '10001-12705', '8319-12705', '9869-12705', '9045-1901', 
                  '8446-1901', '8555-1901', '8551-1901', '8158-1901', 
                  '8453-1901', '8244-1902', '9888-3701', '8335-3701', 
                  '8261-3701', '8552-3702', '8725-3703', '8999-3704', 
                  '9501-3704', '8257-6101', '8483-6101', '8244-6101', 
                  '8131-6101', '8338-6103', '8985-6104', '8935-6104', 
                  '8591-6104', '8944-6104', '8239-9101', '8154-9101', 
                  '8320-9101', '8086-9102', '8081-9102', '9184-9102']:
        bin_width = 20
    elif gal_ID in ['8341-12704']:
        bin_width = 5

    sigma_extreme = ma.max(mHa_sigma) # Maximum line width in the map

    sigma_bin_counts, sigma_bin_edges = np.histogram(mHa_sigma.compressed(), 
                                                     bins=np.arange(0, 
                                                                    sigma_extreme + bin_width, 
                                                                    bin_width))
    ############################################################################
    # Find the smallest-width bin with counts
    #---------------------------------------------------------------------------
    low_bin = 0

    while low_bin < len(sigma_bin_counts) and sigma_bin_counts[low_bin] == 0:
        low_bin += 1
    ############################################################################


    ############################################################################
    # Find the highest bin in which data is connected back to this lowest bin
    #---------------------------------------------------------------------------
    max_bin = low_bin + 1

    while max_bin < len(sigma_bin_counts) and sigma_bin_counts[max_bin] > 0:
        max_bin += 1

    max_sigma = sigma_bin_edges[max_bin]
    ############################################################################


    ############################################################################
    # Build the new mask, masking out all spaxels with line widths larger than 
    # max_sigma.
    #---------------------------------------------------------------------------
    bad_sigma = mHa_sigma > max_sigma

    new_mask = np.logical_or(mHa_sigma.mask > 0, bad_sigma)
    ############################################################################

    return max_sigma, new_mask




################################################################################
################################################################################
################################################################################
def find_vel_bounds(mHa_vel, gal_ID):
    '''
    Locate the minimum and maximum velocities of the velocity distribution that 
    is continuously linked to the most common velocity (normally around 0) 
    found in the velocity map.


    PARAMETERS
    ==========

    mHa_vel : masked ndarray of shape (n,n)
        Masked H-alpha velocity map, in units of km/s

    gal_ID : string
        <MaNGA plate>-<MaNGA IFU>


    RETURNS
    =======

    min_vel, max_vel : float
        The minimum and maximum velocities of this continous distribution (in 
        units of km/s).

    modified_mask : boolean ndarray of shape (n,n)
        Values of true correspond to masked locations (either already masked, or 
        with velocities outside of the determined velocity range).
    '''

    bin_width = 10 # Bin the velocity in bins of 10 km/s

    if gal_ID in ['8150-12703', '8980-1902', '8261-6104']:
        bin_width = 5

    vel_extreme = ma.max(ma.abs(mHa_vel)) # Maximum velocity in the map

    vel_bin_counts, vel_bin_edges = np.histogram(mHa_vel.compressed(), 
                                                 bins=np.arange(-vel_extreme, 
                                                                vel_extreme + bin_width, 
                                                                bin_width))
    ############################################################################
    # Find the bin with the most counts
    #---------------------------------------------------------------------------
    center_bin = np.argmax(vel_bin_counts)
    ############################################################################


    ############################################################################
    # Find the highest bin in which data is connected back to this central bin
    #---------------------------------------------------------------------------
    max_bin = center_bin + 1

    while max_bin < len(vel_bin_counts) and vel_bin_counts[max_bin] > 0:
        max_bin += 1

    max_vel = vel_bin_edges[max_bin]
    ############################################################################


    ############################################################################
    # Find the lowest bin in which data is connected back to the central bin
    #---------------------------------------------------------------------------
    if center_bin == 0:
        min_bin = 0
    else:
        min_bin = center_bin - 1

    while vel_bin_counts[min_bin] > 0 and min_bin > 0:
        min_bin -= 1

    min_vel = vel_bin_edges[min_bin]
    ############################################################################


    ############################################################################
    # Build the new mask, masking out all spaxels with velocities outside the 
    # range (min_vel, max_vel).
    #---------------------------------------------------------------------------
    bad_vel = np.logical_or(mHa_vel < min_vel, mHa_vel > max_vel)

    new_mask = np.logical_or(mHa_vel.mask > 0, bad_vel)
    ############################################################################

    return min_vel, max_vel, new_mask




################################################################################
################################################################################
################################################################################
def find_phi(center_coords, phi_angle, vel_map):
    '''
    Find a point along the semi-major axis that has data to determine if phi 
    needs to be adjusted.  (This is necessary because the positive y-axis is 
    defined as being along the semi-major axis of the positive velocity side of 
    the velocity map.)
    
    
    PARAMETERS
    ==========
    
    center_coords : tuple
        Coordinates of the center of the galaxy
        
    phi_angle : float
        Initial rotation angle of the galaxy, E of N.  Units are degrees.
        
    vel_map : masked ndarray of shape (n,n)
        Masked H-alpha velocity map
        
        
    RETURNS
    =======
    
    phi_adjusted : float
        Rotation angle of the galaxy, E of N, that points along the positive 
        velocity sector.  Units are radians.
    '''
    
    # Convert phi_angle to radians
    phi = phi_angle*np.pi/180.

    # Extract "systemic" velocity (velocity at center spaxel)
    v_sys = vel_map[center_coords]
    
    f = 0.4
    
    checkpoint_masked = True
    
    while checkpoint_masked:
        delta_x = int(center_coords[1]*f)
        delta_y = int(delta_x/np.tan(phi))
        semi_major_axis_spaxel = np.subtract(center_coords, (-delta_y, delta_x))
        
        for i in range(len(semi_major_axis_spaxel)):
            if semi_major_axis_spaxel[i] < 0:
                semi_major_axis_spaxel[i] = 0
            elif semi_major_axis_spaxel[i] >= vel_map.shape[i]:
                semi_major_axis_spaxel[i] = vel_map.shape[i] - 1
                
        # Check value along semi-major axis
        if vel_map.mask[tuple(semi_major_axis_spaxel)] == 0:
            checkpoint_masked = False
        else:
            f *= 0.9
            
    if vel_map[tuple(semi_major_axis_spaxel)] - v_sys < 0:
        phi_adjusted = phi + np.pi
    else:
        phi_adjusted = phi
            
    return phi_adjusted



################################################################################
################################################################################
################################################################################

def find_center(vel_map):
    '''
    Locate the center of the galaxy velocity map, defined as the position with 
    the velocity value closest to 0.


    PARAMETERS
    ==========

    vel_map : numpy ndarray of shape (n,n)
        Masked H-alpha velocity map


    RETURNS
    =======

    center : numpy ndarray of shape (,2)
        Center coordinates of velocity map, [y,x] or [row, column]
    '''

    ############################################################################
    # Transform velocity map so that the 0th value is the maximum in the array 
    # (including the masked points)
    #---------------------------------------------------------------------------
    neg_map = -np.abs(vel_map)
    shifted_neg_map = neg_map + np.abs(ma.min(vel_map))

    # Set all masked values to have a value of -1
    shifted_neg_map[vel_map.mask] = -1
    ############################################################################


    ############################################################################
    # Find the location of the maximum in the shifted negative map
    #---------------------------------------------------------------------------
    smoothed_map = gaussian(shifted_neg_map, sigma=2)

    center = np.unravel_index(np.argmax(smoothed_map), smoothed_map.shape)
    ############################################################################

    return center






################################################################################
################################################################################
################################################################################



def deproject_spaxel(coords, center, phi, i_angle):
    '''
    Calculate the deprojected radius for the given coordinates in the map.


    PARAMETERS
    ==========

    coords : length-2 tuple
        (i,j) coordinates of the current spaxel

    center : length-2 tuple
        (i,j) coordinates of the galaxy's center

    phi : float
        Rotation angle (in radians) east of north of the semi-major axis.

    i_angle : float
        Inclination angle (in radians) of the galaxy.


    RETURNS
    =======

    r : float
        De-projected radius from the center of the galaxy for the given spaxel 
        coordinates.
    '''


    # Distance components between center and current location
    delta = np.subtract(coords, center)

    # x-direction distance relative to the semi-major axis
    if i_angle < 0.5*np.pi:
        dx_prime = (delta[1]*np.cos(phi) + delta[0]*np.sin(phi))/np.cos(i_angle)
    else:
        dx_prime = 0.

    # y-direction distance relative to the semi-major axis
    dy_prime = (-delta[1]*np.sin(phi) + delta[0]*np.cos(phi))

    # De-projected radius for the current point
    r = np.sqrt(dx_prime**2 + dy_prime**2)

    # Angle (counterclockwise) between North and current position
    theta = np.arctan2(-dx_prime, dy_prime)

    return r, theta



################################################################################
# Model velocity map
#-------------------------------------------------------------------------------
def model_vel_map(params, map_shape, scale, fit_function):
    '''
    Create a model velocity map of shape map_shape based on the values in params


    PARAMETERS
    ==========

    params : list
        List of fit parameters

    map_shape : tuple
        Shape of velocity map array

    scale : float
        Pixel scale (to convert from pixels to kpc)

    fit_function : string
        Determines which function to use for the velocity.  Options are 'BB' and 
        'tanh'.


    RETURNS
    =======

    vel_map : numpy array of shape map_shape
        Model velocity map array
    '''


    ############################################################################
    # Unpack fit parameters
    #---------------------------------------------------------------------------
    if fit_function == 'BB':
        v_sys, i_angle, i_center, j_center, phi, v_max, r_turn, alpha = params
    elif fit_function == 'tanh':
        v_sys, i_angle, i_center, j_center, phi, v_max, r_turn = params
    else:
        print('Unknown fit function.  Please update model_vel_map function.', 
              flush=True)
    ############################################################################


    ############################################################################
    # Initialize velocity map
    #---------------------------------------------------------------------------
    vel_map = np.zeros(map_shape)

    v = np.zeros(map_shape)
    theta = np.zeros(map_shape)
    ############################################################################


    ############################################################################
    # Calculate velocity at each point in the velocity map
    #---------------------------------------------------------------------------
    center = (i_center, j_center)
    
    for i in range(map_shape[0]):
        for j in range(map_shape[1]):

            # De-projected radius for the current point
            r, theta[i,j] = deproject_spaxel((i,j), center, phi, i_angle)

            # Rotational velocity at current point
            if fit_function == 'BB':
                v[i,j] = rot_fit_BB(r*scale, [v_max, r_turn, alpha])
            elif fit_function == 'tanh':
                v[i,j] = rot_fit_tanh(r*scale, [v_max, r_turn])
            else:
                print('Fit function not known.  Please update model_vel_map function.', 
                      flush=True)

    #print([v_max, r_turn, alpha])
    #print(params)

    # Observed velocity at current point
    vel_map = v*np.sin(i_angle)*np.cos(theta) + v_sys
    ############################################################################

    return vel_map




################################################################################
# Function to minimize
#-------------------------------------------------------------------------------
def calculate_chi2(params, vel_map, vel_map_ivar, pix_scale, fit_function):
    '''
    chi2 of the velocity map


    PARAMETERS
    ==========

    params : list
        List of fit parameters

    vel_map : numpy array of shape (n,n)
        Masked array of measured velocities

    vel_map_ivar : numpy array of shape (n,n)
        Inverse variance of the measured velocities

    pix_scale : float
        Scale of each pixel (to convert from pixel units to kpc)

    fit_function : string
        Determines which function to use for the velocity.  Options are 'BB' and 
        'tanh'.


    RETURNS
    =======

    chi2_norm : float
        Chi2 value of the current value of the params normalized by the number 
        of data points (minus the number of free parameters)
    '''

    ############################################################################
    # Create fitted velocity map based on the values in params
    #---------------------------------------------------------------------------
    vel_map_model = model_vel_map(params, vel_map.shape, pix_scale, fit_function)
    ############################################################################


    ############################################################################
    # Calculate chi2 of current fit
    #---------------------------------------------------------------------------
    chi2 = ma.sum(vel_map_ivar*(vel_map_model - vel_map)**2)

    chi2_norm = chi2/(np.sum(~vel_map.mask) - len(params))
    ############################################################################


    return chi2_norm
    
    
    
    
def calculate_chi2_flat(params, 
                        flat_vel_map, 
                        flat_vel_map_ivar, 
                        map_mask,
                        pix_scale, 
                        fit_function):
    '''
    chi2 of the velocity map


    PARAMETERS
    ==========

    params : list
        List of fit parameters

    flat_vel_map : numpy array of shape (n,)
        Flattened array of unmasked measured velocities

    flat_vel_map_ivar : numpy array of shape (n,)
        Flattened inverse variance of the unmasked measured velocities
        
    map_mask : numpy array of shape (m,m)
        Bit mask for map.  Values of 0 are valid, all others are bad.

    pix_scale : float
        Scale of each pixel (to convert from pixel units to kpc)

    fit_function : string
        Determines which function to use for the velocity.  Options are 'BB' and 
        'tanh'.


    RETURNS
    =======

    chi2_norm : float
        Chi2 value of the current value of the params normalized by the number 
        of data points (minus the number of free parameters)
    '''

    ############################################################################
    # Create fitted velocity map based on the values in params
    #---------------------------------------------------------------------------
    vel_map_model = model_vel_map(params, map_mask.shape, pix_scale, fit_function)
    ############################################################################
    
    
    ############################################################################
    # Flatten both the mask and model, and remove all elements from the model 
    # which are masked.
    #---------------------------------------------------------------------------
    mvel_map_model = ma.array(vel_map_model, mask=map_mask)
    
    flat_vel_map_model = mvel_map_model.compressed()
    ############################################################################


    ############################################################################
    # Calculate chi2 of current fit
    #---------------------------------------------------------------------------
    chi2 = np.sum(flat_vel_map_ivar*(flat_vel_map_model - flat_vel_map)**2)
    #chi2 = np.sum((flat_vel_map_model - flat_vel_map)**2)

    #chi2_norm = chi2/(len(flat_vel_map) - len(params))
    ############################################################################


    #return chi2_norm
    return chi2
################################################################################




def calculate_residual_flat(params, 
                            flat_vel_map, 
                            map_mask,
                            pix_scale, 
                            fit_function):
    '''
    residual of the velocity map


    PARAMETERS
    ==========

    params : list
        List of fit parameters

    flat_vel_map : numpy array of shape (n,)
        Flattened array of unmasked measured velocities
        
    map_mask : numpy array of shape (m,m)
        Bit mask for map.  Values of 0 are valid, all others are bad.

    pix_scale : float
        Scale of each pixel (to convert from pixel units to kpc)

    fit_function : string
        Determines which function to use for the velocity.  Options are 'BB' and 
        'tanh'.


    RETURNS
    =======

    residual_norm : float
        Residual value of the current value of the params normalized by the 
        number of data points (minus the number of free parameters)
    '''

    ############################################################################
    # Create fitted velocity map based on the values in params
    #---------------------------------------------------------------------------
    vel_map_model = model_vel_map(params, map_mask.shape, pix_scale, fit_function)
    ############################################################################
    
    
    ############################################################################
    # Flatten both the mask and model, and remove all elements from the model 
    # which are masked.
    #---------------------------------------------------------------------------
    mvel_map_model = ma.array(vel_map_model, mask=map_mask)
    
    flat_vel_map_model = mvel_map_model.compressed()
    ############################################################################


    ############################################################################
    # Calculate residual of current fit
    #---------------------------------------------------------------------------
    residual = np.sum((flat_vel_map_model - flat_vel_map)**2)

    residual_norm = residual/(len(flat_vel_map) - len(params))
    ############################################################################


    return residual_norm
################################################################################




################################################################################
# Function to minimize
#-------------------------------------------------------------------------------
def chi2_velocity(params, 
                  pos_params,
                  vel_map, 
                  vel_map_ivar, 
                  pix_scale, 
                  fit_function):
    '''
    Calculate the chi2 of the velocity map.  The free parameters are the 
    parameters of the velocity funciton.


    PARAMETERS
    ==========

    params : list
        List of fit parameters

    pos_params : list
        List of the fit parameters for the velocity map's position

    vel_map : numpy array of shape (n,n)
        Masked array of measured velocities

    vel_map_ivar : numpy array of shape (n,n)
        Inverse variance of the measured velocities

    pix_scale : float
        Scale of each pixel (to convert from pixel units to kpc)

    fit_function : string
        Determines which function to use for the velocity.  Options are 'BB' and 
        'tanh'.


    RETURNS
    =======

    chi2_norm : float
        Chi2 value of the current value of the params normalized by the number 
        of data points (minus the number of free parameters)
    '''

    ############################################################################
    # Create fitted velocity map based on the values in params
    #---------------------------------------------------------------------------
    map_params = np.concatenate([pos_params, params])
    vel_map_model = model_vel_map(map_params, 
                                  vel_map.shape, 
                                  pix_scale, 
                                  fit_function)
    ############################################################################


    ############################################################################
    # Calculate chi2 of current fit
    #---------------------------------------------------------------------------
    chi2 = ma.sum(vel_map_ivar*(vel_map_model - vel_map)**2)

    chi2_norm = chi2/(np.sum(~vel_map.mask) - len(params))
    ############################################################################


    return chi2_norm
################################################################################




################################################################################
# Function to minimize
#-------------------------------------------------------------------------------
def chi2_position(params, 
                  vel_params,
                  vel_map, 
                  vel_map_ivar, 
                  pix_scale, 
                  fit_function):
    '''
    Calculate the chi2 of the velocity map.  The free parameters are the 
    position parameters of the velocity field.


    PARAMETERS
    ==========

    params : list
        List of fit parameters

    vel_params : list
        List of the best-fit parameters for the velocity function

    vel_map : numpy array of shape (n,n)
        Masked array of measured velocities

    vel_map_ivar : numpy array of shape (n,n)
        Inverse variance of the measured velocities

    pix_scale : float
        Scale of each pixel (to convert from pixel units to kpc)

    fit_function : string
        Determines which function to use for the velocity.  Options are 'BB' and 
        'tanh'.


    RETURNS
    =======

    chi2_norm : float
        Chi2 value of the current value of the params normalized by the number 
        of data points (minus the number of free parameters)
    '''

    ############################################################################
    # Create fitted velocity map based on the values in params
    #---------------------------------------------------------------------------
    map_params = np.concatenate([params, vel_params])
    vel_map_model = model_vel_map(map_params, vel_map.shape, pix_scale, fit_function)
    ############################################################################


    ############################################################################
    # Calculate chi2 of current fit
    #---------------------------------------------------------------------------
    chi2 = ma.sum(vel_map_ivar*(vel_map_model - vel_map)**2)

    chi2_norm = chi2/(np.sum(~vel_map.mask) - len(params))
    ############################################################################


    return chi2_norm
################################################################################




################################################################################
################################################################################
################################################################################

def logL_BB(params, pix_scale, vel_map, vel_map_ivar):
    '''
    Log likelihood of the data and the fit values for the BB fit function.


    PARAMETERS
    ==========

    params : list or ndarray
        List of fit parameters

    pix_scale : float
        Conversion from spaxels to kpc

    vel_map : ndarray of shape (n,n)
        Data velocity map

    vel_map_ivar : ndarray of shape (n,n)
        Velocity error values for the data points


    RETURNS
    =======

    logL : float
        Log likelihood of set velocity given model parameters
    '''

    lambda1 = model_vel_map(params, vel_map.shape, pix_scale, 'BB')
    lambda1[lambda1 <= 0] = np.finfo( dtype=np.float64).tiny

    return -0.5*ma.sum( vel_map_ivar*(vel_map - lambda1)**2 - ma.log(vel_map_ivar))




################################################################################
################################################################################
################################################################################

def nlogL_BB(params, pix_scale, vel_map, vel_map_ivar):
    '''
    Returns the negative log likelihood of the data and the fit values for the 
    BB fit function.
    '''

    return -logL_BB(params, pix_scale, vel_map, vel_map_ivar)




################################################################################
################################################################################
################################################################################

def vel_logL_BB(vel_params, pos_params, pix_scale, vel_map, vel_map_ivar):
    '''
    Log likelihood of the data and the fit values for the BB fit function.


    PARAMETERS
    ==========

    vel_params : list or ndarray
        Fit parameters for the velocity function

    pos_params : list or ndarray
        Position parameters (assumed fixed in this fit)

    pix_scale : float
        Conversion from spaxels to kpc

    vel_map : ndarray of shape (n,n)
        Data velocity map

    vel_map_ivar : ndarray of shape (n,n)
        Velocity error values for the data points


    RETURNS
    =======

    logL : float
        Log likelihood of set velocity given model parameters
    '''

    params = np.concatenate([pos_params, vel_params])

    lambda1 = model_vel_map(params, vel_map.shape, pix_scale, 'BB')
    lambda1[lambda1 <= 0] = np.finfo( dtype=np.float64).tiny

    return -0.5*ma.sum( vel_map_ivar*(vel_map - lambda1)**2 - ma.log(vel_map_ivar))



################################################################################
################################################################################
################################################################################

def vel_nlogL_BB(vel_params, pos_params, pix_scale, vel_map, vel_map_ivar):
    '''
    Returns the negative log likelihood of the data and the best fit values for 
    the BB fit funciton.
    '''

    params = np.concatenate([pos_params, vel_params])

    return -logL_BB(params, pix_scale, vel_map, vel_map_ivar)




################################################################################
# Fit the velocity map
#-------------------------------------------------------------------------------
def find_vel_map(gal_ID, 
                 mHa_vel, 
                 mHa_vel_ivar, 
                 mHa_sigma, 
                 mHa_flux, 
                 mHa_flux_ivar,
                 z, 
                 i_center_guess, 
                 j_center_guess,
                 sys_vel_guess, 
                 inclination_angle_guess, 
                 phi_guess, 
                 fit_function):
    '''
    Fit the H-alpha velocity map to find the best-fit values for the kinematics.


    PARAMETERS
    ==========

    gal_ID : string
        <plate>-<IFU>

    mHa_vel : numpy array of shape (n,n)
        Masked H-alpha velocity field data

    Ha_vel_ivar : numpy array of shape (n,n)
        Masked inverse variance in the H-alpha velocity field data

    mHa_sigma : numpy array of shape (n,n)
        Masked H-alpha line width data

    mHa_flux : numpy array of shape (n,n)
        Masked H-alpha flux field data

    mHa_flux_ivar : numpy array of shape (n,n)
        Masked inverse variance of the H-alpha flux field data

    z : float
        Redshift of galaxy

    i_, j_center_guess : float
        Initial guess for the kinematic center of the galaxy

    sys_vel_guess : float
        Initial guess for the galaxy's systemic velocity

    inclination_angle_guess : float
        Initial guess for the inclination angle of the galaxy

    phi_guess : float
        Initial guess for the orientation angle of the galaxy's major axis, 
        defined as east of north.

    fit_function : string
        Determines which function to use for the velocity.  Options are 'BB' and 
        'tanh'.


    RETURNS
    =======

    best_fit_values : dictionary
        Values and errors of the fit parameters

    best_fit_map : numpy array of shape (n,n)
        Best-fit velocity map
    '''


    ############################################################################
    # Convert pixel distance to physical distances.
    #---------------------------------------------------------------------------
    dist_to_galaxy_Mpc = c*z/H_0
    dist_to_galaxy_kpc = dist_to_galaxy_Mpc*1000

    pix_scale_factor = dist_to_galaxy_kpc*np.tan(MANGA_SPAXEL_SIZE)

    '''
    print("z:", z)
    print("dist_to_galaxy_Mpc:", dist_to_galaxy_Mpc)
    print("dist_to_galaxy_kpc:", dist_to_galaxy_kpc)
    print("pix_scale_factor:", pix_scale_factor)
    '''
    ############################################################################


    ############################################################################
    # Use the maximum velocity in the data as the initial guess for the maximum 
    # velocity of the rotation curve.
    #---------------------------------------------------------------------------
    v_max_index = np.unravel_index(ma.argmax(ma.abs(mHa_vel)), mHa_vel.shape)
    v_max_guess = ma.abs(mHa_vel[v_max_index]/np.sin(inclination_angle_guess))

    #print('v_max index:', v_max_index)
    #print("v_max_guess:", v_max_guess)
    #print('i_angle guess:', inclination_angle_guess)
    ############################################################################


    ############################################################################
    # Set the extremes for each of the fit parameters
    #---------------------------------------------------------------------------
    # Systemic velocity
    sys_vel_low = -1100
    sys_vel_high = 1100
    sys_vel_bounds = (sys_vel_low, sys_vel_high)

    # Inclination angle
    #inclination_angle_low = 0
    #inclination_angle_high = 0.5*np.pi
    inclination_angle_low = np.max([0, inclination_angle_guess - 0.1])
    inclination_angle_high = np.min([0.5*np.pi, inclination_angle_guess + 0.1])
    inclination_angle_bounds = (inclination_angle_low, inclination_angle_high)

    # Center coordinates
    i_center_low = i_center_guess - 10
    i_center_high = i_center_guess + 10
    #i_center_low = i_center_guess - 5
    #i_center_high = i_center_guess + 5
    i_center_bounds = (i_center_low, i_center_high)

    j_center_low = j_center_guess - 10
    j_center_high = j_center_guess + 10
    #j_center_low = j_center_guess - 5
    #j_center_high = j_center_guess + 5
    j_center_bounds = (j_center_low, j_center_high)

    # Orientation angle
    phi_low = 0
    phi_high = 2*np.pi + 0.1
    phi_bounds = (phi_low, phi_high)

    # Maximum velocity [km/s]
    v_max_low = 10
    v_max_high = 5100
    v_max_bounds = (v_max_low, v_max_high)

    # Turn radius [kpc]
    r_turn_low = 0.01
    r_turn_high = 100
    r_turn_bounds = (r_turn_low, r_turn_high)
    ############################################################################


    ############################################################################
    # Set the initial guess for r_turn to be equal to half of the radius where 
    # the maximum velocity occurs
    #---------------------------------------------------------------------------
    center_guess = (i_center_guess, j_center_guess)

    r_turn_guess_spaxels,_ = deproject_spaxel(v_max_index, 
                                              center_guess, 
                                              phi_guess, 
                                              inclination_angle_guess)

    r_turn_guess_kpc = 0.5*r_turn_guess_spaxels*pix_scale_factor

    if r_turn_guess_kpc < r_turn_low:
        r_turn_guess_kpc = 1.1*r_turn_low
    ############################################################################


    ############################################################################
    # Set initial guesses, bounds for parameters specific to the fit function 
    # selected.
    #---------------------------------------------------------------------------
    if fit_function == 'BB':

        # Alpha
        alpha_guess = 2
        alpha_low = 0.001
        alpha_high = 100
        alpha_bounds = (alpha_low, alpha_high)

        # Parameter guesses
        vel_guesses = [v_max_guess, \
                       r_turn_guess_kpc, \
                       alpha_guess]
        pos_guesses = [sys_vel_guess, \
                       inclination_angle_guess, \
                       i_center_guess, \
                       j_center_guess, \
                       phi_guess]
        # Parameter bounds
        vel_bounds = [v_max_bounds, \
                      r_turn_bounds, \
                      alpha_bounds]
        pos_bounds = [sys_vel_bounds, \
                      inclination_angle_bounds, \
                      i_center_bounds, \
                      j_center_bounds, \
                      phi_bounds]

    elif fit_function == 'tanh':

        # Parameter guesses
        vel_guesses = [v_max_guess, r_turn_guess_kpc]
        pos_guesses = [sys_vel_guess, \
                       inclination_angle_guess, \
                       i_center_guess, \
                       j_center_guess, \
                       phi_guess]
        # Parameter bounds
        vel_bounds = [v_max_bounds, r_turn_bounds]
        pos_bounds = [sys_vel_bounds, \
                      inclination_angle_bounds, \
                      i_center_bounds, \
                      j_center_bounds, \
                      phi_bounds]

    else:
        print('Selected fit function is not known!  Please edit find_vel_map function in DRP_vel_map_functions.py.', 
              flush=True)
        
    #print('Position guesses:', pos_guesses)
    #print('Velocity guesses:', vel_guesses)
    ############################################################################


    ############################################################################
    #---------------------------------------------------------------------------
    try:
        ########################################################################
        # Flatten maps
        #-----------------------------------------------------------------------
        mHa_vel_flat = mHa_vel.compressed()
        
        mHa_vel_ivar_flat = mHa_vel_ivar.compressed()
        ########################################################################
        
        ########################################################################
        # Fit velocity map using scipy.optimize.minimize with chi2 minimization
        #
        # Fits all 8 parameters at once
        #-----------------------------------------------------------------------
        print('Fitting entire map', flush=True)

        result_all = minimize(calculate_chi2_flat, 
                              np.concatenate([pos_guesses, vel_guesses]), 
                              method='Powell', 
                              args=(mHa_vel_flat, mHa_vel_ivar_flat, mHa_vel.mask, pix_scale_factor, fit_function),
                              bounds=np.concatenate([pos_bounds, vel_bounds]),
                              options={'disp':True})

        # Calculate the normalized chi2 for the fit
        result_all.fun /= (len(mHa_vel_flat) - len(result_all.x))
        ########################################################################
        
        """
        ########################################################################
        # Fit velocity map using only continuous velocity field
        #-----------------------------------------------------------------------
        print('Fitting continuous map', flush=True)
        #-----------------------------------------------------------------------
        # Remove spaxels that are not part of the continuous velocity field
        #-----------------------------------------------------------------------
        min_vel, max_vel, modified_mask = find_vel_bounds(mHa_vel, gal_ID)
        #-----------------------------------------------------------------------

        if np.sum(modified_mask == 0) == np.sum(mHa_vel.mask == 0):
            result_continuous = result_all
        else:
            #-------------------------------------------------------------------
            # Update velocity field
            #-------------------------------------------------------------------
            modified_mHa_vel = ma.array(mHa_vel.data, mask=modified_mask)
            modified_mHa_vel_ivar = ma.array(mHa_vel_ivar.data, mask=modified_mask)

            modified_mHa_vel_flat = modified_mHa_vel.compressed()
            modified_mHa_vel_ivar_flat = modified_mHa_vel_ivar.compressed()
            #-------------------------------------------------------------------

            #-------------------------------------------------------------------
            # Update maximum velocity initial guess
            #-------------------------------------------------------------------
            v_max_guess = ma.max(ma.abs(modified_mHa_vel)/np.sin(inclination_angle_guess))

            #print(v_max_guess)

            vel_guesses[0] = v_max_guess
            #-------------------------------------------------------------------

            #-------------------------------------------------------------------
            # Refit galaxy
            #-------------------------------------------------------------------
            result_continuous = minimize(calculate_chi2_flat, 
                                         np.concatenate([pos_guesses, vel_guesses]), 
                                         method='Powell', 
                                         args=(modified_mHa_vel_flat, modified_mHa_vel_ivar_flat, modified_mask, pix_scale_factor, fit_function),
                                         bounds=np.concatenate([pos_bounds, vel_bounds]),
                                         options={'disp':True})

            # Calculate normalize chi2 of fit
            result_continuous.fun /= (len(modified_mHa_vel_flat) - len(result_continuous.x))
        ########################################################################
        
        
        ########################################################################
        # Fit velocity field using the residual
        #-----------------------------------------------------------------------
        print('Fitting using residual', flush=True)
        result_residual = minimize(calculate_residual_flat, 
                                   np.concatenate([pos_guesses, vel_guesses]), 
                                   method='Powell', 
                                   args=(mHa_vel_flat, mHa_vel.mask, pix_scale_factor, fit_function),
                                   bounds=np.concatenate([pos_bounds, vel_bounds]),
                                   options={'disp':True})

        # Calculate chi2 of the fit
        result_residual.fun = calculate_chi2_flat(result_residual.x, 
                                                  mHa_vel_flat, 
                                                  mHa_vel_ivar_flat, 
                                                  mHa_vel.mask, 
                                                  pix_scale_factor, 
                                                  fit_function)

        # Calculate normalized chi2 of the fit
        result_residual.fun /= (len(mHa_vel_flat) - len(result_residual.x))
        ########################################################################
        
        
        ########################################################################
        # Fit velocity field using only spaxels with S/N > 5
        #-----------------------------------------------------------------------
        print('Fitting S/N > 5', flush=True)
        #-----------------------------------------------------------------------
        # Remove spaxels with S/N < 5% of the maximum S/N in the data map (up to 
        # a S/N = 1)
        #-----------------------------------------------------------------------
        SN = ma.abs(mHa_flux*ma.sqrt(mHa_flux_ivar))

        SN_cut = 5
        '''
        SN_cut = 0.05*np.max(SN)
        if SN_cut > 1:
            SN_cut = 1
        '''

        modified_mask = np.logical_or(mHa_vel.mask + mHa_flux.mask, SN < SN_cut)
        #-----------------------------------------------------------------------

        if np.sum(modified_mask == 0) == np.sum(mHa_vel.mask == 0):
            result_SN = result_all
        else:

            modified_mHa_vel = ma.array(mHa_vel.data, mask=modified_mask)
            modified_mHa_vel_ivar = ma.array(mHa_vel_ivar.data, mask=modified_mask)

            modified_mHa_vel_flat = modified_mHa_vel.compressed()
            modified_mHa_vel_ivar_flat = modified_mHa_vel_ivar.compressed()


            result_SN = minimize(calculate_chi2_flat, 
                                 np.concatenate([pos_guesses, vel_guesses]), 
                                 method='Powell', 
                                 args=(modified_mHa_vel_flat, modified_mHa_vel_ivar_flat, modified_mask, pix_scale_factor, fit_function),
                                 bounds=np.concatenate([pos_bounds, vel_bounds]),
                                 options={'disp':True})

            # Calculate normalized chi2 of fit
            result_SN.fun /= (len(modified_mHa_vel_flat) - len(result_SN.x))
        ########################################################################
        

        ########################################################################
        # Fit velocity map using only spaxels with small line widths
        #-----------------------------------------------------------------------
        print('Fitting non-AGN map', flush=True)
        #-----------------------------------------------------------------------
        # Remove spaxels with large line widths
        #-----------------------------------------------------------------------
        max_sigma, modified_mask_sigma = find_sigma_bounds(mHa_sigma, gal_ID)
        #-----------------------------------------------------------------------

        if np.sum(modified_mask_sigma == 0) == np.sum(mHa_vel.mask == 0):
            result_nonAGN = result_all
        else:
            #-------------------------------------------------------------------
            # Update velocity field
            #-------------------------------------------------------------------
            modified_mHa_vel = ma.array(mHa_vel.data, mask=modified_mask_sigma)
            modified_mHa_vel_ivar = ma.array(mHa_vel_ivar.data, mask=modified_mask_sigma)

            modified_mHa_vel_flat = modified_mHa_vel.compressed()
            modified_mHa_vel_ivar_flat = modified_mHa_vel_ivar.compressed()
            #-------------------------------------------------------------------

            #-------------------------------------------------------------------
            # Update maximum velocity initial guess
            #-------------------------------------------------------------------
            v_max_guess = ma.max(ma.abs(modified_mHa_vel)/np.sin(inclination_angle_guess))

            vel_guesses[0] = v_max_guess
            #-------------------------------------------------------------------

            #-------------------------------------------------------------------
            # Refit galaxy
            #-------------------------------------------------------------------
            result_nonAGN = minimize(calculate_chi2_flat, 
                                     np.concatenate([pos_guesses, vel_guesses]), 
                                     method='Powell', 
                                     args=(modified_mHa_vel_flat, modified_mHa_vel_ivar_flat, modified_mask_sigma, pix_scale_factor, fit_function),
                                     bounds=np.concatenate([pos_bounds, vel_bounds]),
                                     options={'disp':True})

            # Calculate normalized chi2 of fit
            result_nonAGN.fun /= (len(modified_mHa_vel_flat) - len(result_nonAGN.x))
        ########################################################################


        ########################################################################
        # Choose the best fit (the one with the lowest chi2)
        # 
        # This needs to be based on the normalized chi2.
        #-----------------------------------------------------------------------
        fit_chi2 = np.inf*np.ones(5)

        alpha_max = alpha_high - 5
        
        if (result_all.x[7] < alpha_max) and (result_all.fun > 0):
            fit_chi2[0] = result_all.fun
        if (result_continuous.x[7] < alpha_max) and (result_continuous.fun > 0):
            fit_chi2[1] = result_continuous.fun
        if (result_SN.x[7] < alpha_max) and (result_SN.fun > 0):
            fit_chi2[2] = result_SN.fun
        if (result_residual.x[7] < alpha_max) and (result_residual.fun > 0):
            fit_chi2[3] = result_residual.fun
        if (result_nonAGN.x[7] < alpha_max) and (result_nonAGN.fun > 0):
            fit_chi2[4] = result_nonAGN.fun

        print(fit_chi2)

        if np.sum(np.isfinite(fit_chi2)) == 0:
            print('All fit methods have bad alpha values and/or negative chi2.', 
                  flush=True)
            min_pos = 0
        else:
            min_pos = np.argmin(fit_chi2)

        if min_pos == 0:
            fit_flag = -1
            result = result_all
        elif min_pos == 1:
            fit_flag = -3
            result = result_continuous
        elif min_pos == 2:
            fit_flag = SN_cut
            result = result_SN
        elif min_pos == 3:
            fit_flag = -2
            result = result_residual
        else:
            fit_flag = -4
            result = result_nonAGN
        ########################################################################
        """
        
        result = result_all
        fit_flag = -1


        if result.success:
            print('Successful velocity fit!', flush=True)

            #print(result)

            #-------------------------------------------------------------------
            # Determine uncertainties in the fitted parameters
            #-------------------------------------------------------------------
            # For use with L-BFGS-B method
            #fit_params_err = np.sqrt(np.diag(result.hess_inv.todense()))

            # Generic errors
            #fit_params_err = -np.ones(len(result.x))

            # For use when Hessian is not provided by fit method
            #hessian = ndt.Hessian(calculate_chi2)
            #hess = hessian(result.x, mHa_vel, mHa_vel_ivar, pix_scale_factor, fit_function)
            hessian = ndt.Hessian(calculate_chi2_flat)
            hess = hessian(result.x, mHa_vel_flat, mHa_vel_ivar_flat, mHa_vel.mask, pix_scale_factor, fit_function)

            # Save Hessian matrix (for uncertainty calculations)
            np.save('DRP_map_Hessians/' + gal_ID + '_Hessian.npy', hess)
            #print('Hessian:', hess)
            try:
                hess_inv = 2*np.linalg.inv(hess)
                fit_params_err = np.sqrt(np.diag(np.abs(hess_inv)))
            except np.linalg.LinAlgError:
                fit_params_err = np.nan*np.ones(len(result.x))
            #-------------------------------------------------------------------


            #-------------------------------------------------------------------
            # Unpack results
            #-------------------------------------------------------------------
            ba = np.sqrt(np.cos(result.x[1])**2*(1 - q0**2) + q0**2)
            ba_err = (fit_params_err[1]/np.sqrt(ba))*(1 - q0**2)*np.sin(result.x[1])*np.cos(result.x[1])

            best_fit_values = {'v_sys': result.x[0],
                               'v_sys_err': fit_params_err[0],
                               'ba': ba, 
                               'ba_err': ba_err, 
                               #'ba': np.cos(result.x[1]),
                               #'ba_err': np.cos(fit_params_err[1]),
                               'x0': result.x[2],
                               'x0_err': fit_params_err[2],
                               'y0': result.x[3],
                               'y0_err': fit_params_err[3],
                               'phi': result.x[4]*180/np.pi,
                               'phi_err': fit_params_err[4]*180/np.pi,
                               'v_max': result.x[5],
                               'v_max_err': fit_params_err[5],
                               'r_turn': result.x[6],
                               'r_turn_err': fit_params_err[6], 
                               'chi2': result.fun}

            if fit_function == 'BB':
                best_fit_values['alpha'] = result.x[7]
                best_fit_values['alpha_err'] = fit_params_err[7]
            #-------------------------------------------------------------------

            best_fit_map = model_vel_map(result.x,
                                         mHa_vel.shape, 
                                         pix_scale_factor, 
                                         fit_function)
        else:
            print('Fit did not converge.', flush=True)
            best_fit_values = None
            best_fit_map = None
            fit_flag = None
        ########################################################################
    except RuntimeError:
        best_fit_values = None
        best_fit_map = None
        fit_flag = None
    ############################################################################
        

    return best_fit_values, best_fit_map, pix_scale_factor, fit_flag




################################################################################
################################################################################
################################################################################




def mass_newton(v,v_err,r):
    '''
    Calculate the mass within radius r orbiting with a velocity v.


    PARAMETERS
    ==========

    v : float
        Velocity at radius r [km/s]

    v_err : float
        Uncertainty in the velocity [km/s]

    r : float
        Radius within which to calculate the mass [kpc]


    RETURNS
    =======

    m : float
        Mass within radius r [Msun]
    '''


    ############################################################################
    # Convert radius to pc
    #---------------------------------------------------------------------------
    r_pc = r*1000
    ############################################################################


    ############################################################################
    # Calculate the mass using Newton's law of gravity
    #---------------------------------------------------------------------------
    m = v*v*r_pc/G
    ############################################################################


    ############################################################################
    # Calculate the uncertainty in the mass
    #---------------------------------------------------------------------------
    m_err = 2*m*(v_err/v)
    ############################################################################

    return m, m_err








