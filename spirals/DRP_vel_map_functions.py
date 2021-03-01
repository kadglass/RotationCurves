
import numpy as np
import numpy.ma as ma

import numdifftools as ndt

from skimage.filters import gaussian

from scipy.optimize import minimize

from dark_matter_mass_v1 import rot_fit_BB, rot_fit_tanh





################################################################################
# Constants
#-------------------------------------------------------------------------------
H_0 = 100      # Hubble's Constant in units of h km/s/Mpc
c = 299792.458 # Speed of light in units of km/s
G = 4.30091E-3 # Gravitation constant in units of (km/s)^2 pc/Msun

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
    dx_prime = (delta[1]*np.cos(phi) + delta[0]*np.sin(phi))/np.cos(i_angle)

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
        print('Unknown fit function.  Please update model_vel_map function.')
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
                print('Fit function not known.  Please update model_vel_map function.')

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
def find_vel_map(mHa_vel, 
                 mHa_vel_ivar, 
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

    mHa_vel : numpy array of shape (n,n)
        H-alpha velocity field data

    Ha_vel_ivar : numpy array of shape (n,n)
        Inverse variance in the H-alpha velocity field data

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
    # Convert pixel distance to physical distances in units of both
    # kiloparsecs and centimeters.
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
    v_max_index = np.unravel_index(ma.argmax(mHa_vel), mHa_vel.shape)
    v_max_guess = mHa_vel[v_max_index]/np.sin(inclination_angle_guess)

    #print("v_max_guess:", v_max_guess)
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
    ############################################################################


    ############################################################################
    # Set the extremes for each of the fit parameters
    #---------------------------------------------------------------------------
    # Systemic velocity
    sys_vel_low = -100
    sys_vel_high = 100
    sys_vel_bounds = (sys_vel_low, sys_vel_high)

    # Inclination angle
    inclination_angle_low = 0
    inclination_angle_high = 0.5*np.pi
    inclination_angle_bounds = (inclination_angle_low, inclination_angle_high)

    # Center coordinates
    i_center_low = i_center_guess - 5
    i_center_high = i_center_guess + 5
    i_center_bounds = (i_center_low, i_center_high)

    j_center_low = j_center_guess - 5
    j_center_high = j_center_guess + 5
    j_center_bounds = (j_center_low, j_center_high)

    # Orientation angle
    phi_low = 0
    phi_high = 2*np.pi
    phi_bounds = (phi_low, phi_high)

    # Maximum velocity [km/s]
    v_max_low = 10
    v_max_high = 4100
    v_max_bounds = (v_max_low, v_max_high)

    # Turn radius [kpc]
    r_turn_low = 0.5
    r_turn_high = 100
    r_turn_bounds = (r_turn_low, r_turn_high)
    ############################################################################


    ############################################################################
    # Set initial guesses, bounds for parameters specific to the fit function 
    # selected.
    #---------------------------------------------------------------------------
    if fit_function == 'BB':

        # Alpha
        alpha_guess = 2
        alpha_low = 0.001
        alpha_high = 15
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
        print('Selected fit function is not known!  Please edit find_vel_map function in DRP_vel_map_functions.py.')
    ############################################################################


    ############################################################################
    #---------------------------------------------------------------------------
    try:
        
        ########################################################################
        # Fit velocity map using scipy.optimize.minimize with chi2 minimization
        #
        # Fits all 8 parameters at once
        #-----------------------------------------------------------------------
        result = minimize(calculate_chi2,
                          np.concatenate([pos_guesses, vel_guesses]),
                          method='Powell',#'L-BFGS-B',
                          args=(mHa_vel, mHa_vel_ivar, pix_scale_factor, fit_function),
                          bounds=np.concatenate([pos_bounds, vel_bounds]), 
                          options={'disp':True}
                          )

        if result.success:
            print('Successful velocity fit!')

            #print(result)

            #-------------------------------------------------------------------
            # Determine uncertainties in the fitted parameters
            #-------------------------------------------------------------------
            # For use with L-BFGS-B method
            #fit_params_err = np.sqrt(np.diag(result.hess_inv.todense()))

            # Generic errors
            #fit_params_err = -np.ones(len(result.x))

            # For use when Hessian is not provided by fit method
            hessian = ndt.Hessian(calculate_chi2)
            hess = hessian(result.x, mHa_vel, mHa_vel_ivar, pix_scale_factor, fit_function)
            hess_inv = np.linalg.inv(hess)
            fit_params_err = np.sqrt(np.diag(hess_inv))
            #-------------------------------------------------------------------


            #-------------------------------------------------------------------
            # Unpack results
            #-------------------------------------------------------------------
            best_fit_values = {'v_sys': result.x[0],
                               'v_sys_err': fit_params_err[0],
                               'ba': np.cos(result.x[1]),
                               'ba_err': np.cos(fit_params_err[1]),
                               'x0': result.x[2],
                               'x0_err': fit_params_err[2],
                               'y0': result.x[3],
                               'y0_err': fit_params_err[3],
                               'phi': result.x[4]*180/np.pi,
                               'phi_err': fit_params_err[4]*180/np.pi,
                               'v_max': result.x[5],
                               'v_max_err': fit_params_err[5],
                               'r_turn': result.x[6],
                               'r_turn_err': fit_params_err[6]}

            if fit_function == 'BB':
                best_fit_values['alpha'] = result.x[7]
                best_fit_values['alpha_err'] = fit_params_err[7]
            #-------------------------------------------------------------------

            best_fit_map = model_vel_map(result.x,
                                         mHa_vel.shape, 
                                         pix_scale_factor, 
                                         fit_function)
        else:
            print('Fit did not converge.')
            best_fit_values = None
            best_fit_map = None
        ########################################################################
    except RuntimeError:
        best_fit_values = None
        best_fit_map = None
    ############################################################################
        

    return best_fit_values, best_fit_map, pix_scale_factor




################################################################################
################################################################################
################################################################################




def mass_newton(v,v_err,r,z):
    '''
    Calculate the mass within radius r orbiting with a velocity v.


    PARAMETERS
    ==========

    v : float
        Velocity at radius r [km/s]

    v_err : float
        Uncertainty in the velocity [km/s]

    r : float
        Radius within which to calculate the mass [arcsec]

    z : float
        Galaxy redshift


    RETURNS
    =======

    m : float
        Mass within radius r [Msun]
    '''


    ############################################################################
    # Convert radius to pc
    #---------------------------------------------------------------------------
    dist_to_galaxy_Mpc = c*z/H_0
    dist_to_galaxy_pc = dist_to_galaxy_Mpc*1000000

    r_pc = dist_to_galaxy_pc*np.tan(r*(1/60)*(1/60)*(np.pi/180))
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








