#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Created on Thur Feb 14 2019
@author: Jacob A. Smith
@version: 1.2

Extracts rotation curve data points from files written with rotation_curve_vX_X
and fits a function to the data given several parameters. A total mass for the
galaxy is then extracted from the v_max parameter and the stellar mass is then
subtracted to find the galaxy's dark matter mass.
"""

import numpy as np
from decimal import Decimal

from random import random

#from scipy.optimize import curve_fit
from scipy.optimize import minimize

from astropy.table import Table, QTable
import astropy.units as u
import astropy.constants as const


################################################################################
# Units
#-------------------------------------------------------------------------------
vel_unit = u.km / u.s

rad_unit = u.kpc
################################################################################



################################################################################
################################################################################
################################################################################

def rot_fit_BB( depro_radius, params):
    """
    Function to fit the rotation curve data to.


    PARAMETERS
    ==========
    
    depro_radius : float or ndarray of shape (n,)
        Deprojected radius as taken from the [PLATE]-[FIBERID] rotation curve 
        data file (in units of kpc); the "x" data of the rotation curve equation

    v_max : float
        The maximum velocity (or in the case of fitting the negative, the
        absolute value of the minimum velocity) parameter of the rotation curve 
        equation (given in km/s)

    r_turn : float
        The radius at which the rotation curve trasitions from increasing to 
        flat-body for the rotation curve equation (given in kpc)

    alpha : float
        The exponential parameter for the rotation curve equation


    RETURNS
    =======
        
    The rotation curve equation with the given '@param' parameters and
    'depro_radius' data
    """

    v_max, r_turn, alpha = params

    v = v_max * (np.abs(depro_radius) / (r_turn**alpha + np.abs(depro_radius)**alpha)**(1/alpha))

    v = v*np.sign(depro_radius)

    return v



################################################################################
################################################################################
################################################################################

def rot_loss_BB(params, r_data, v_data, scales):
    '''
    Calculate the loss function for the BB rotation curve fit, where
      loss = penalty + (1/N) sum (y_i - f(x_i))^2
    where
      penalty = sum parameter_i / scale_i


    PARAMETERS
    ==========
    
    params : list or array-like
        Best-fit values for the parameters of the fit.

    r_data : list or array-like
        Radii of the data points to be fit

    v_data : list or array-like
        Velocities of the data points to be fit

    scales : list or array-like
        Loss scale for the best-fit parameters.  Is the same length as params, 
        where scales[i] is the scale corresponding to params[i].


    RETURNS
    =======

    total_loss : scalar
        The total loss of the fit.
    '''

    residuals = v_data - rot_fit_BB(r_data, params)

    res_squared = residuals*residuals

    penalty = np.sum(params/scales)

    total_loss = np.sum(res_squared)/len(r_data) #+ penalty

    return total_loss


################################################################################
################################################################################
################################################################################

def logL_BB(params, r, v, v_err):
    '''
    Log likelihood of the data and the fit values for the BB fit function.


    PARAMETERS
    ==========

    params : list or ndarray
        List of 3 fit parameters

    r : ndarray
        Radius values for the data points

    v : ndarray
        Velocity values for the data points

    v_err : ndarray
        Velocity error values for the data points


    RETURNS
    =======

    logL : float
        Log likelihood of set velocity given model parameters
    '''

    lambda1 = rot_fit_BB(r, params)
    lambda1[lambda1 <= 0] = np.finfo( dtype=np.float64).tiny

    return np.sum( ((v - lambda1)/v_err)**2 + np.log(2*np.pi*np.array(v_err)**2))


################################################################################
################################################################################
################################################################################

def nlogL_BB(params, radius, velocity, velocity_err):
    '''
    Negative log likelihood, for minimizing.
    '''
    return -logL_BB(params, radius, velocity, velocity_err)


################################################################################
################################################################################
################################################################################

def rot_fit_tanh( depro_radius, params):
    """
    Function to fit the rotation curve data to.

    PARAMETERS
    ==========
    
    depro_radius : float
        Deprojected radius as taken from the [PLATE]-[FIBERID] rotation curve 
        data file (in units of kpc); the "x" data of the rotation curve equation

    v_max : float
        The maximum velocity (or in the case of fitting the negative, the
        absolute value of the minimum velocity) parameter of the rotation curve 
        equation (given in km/s)

    r_turn : float
        The radius at which the rotation curve trasitions from increasing to 
        flat-body for the rotation curve equation (given in kpc)


    RETURNS
    =======
        
    The rotation curve equation with the given '@param' parameters and
    'depro_radius' data
    """

    v_max, r_turn = params

    return v_max * np.tanh(depro_radius/r_turn)



################################################################################
################################################################################
################################################################################


def fit_data( depro_dist, rot_vel, rot_vel_err, fit_func, TRY_N):
    '''
    Fit the rotation curve data via rot_fit_func() with
    scipy.optimize.curve_fit().


    Parameters:
    ===========

    depro_dist : numpy array of shape (n,)
        Absolute value of the deprojected distances

    rot_vel : numpy array of shape (n,)
        Rotational velocity at each radius

    rot_vel_err : numpy array of shape (n,)
        Uncertainty in the rotational velocities

    fit_func : string
        Identifies which function to use to fit to the rotation curve.  
        Options include
          - 'BB' : eqn. 4 from Barrera-Ballesteros et al. (2018)
          - 'tanh' : hyperbolic tanh function

    TRY_N : integer
        Number of times to try finding the equation of the line of best fit 
        before generating a timeout error (see scipy.optimize.curve_fit
        documentation for more information)


    Returns:
    ========

        best_param : list
            Best-fit parameters from the rot_fit_func() as calculated from 
            scipy.optimize.curve_fit()

        param_err : list
            Uncertainties in the best-fit parameters calculated from the 
            square-root of the diagonal of the covarience matrix obtained in 
            fitting the data via scipy.optimize.curve_fit()

        chi_square_rot : float
            Goodness of fit statistic for the data fitted to rot_fit_func()
    '''

    ############################################################################
    # NOTES:
    #---------------------------------------------------------------------------
    # Following from 'rot_fit_func,' the following first guesses of the
    # parameters 'v_max,' 'r_turn,' and 'alpha' are described as such:
    #
    #   v_max_guess / v_min_guess:
    #       the absolute maximum and absolute minimum (respectively) of the data
    #       file in question; first guess of the 'v_max' parameter
    #
    #   r_turn_max_guess / r_turn_min_guess:
    #       the radius at which 'v_max' and 'v_min' are respectively found; 
    #       first guess for 'r_turn' parameter
    #
    #   alpha_guess: emperically-estimated, first guess of the 'alpha' parameter
    ############################################################################


    ############################################################################
    # Fitting function options
    #---------------------------------------------------------------------------
    fit_options = {'BB': rot_fit_BB,
                   'tanh': rot_fit_tanh
                  }
    ############################################################################


    ############################################################################
    # Find the array index of the point with the maximum rotational velocity and 
    # extract the rotational velocity at that point.
    #---------------------------------------------------------------------------
    v_max_index = np.argmax( rot_vel)
    v_max_guess = rot_vel[ v_max_index]

    #print("v_max_guess:", v_max_guess)
    ############################################################################


    ############################################################################
    # If the initial guesses for the maximum rotational velocity is greater than 
    # 0, continue with the fitting process.
    #
    # Note: The minimum bound on R_turn was originally 0.25*r_turn_guess.  This 
    #       introduced an artificial correlation between R_max and R_turn, where 
    #       R_turn = 0.25*R_max (because r_turn_guess is often equal to R_max).  
    #       Similar effects were seen on the upper bound of R_turn (originally 
    #       2*r_turn_guess), and on V_max (originally 0.5*v_max_guess -> 
    #       1.5*v_max_guess).
    #---------------------------------------------------------------------------
    if v_max_guess > 0:

        # Rturn
        r_turn_guess = 0.5*depro_dist[ v_max_index]
        r_turn_low = 0.5 # kpc
        r_turn_high = 100. # kpc
        r_turn_bounds = (r_turn_low, r_turn_high)
        r_turn_scale = 3.

        # Vmax
        v_max_low = 1.0     # km/s
        v_max_high = 41000.0 # km/s
        v_max_bounds = (v_max_low, v_max_high)
        v_max_scale = 500.

        if fit_func == 'BB':
            alpha_guess = 2.
            alpha_low = np.nextafter(0, 1)
            alpha_high = 100.
            alpha_bounds = (alpha_low, alpha_high)
            alpha_scale = 10.

            scales = [v_max_scale, r_turn_scale, alpha_scale]

            rot_param_guess = [ v_max_guess, r_turn_guess, alpha_guess]
            rot_param_bounds = [v_max_bounds, r_turn_bounds, alpha_bounds]
            #rot_param_low = ( v_max_low, r_turn_low, np.nextafter(0, 1))
            #rot_param_high = ( v_max_high, r_turn_high, 10.)
        else:
            scales = [v_max_scale, r_turn_scale]

            rot_param_guess = [ v_max_guess, r_turn_guess]
            rot_param_bounds = [v_max_bounds, r_turn_bounds]
            #rot_param_low = ( v_max_low, r_turn_low)
            #rot_param_high = ( v_max_high, r_turn_high)

        #print(rot_param_guess, rot_param_high)

        '''
        ########################################################################
        # Print statement to track the first guesses for the 'rot_fit_func'
        # parameters.
        #-----------------------------------------------------------------------
        print("Rot Parameter Guess:", rot_param_guess)
        ########################################################################
        '''

        try:
            '''
            #-------------------------------------------------------------------
            # Fit using scipy.optimize.curve_fit
            #-------------------------------------------------------------------
            rot_popt, rot_pcov = curve_fit( fit_options[fit_func], 
                                            depro_dist, 
                                            rot_vel,
                                            p0=rot_param_guess,
                                            sigma=rot_vel_err,
                                            bounds=(rot_param_low, rot_param_high),
                                            max_nfev=TRY_N, 
                                            method='trf'
                                            #loss='cauchy'
                                            )

            rot_perr = np.sqrt( np.diag( rot_pcov))
            #-------------------------------------------------------------------
            '''
            '''
            #-------------------------------------------------------------------
            # Fit using scipy.optimize.minimize with simple custom loss function
            #-------------------------------------------------------------------
            results = minimize(rot_loss_BB, 
                               rot_param_guess, 
                               args=(depro_dist, rot_vel, scales), 
                               bounds=rot_param_bounds
                              )

            rot_perr = np.sqrt(np.diag(results.hess_inv.todense()))
            #-------------------------------------------------------------------
            '''
            #-------------------------------------------------------------------
            # Fit using scipy.optimize.minimize with negative log-likelihood 
            # minimization
            #-------------------------------------------------------------------
            results = None

            for i in range(30):
                p0 = [np.random.uniform(b[0], b[1]) for b in rot_param_bounds]
                result = minimize( nlogL_BB, 
                                   p0, 
                                   method='L-BFGS-B', 
                                   args=(depro_dist, rot_vel, rot_vel_err), 
                                   bounds=rot_param_bounds)
                if result.success:
                    if results is None:
                        results = result
                    else:
                        if result.fun < results.fun:
                            results = result

            rot_perr = np.sqrt(np.diag(results.hess_inv.todense()))
            #-------------------------------------------------------------------

            v_max_best = results.x[0]
            r_turn_best = results.x[1]
            
            v_max_sigma = rot_perr[0]
            r_turn_sigma = rot_perr[1]

            if fit_func == 'BB':
                alpha_best = results.x[2]
                alpha_sigma = rot_perr[2]


            chi_square_rot = 0
            for radius, velocity, vel_err in zip( depro_dist, rot_vel, rot_vel_err):
                observed = velocity

                if fit_func == 'BB':
                    expected = rot_fit_BB( radius, [v_max_best, r_turn_best, alpha_best])
                else:
                    expected = rot_fit_tanh( radius, v_max_best, r_turn_best)
                
                error = vel_err

                chi_square_rot += ((observed - expected) / error)**2

            if chi_square_rot == float('inf'):
                chi_square_rot = -50

        except RuntimeError:
            v_max_best = -999
            r_turn_best = -999

            v_max_sigma = -999
            r_turn_sigma = -999

            if fit_func == 'BB':
                alpha_best = -999
                alpha_sigma = -999

            chi_square_rot = -999
    # ~   ~   ~   ~   ~   ~   ~   ~   ~   ~   ~   ~   ~   ~   ~   ~   ~   ~   ~


    ############################################################################
    # If the maximum velocity of the data file in question is less than or equal 
    # to zero, the best fit parameters and their errors are set to -100.
    #---------------------------------------------------------------------------
    else:
        v_max_best = -100
        r_turn_best = -100

        v_max_sigma = -100
        r_turn_sigma = -100

        if fit_func == 'BB':
            alpha_best = -100
            alpha_sigma = -100

        chi_square_rot = -100
    ############################################################################

    if fit_func == 'BB':
        best_param = ( v_max_best, r_turn_best, alpha_best)
        param_err = ( v_max_sigma, r_turn_sigma, alpha_sigma)
    else:
        best_param = ( v_max_best, r_turn_best)
        param_err = ( v_max_sigma, r_turn_sigma)

    return best_param, param_err, chi_square_rot



###############################################################################
###############################################################################
###############################################################################


def fit_rot_curve( rot_curve_file, gal_stat_file, fit_func, TRY_N, points_to_cut=0):
    '''
    Finds the function of the line of best fit TRY_N times for the rotation
    curve data file in question. Returns back a table containg the best fit
    parameters for the fit function obtained with scipy.optimize.curve_fit.


    Parameters:
    ===========

    rot_curve_files : string
        Rotation curve file name in the 'rotation_curve_vX_X' output folder

    gal_stat_files : string
        Galaxy statistic file name within the 'rotation_curve_vX_X' output 
        folder

    fit_func : string
        Fitting function to use for rotation curve.  Options include
          - 'BB' : eqn. 4 from Barrera-Ballesteros et al. (2018)
          - 'tanh' : hyperbolic tanh function

    TRY_N : integer
        Number of times to try finding the equation of the line of best fit 
        before generating a timeout error (see scipy.optimize.curve_fit
        documentation for more information)

    points_to_cut : float
        Number of points to remove from the end of the rotation curve before 
        fitting.  Default is 0 (fit to all points).


    Returns:
    ========
        row_data_dic : dictionary
            Best fit parameters, center flux and its error, and stellar mass 
            processed
    '''

    ###########################################################################
    # Extract the data from the gal_stat_file.
    #--------------------------------------------------------------------------
    gal_stat_table = QTable.read( gal_stat_file, format='ascii.ecsv')
    gal_ID = gal_stat_table['gal_ID'][0]
    center_flux = gal_stat_table['center_flux'][0]
    center_flux_err = gal_stat_table['center_flux_error'][0]
    frac_masked_spaxels = gal_stat_table['frac_masked_spaxels'][0]

    print("gal_ID IN FUNC:", gal_ID)
    ###########################################################################


    ###########################################################################
    # Import the necessary data from the rotation curve files.
    #--------------------------------------------------------------------------
    rot_data_table = QTable.read( rot_curve_file, format='ascii.ecsv')


    depro_radii_fit = np.abs( rot_data_table['deprojected_distance'])

    rot_vel_avg = rot_data_table['rot_vel_avg']
    rot_vel_pos = rot_data_table['max_velocity']
    rot_vel_neg = np.abs( rot_data_table['min_velocity'])

    rot_vel_avg_err = rot_data_table['rot_vel_avg_error']
    rot_vel_pos_err = rot_data_table['max_velocity_error']
    rot_vel_neg_err = rot_data_table['min_velocity_error']


    if points_to_cut > 0:
        depro_radii_fit = depro_radii_fit[:-points_to_cut]

        rot_vel_avg = rot_vel_avg[:-points_to_cut]
        rot_vel_pos = rot_vel_pos[:-points_to_cut]
        rot_vel_neg = rot_vel_neg[:-points_to_cut]

        rot_vel_avg_err = rot_vel_avg_err[:-points_to_cut]
        rot_vel_pos_err = rot_vel_pos_err[:-points_to_cut]
        rot_vel_neg_err = rot_vel_neg_err[:-points_to_cut]
    ############################################################################


    ############################################################################
    # Best fit parameters and errors in those parameters are initialized to be
    # -1 to indicate that they start out as 'not found.' A value of -1 in the 
    # master file therefore indicates that the galaxy was not fitted because 
    # there was insufficent data to find the best fit parameters and their 
    # errors.
    #---------------------------------------------------------------------------
    avg_v_max = -1.* vel_unit
    pos_v_max = -1.* vel_unit
    neg_v_max = -1.* vel_unit

    avg_r_turn = -1.* rad_unit
    pos_r_turn = -1.* rad_unit
    neg_r_turn = -1.* rad_unit

    avg_v_max_sigma = -1.* vel_unit
    pos_v_max_sigma = -1.* vel_unit
    neg_v_max_sigma = -1.* vel_unit

    avg_r_turn_sigma = -1.* rad_unit
    pos_r_turn_sigma = -1.* rad_unit
    neg_r_turn_sigma = -1.* rad_unit

    if fit_func == 'BB':
        avg_alpha = -1.
        pos_alpha = -1.
        neg_alpha = -1.

        avg_alpha_sigma = -1.
        pos_alpha_sigma = -1.
        neg_alpha_sigma = -1.

    avg_chi_square_rot = -1.
    pos_chi_square_rot = -1.
    neg_chi_square_rot = -1.

    avg_chi_square_ndf = -1.
    pos_chi_square_ndf = -1.
    neg_chi_square_ndf = -1.
    ###########################################################################


    ###########################################################################
    # If there are more than three data points, fit the rotation curve data to
    #    'rot_fit_func()' via 'scipy.optimize.curve_fit().'
    # ~   ~   ~   ~   ~   ~   ~   ~   ~   ~   ~   ~   ~   ~   ~   ~   ~   ~   ~
    #print("length rot_data_table:", len( rot_data_table))
    N = len( depro_radii_fit)

    if N > 3:

        #######################################################################
        # Fit the rotation curve data and extract the best fit parameters,
        #    their errors, and the chi-square (goodness of fit) statistic.
        #----------------------------------------------------------------------
        # Average rotation curve
        avg_fit_param, avg_fit_param_err, avg_chi_square_rot = fit_data( depro_radii_fit.data,
                                                                         rot_vel_avg.data, 
                                                                         rot_vel_avg_err.data,
                                                                         fit_func, 
                                                                         TRY_N)
        avg_v_max = avg_fit_param[0] * (u.km / u.s)
        avg_r_turn = avg_fit_param[1] * (u.kpc)

        avg_v_max_sigma = avg_fit_param_err[0] * (u.km / u.s)
        avg_r_turn_sigma = avg_fit_param_err[1] * (u.kpc)

        if fit_func == 'BB':
            avg_alpha = avg_fit_param[2]
            avg_alpha_sigma = avg_fit_param_err[2]

        avg_chi_square_ndf = avg_chi_square_rot/(N - 3)


        # positive rotation curve data
        pos_fit_param, pos_fit_param_err, pos_chi_square_rot = fit_data( depro_radii_fit.data,
                                                                         rot_vel_pos.data, 
                                                                         rot_vel_pos_err.data,
                                                                         fit_func, 
                                                                         TRY_N)
        pos_v_max = pos_fit_param[0] * (u.km / u.s)
        pos_r_turn = pos_fit_param[1] * (u.kpc)

        pos_v_max_sigma = pos_fit_param_err[0] * (u.km / u.s)
        pos_r_turn_sigma = pos_fit_param_err[1] * (u.kpc)

        if fit_func == 'BB':
            pos_alpha = pos_fit_param[2]
            pos_alpha_sigma = pos_fit_param_err[2]

        pos_chi_square_ndf = pos_chi_square_rot/(N - 3)


        # negative rotation curve data
        neg_fit_param, neg_fit_param_err, neg_chi_square_rot = fit_data( depro_radii_fit.data,
                                                                         rot_vel_neg.data, 
                                                                         rot_vel_neg_err.data,
                                                                         fit_func, 
                                                                         TRY_N)
        neg_v_max = neg_fit_param[0] * (u.km / u.s)
        neg_r_turn = neg_fit_param[1] * (u.kpc)

        neg_v_max_sigma = neg_fit_param_err[0] * (u.km / u.s)
        neg_r_turn_sigma = neg_fit_param_err[1] * (u.kpc)

        if fit_func == 'BB':
            neg_alpha = neg_fit_param[2]
            neg_alpha_sigma = neg_fit_param_err[2]

        neg_chi_square_ndf = neg_chi_square_rot/(N - 3)
        #######################################################################

        '''
        #######################################################################
        # Print statement to track the best fit parameters for the data file in
        #    question as well as the chi square (goodness of fit) statistic
        #----------------------------------------------------------------------
        print("Rot Curve Parameters:", avg_fit_param)
        print("Rot Curve Parameters (Positive):", pos_fit_param)
        print("Rot Curve Parameters (Negative):", neg_fit_param)
        print("Chi^{2}:", avg_chi_square_rot)
        print("Chi^{2} (Pos):", pos_chi_square_rot)
        print("Chi^{2} (Neg):", neg_chi_square_rot)
        print("-----------------------------------------------------")
        #######################################################################
        '''
    ###########################################################################


    ###########################################################################
    # Create a dictionary to house each galaxy's data.
    #
    # NOTE: All entries are initialized to be -1s to indicate they are not
    #       found.
    #--------------------------------------------------------------------------
    row_data_dict = {'center_flux': center_flux,
                    'center_flux_error': center_flux_err,
                    'frac_masked_spaxels': frac_masked_spaxels,
                    'avg_v_max': avg_v_max,
                    'avg_r_turn': avg_r_turn,
                    'avg_v_max_sigma': avg_v_max_sigma,
                    'avg_r_turn_sigma': avg_r_turn_sigma,
                    'avg_chi_square_rot': avg_chi_square_rot,
                    'avg_chi_square_ndf': avg_chi_square_ndf,
                    'pos_v_max': pos_v_max,
                    'pos_r_turn': pos_r_turn,
                    'pos_v_max_sigma': pos_v_max_sigma,
                    'pos_r_turn_sigma': pos_r_turn_sigma,
                    'pos_chi_square_rot': pos_chi_square_rot,
                    'pos_chi_square_ndf': pos_chi_square_ndf,
                    'neg_v_max': neg_v_max,
                    'neg_r_turn': neg_r_turn,
                    'neg_v_max_sigma': neg_v_max_sigma,
                    'neg_r_turn_sigma': neg_r_turn_sigma,
                    'neg_chi_square_rot': neg_chi_square_rot,
                    'neg_chi_square_ndf': neg_chi_square_ndf}

    if fit_func == 'BB':
        row_data_dict['avg_alpha'] = avg_alpha
        row_data_dict['pos_alpha'] = pos_alpha
        row_data_dict['neg_alpha'] = neg_alpha

        row_data_dict['avg_alpha_sigma'] = avg_alpha_sigma
        row_data_dict['pos_alpha_sigma'] = pos_alpha_sigma
        row_data_dict['neg_alpha_sigma'] = neg_alpha_sigma
    ###########################################################################

    return row_data_dict



###############################################################################
###############################################################################
###############################################################################


def estimate_dark_matter(parameter_dict, 
                         fit_func, 
                         chi2_max, 
                         rot_curve_file, 
                         gal_stat_file):
    '''
    Estimate the total mass interior to a radius from the fitted v_max
    parameter and the last recorded radius for the galaxy.  Then estimate the
    total dark matter interior to that radius by subtracting the stellar mass
    interior to that radius.


    Parameters:
    ===========

    parameter_dict : dictionary
        Best fit parameters for each galaxy along with the errors associated 
        with them and the chi-square goodness of fit statistics

    fit_func : string
        Identifies which function to use to fit to the rotation curve.  
        Options include
          - 'BB' : eqn. 4 from Barrera-Ballesteros et al. (2018)
          - 'tanh' : hyperbolic tanh function

    chi2_max : float
        Maximum allowed normalized chi2 (chi2/DOF) for a successful fit.

    rot_curve_file : string
        File name of data file containing rotation curve data

    gal_stat_file : string
        File name of galaxy statistics file


    Returns:
    ========

    row_data_dict : dictionary
        Contains the stellar, dark matter, and total mass estimates interior to 
        the outermost radius in the galaxy
    '''

    
    ############################################################################
    # Find the optimal fit parameters to use for estimating the mass
    #---------------------------------------------------------------------------
    parameter_dict, curve_used, points_cut = parameter_restrictions(fit_func, 
                                                                    chi2_max, 
                                                                    parameter_dict, 
                                                                    0, 
                                                                    rot_curve_file,
                                                                    gal_stat_file)
    '''
    curve_used = 'avg'
    points_cut = 0
    '''
    ############################################################################
    
    
    ############################################################################
    # Define maximum velocities and turn radius to use for mass estimates
    #---------------------------------------------------------------------------
    if curve_used == 'none':
        v_max = -1. * ( u.km / u.s)
    else:
        v_max_field = curve_used + '_v_max'
        v_max_sigma_field = curve_used + '_v_max_sigma'
    
        v_max = parameter_dict[v_max_field]
        v_max_sigma = parameter_dict[v_max_sigma_field]
    ############################################################################
    

    ############################################################################
    # For each galaxy in the 'master_file,' calculate the total mass interior to
    # the final radius recorded of the galaxy.  Then, subtract the stellar mass
    # interior to that radius to find the theoretical estimate for the dark 
    # matter interior to that radius.
    #
    # In addition, the errors associated with the total mass and dark matter
    # mass interior to a radius are calculated.
    #--------------------------------------------------------------------------
    rot_curve_table = QTable.read( rot_curve_file, format='ascii.ecsv')
    depro_dist = rot_curve_table['deprojected_distance']
    Mstar_interior = rot_curve_table['sMass_interior']

    Mstar_processed = Mstar_interior[-1]


    if (v_max.value == -1) or (v_max.value == -100) or (v_max.value == -999):
        gal_mass = -1. * u.M_sun
        gal_mass_err = -1. * u.M_sun

        theorized_Mdark = -1. * u.M_sun
        theorized_Mdark_err = -1. * u.M_sun

        Mdark_Mstar_ratio = -1.
        Mdark_Mstar_ratio_err = -1.

        Mtot_Mstar_ratio = -1.
        Mtot_Mstar_ratio_err = -1.

    else:
        depro_dist_end = np.abs( depro_dist[-1])
        depro_dist_end_m = depro_dist_end.to('m')

        v_max_m_per_s = v_max.to('m/s')
        v_max_sigma_m_per_s = v_max_sigma.to('m/s')

        gal_mass = v_max_m_per_s**2 * depro_dist_end_m / const.G
        gal_mass = gal_mass.to('M_sun')
        #gal_mass /= u.M_sun  # strip 'gal_mass' of its units

        gal_mass_err = gal_mass * np.sqrt( (2 * v_max_sigma_m_per_s / v_max_m_per_s)**2 \
                                         + (const.G.uncertainty * const.G.unit / const.G)**2)

        theorized_Mdark = gal_mass - Mstar_processed
        theorized_Mdark_err = gal_mass_err  # no uncertainties given for stellar mass density from Pipe3D

        Mdark_Mstar_ratio = theorized_Mdark / Mstar_processed
        Mdark_Mstar_ratio_err = theorized_Mdark_err / Mstar_processed

        Mtot_Mstar_ratio = gal_mass / Mstar_processed
        Mtot_Mstar_ratio_err = gal_mass_err / Mstar_processed
        '''
        print('Total mass:', gal_mass)
        print('Stellar mass:', Mstar_processed)
        print('Uncertainty in v_max:', v_max_sigma_m_per_s.to('km/s'))
        print('v_max:', v_max_m_per_s.to('km/s'))
        print('Ratio error:', Mdark_Mstar_ratio_err)
        '''
    ###########################################################################


    ###########################################################################
    # Add new output fields to parameter_dict
    #--------------------------------------------------------------------------
    parameter_dict['Rmax'] = depro_dist[-1]

    parameter_dict['Mtot'] = gal_mass
    parameter_dict['Mtot_error'] = gal_mass_err

    parameter_dict['Mdark'] = theorized_Mdark
    parameter_dict['Mdark_error'] = theorized_Mdark_err

    parameter_dict['Mstar'] = Mstar_processed

    parameter_dict['Mdark_Mstar_ratio'] = Mdark_Mstar_ratio
    parameter_dict['Mdark_Mstar_ratio_error'] = Mdark_Mstar_ratio_err

    parameter_dict['Mtot_Mstar_ratio'] = Mtot_Mstar_ratio
    parameter_dict['Mtot_Mstar_ratio_error'] = Mtot_Mstar_ratio_err

    parameter_dict['curve_used'] = curve_used
    parameter_dict['points_cut'] = points_cut
    ###########################################################################

    return parameter_dict



###############################################################################
###############################################################################
###############################################################################



def parameter_restrictions(fit_func, 
                           chi2_max, 
                           fitted_parameters, 
                           points_cut, 
                           rot_curve_file, 
                           gal_stat_file
                          ):
    '''
    Implement desired cuts to 'clean' the galaxy sample


    Parameters:
    ===========

    fit_func : string
        Identifies which function to use to fit to the rotation curve.  
        Options include
          - 'BB' : eqn. 4 from Barrera-Ballesteros et al. (2018)
          - 'tanh' : hyperbolic tanh function

    chi2_max : float
        Maximum allowed value of normalized chi2 parameter (chi2/DOF) for a 
        successful fit.

    fitted_parameters : dictionary
        Contains all fit parameters and statistics

    points_cut : float
        Number of points cut from rotation curve

    rot_curve_file : string
        File name of rotation curve data


    Returns:
    ========

    fitted_parameters : dictionary
        Contains all parameters from fits to rotation curves

    curve_used : string
        Which rotation curve to use for the best fit.  Options are 'avg', 
        'pos', 'neg', or 'none'

    points_cut : float
        Number of data points removed from the end of the rotation curve 
        sample to achieve the best fit.
    '''

    ############################################################################
    # Measure total points available in rotation curve
    #---------------------------------------------------------------------------
    rot_data_table = Table.read(rot_curve_file, format='ascii.ecsv')
    total_points = len(rot_data_table)
    ############################################################################


    ############################################################################
    # Check chi^2 of average curve
    #---------------------------------------------------------------------------
    if (fitted_parameters['avg_chi_square_ndf'] < chi2_max) and (fitted_parameters['avg_chi_square_rot'] >= 0):
        curve_used = 'avg'
    #---------------------------------------------------------------------------
    # Check chi^2 of positive curve
    #---------------------------------------------------------------------------
    elif (fitted_parameters['pos_chi_square_ndf'] < chi2_max) and (fitted_parameters['pos_chi_square_rot'] >= 0):
        curve_used = 'pos'
    #---------------------------------------------------------------------------
    # Check chi^2 of negative curve
    #---------------------------------------------------------------------------
    elif (fitted_parameters['neg_chi_square_ndf'] < chi2_max) and (fitted_parameters['neg_chi_square_rot'] >= 0):
        curve_used = 'neg'
    #---------------------------------------------------------------------------
    # Remove one point from curves and refit
    #---------------------------------------------------------------------------
    elif points_cut < 0.5*total_points:
        points_cut += 1

        ########################################################################
        # Refit galaxy
        #-----------------------------------------------------------------------
        fitted_parameters = fit_rot_curve( rot_curve_file, 
                                           gal_stat_file, 
                                           fit_func, 
                                           100000, 
                                           points_cut
                                          )
        ########################################################################


        ########################################################################
        # Check new fitted parameters
        #-----------------------------------------------------------------------
        fitted_parameters, curve_used, points_cut = parameter_restrictions( fit_func, 
                                                                            chi2_max, 
                                                                            fitted_parameters, 
                                                                            points_cut,
                                                                            rot_curve_file,
                                                                            gal_stat_file)
        ########################################################################
    #---------------------------------------------------------------------------
    # Galaxy fit is not good
    #---------------------------------------------------------------------------
    else:
        curve_used = 'none'
    ############################################################################


    return fitted_parameters, curve_used, points_cut