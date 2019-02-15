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
import matplotlib.pyplot as plt
import numpy as np
from decimal import Decimal

from scipy.optimize import curve_fit

from astropy.io import ascii
import astropy.units as u
import astropy.constants as const


def rot_fit_func( depro_radius, v_max, r_turn, alpha):
    """Function in which to fit the rotation curve data to.

    @param:
        depro_radius:
            float representation of the deprojected radius as taken from the
            [PLATE]-[FIBERID] rotation curve data file (in units of kpc); the
            "x" data of the rotation curve equation

        v_max:
            the maximum velocity (or in the case of fitting the minimum, the
            absolute value of the minimum velocity) parameter of the rotation
            curve equation (given in km/s)

        r_turn:
            the radius at which the rotation curve trasitions from increasing
            to flat-body parameter for the rotation curve equation (given in
            kpc)

        alpha:
            the exponential parameter for the rotation curve equation
            (unitless)

    @return:
        the rotation curve equation with the given '@param' parameters and
        'depro_radius' data

    """
    return v_max * \
           (depro_radius / (r_turn**alpha + depro_radius**alpha)**(1/alpha))


def fit_data( depro_dist, rot_vel, rot_vel_err, TRY_N):
    """Fit the rotation curve data via rot_fit_func() with
    scipy.optimize,curve_fit().

    @param:
        depro_dist:
            numpy array of the absolute value of the deprojected distance data
            for the galaxy in question

        rot_vel:
            numpy array of the rotational velocity data for the galaxy in
            question

        rot_vel_err:
            numpy array of the error in rotational velocity data for the galaxy
            in question

        TRY_N:
            int number of times to try finding the equation of the line of best
            fit before generating a timeout error (see scipy.optimize.curve_fit
            documentation for more information)

    @return:
        best_param:
            list of the best-fit parameters from the rot_fit_func() as
            calculated from scipy.optimize.curve_fit()

        param_err:
            error in the best-fit parameters calculated from the square-root
            of the diagnol of the covarience matrix obtained in fitting the
            data via scipy.optimize.curve_fit()

        chi_square_rot:
            goodness of fit statistic for the data fitted to rot_fit_func()
    """
    # ! # ! # ! # ! # ! # ! # ! # ! # ! # ! # ! # ! # ! # ! # ! # ! # ! # ! # !
    # NOTES:
    #--------------------------------------------------------------------------
    # Following from 'rot_fit_func,' the following first guesses of the
    #    parameters 'v_max,' 'r_turn,' and 'alpha' are described as such:
    #
    # v_max_guess / v_min_guess:
    #    the absolute maximum and absolute minimum (respectively) of the data
    #    file in question; first guess of the 'v_max' parameter
    #
    # r_turn_max_guess / r_turn_min_guess:
    #    the radius atwhich 'v_max' and 'v_min' are respectively found; first
    #    guess for 'r_turn' parameter
    #
    # alpha_guess: imperically-estimated, first guess of the 'alpha' parameter
    # ! # ! # ! # ! # ! # ! # ! # ! # ! # ! # ! # ! # ! # ! # ! # ! # ! # ! # !

    ###########################################################################
    # Find the array index of the point with the maximum rotational velocity
    #    and extract the rotational velocity at that point.
    #--------------------------------------------------------------------------
    v_max_loc = np.argmax( rot_vel)
    v_max_guess = rot_vel[ v_max_loc]

#    print("v_max_guess:", v_max_guess)
    ###########################################################################


    ###########################################################################
    # If the initial guesses for the maximum rotational velocity is greater
    #    than 0, continue with the fitting process.
    # ~   ~   ~   ~   ~   ~   ~   ~   ~   ~   ~   ~   ~   ~   ~   ~   ~   ~   ~
    if v_max_guess > 0:
        r_turn_guess = depro_dist[ v_max_loc]
        alpha_guess = 2

        rot_param_guess = [ v_max_guess, r_turn_guess, alpha_guess]

        '''
        #######################################################################
        # Print statement to track the first guesses for the 'rot_fit_func'
        #    parameters.
        #----------------------------------------------------------------------
        print("Rot Parameter Guess:", rot_param_guess)
        #######################################################################
        '''

        try:
            rot_popt, rot_pcov = curve_fit( rot_fit_func,
                                           depro_dist, rot_vel,
                                           p0 = rot_param_guess,
                                           sigma = rot_vel_err,
                                           bounds=( ( v_max_guess / 2,
                                                     r_turn_guess / 4,
                                                     np.nextafter( 0, 1)),
                                                    ( v_max_guess * 1.5,
                                                     r_turn_guess * 2,
                                                     np.inf)),
                                           max_nfev=TRY_N, loss='cauchy')

            rot_perr = np.sqrt( np.diag( rot_pcov))

            v_max_best = rot_popt[0]
            r_turn_best = rot_popt[1]
            alpha_best = rot_popt[2]

            v_max_sigma = rot_perr[0]
            r_turn_sigma = rot_perr[1]
            alpha_sigma = rot_perr[2]


            chi_square_rot = 0
            for radius, velocity, vel_err in \
              zip( depro_dist, rot_vel, rot_vel_err):
                observed = velocity
                expected = rot_fit_func( radius,
                                        v_max_best,
                                        r_turn_best,
                                        alpha_best)
                error = vel_err
                chi_square_rot += (observed - expected)**2 / error**2

            if chi_square_rot == float('inf'):
                chi_square_rot = -50

        except RuntimeError:
            v_max_best = -999
            r_turn_best = -999
            alpha_best = -999

            v_max_sigma = -999
            r_turn_sigma = -999
            alpha_sigma = -999
            chi_square_rot = -999
    # ~   ~   ~   ~   ~   ~   ~   ~   ~   ~   ~   ~   ~   ~   ~   ~   ~   ~   ~


    ###########################################################################
    # If the maximum velocity of the data file in question is less than or
    #    equal to zero, the best fit parameters and their errors are set to
    #    -100.
    #--------------------------------------------------------------------------
    elif v_max_guess <= 0:
        v_max_best = -100
        r_turn_best = -100
        alpha_best = -100

        v_max_sigma = -100
        r_turn_sigma = -100
        alpha_sigma = -100
        chi_square_rot = -100
    ###########################################################################

    best_param = ( v_max_best, r_turn_best, alpha_best)
    param_err = ( v_max_sigma, r_turn_sigma, alpha_sigma)

    return best_param, param_err, chi_square_rot


def fit_rot_curve( rot_curve_file, gal_stat_file, TRY_N):
    """Finds the function of the line of best fit TRY_N times for the rotation
    curve data file in question. Returns back a table containg the best fit
    parameters for the fit function obtained with scipy.optimize.curve_fit.

    @param:
        rot_curve_files:
            a string list containing all of the rotation curve files within the
            'rotation_curve_vX_X' output folder

        gal_stat_files:
            a string list containing all of the galaxy statistic files within
            the 'rotation_curve_vX_X' output folder

        TRY_N:
            int number of times to try finding the equation of the line of best
            fit before generating a timeout error (see scipy.optimize.curve_fit
            documentation for more information)

    @return:
        row_data_dic:
            dictionary containing the best fit parameters, center flux and its
            error, and stellar mass processed for a galaxy
    """
    ###########################################################################
    # Extract the data from the gal_stat_file.
    #--------------------------------------------------------------------------
    gal_stat_table = ascii.read( gal_stat_file, format='ecsv')
    gal_ID = gal_stat_table['gal_ID'][0]
    center_flux = gal_stat_table['center_flux'][0].value
    center_flux_err = gal_stat_table['center_flux_error'][0].value

    print("gal_ID IN FUNC:", gal_ID)
    ###########################################################################


    ###########################################################################
    # Import the necessary data from the rotation curve files.
    #--------------------------------------------------------------------------
    rot_data_table = ascii.read( rot_curve_file, format='ecsv')
    depro_radii_fit = np.abs( rot_data_table['deprojected_distance'].value)
    rot_vel_avg = rot_data_table['rot_vel_avg'].value
    rot_vel_avg_err = rot_data_table['rot_vel_avg_error'].value
    rot_vel_pos = rot_data_table['max_velocity'].value
    rot_vel_pos_err = rot_data_table['max_velocity_error'].value
    rot_vel_neg = np.abs( rot_data_table['min_velocity'].value)
    rot_vel_neg_err = rot_data_table['min_velocity_error'].value

#    print("depro_radii:", depro_radii)
#    print("rot_vel_avg:", rot_vel_avg)
#    print("rot_vel_avg_err:", rot_vel_avg_err)
#    print("rot_vel_max:", rot_vel_max)
#    print("rot_vel_max_err:", rot_vel_max_err)
#    print("rot_vel_min:", rot_vel_min)
#    print("rot_vel_min_err:", rot_vel_min_err)
    ###########################################################################


    ###########################################################################
    # Extract the total stellar mass processed for the galaxy from the last
    #    data point in the 'sMass_interior' column for the galaxy.
    #--------------------------------------------------------------------------
    sMass_interior = rot_data_table['sMass_interior'].value
    sMass_processed = sMass_interior[ -1]

#    print("sMass_processed:", sMass_processed)
    ###########################################################################

    '''
    ###########################################################################
    # General information about the data file in question.
    #--------------------------------------------------------------------------
    print( rotcurve__file, ":\n\n", rot_data_table, "\n\n")
    print("DATA TABLE INFORMATION \n",
          'Columns:', rot_data_table.columns, '\n',
          'Column Names:', rot_data_table.colnames, '\n',
          'Meta Data:', rot_data_table.meta, '\n',
          'Number of Rows:', len( rot_data_table))
    ###########################################################################
    '''

    ###########################################################################
    # Best fit parameters and errors in those parameters are initialized to be
    #    -1 to indicate that they start out as 'not found.' A value of -1 in
    #    the master file therefore indicates that the galaxy was not fitted
    #    because there was insufficent data to find the best fit parameters and
    #    their errors.
    #--------------------------------------------------------------------------
    v_max_best = -1
    r_turn_best = -1
    alpha_best = -1
    pos_v_max_best = -1
    pos_r_turn_best = -1
    pos_alpha_best = -1
    neg_v_max_best = -1
    neg_r_turn_best = -1
    neg_alpha_best = -1

    v_max_sigma = -1
    r_turn_sigma = -1
    alpha_sigma = -1
    chi_square_rot = -1
    pos_v_max_sigma = -1
    pos_r_turn_sigma = -1
    pos_alpha_sigma = -1
    pos_chi_square_rot = -1
    neg_v_max_sigma = -1
    neg_r_turn_sigma = -1
    neg_alpha_sigma = -1
    neg_chi_square_rot = -1
    ###########################################################################


    ###########################################################################
    # If there are more than two data points, fit the rotation curve data to
    #    'rot_fit_func()' via 'scipy.optimize.curve_fit().'
    # ~   ~   ~   ~   ~   ~   ~   ~   ~   ~   ~   ~   ~   ~   ~   ~   ~   ~   ~
#    print("length rot_data_table:", len( rot_data_table))

    if len( rot_data_table) > 2:

        #######################################################################
        # Fit the rotation curve data and extract the best fit parameters,
        #    their errors, and the chi-square (goodness of fit) statistic.
        #----------------------------------------------------------------------
        # average rotation curve data
        best_param, \
        param_err, \
        chi_square_rot = fit_data( depro_radii_fit,
                                  rot_vel_avg, rot_vel_avg_err,
                                  TRY_N)
        v_max_best = best_param[0]
        r_turn_best = best_param[1]
        alpha_best = best_param[2]
        v_max_sigma = param_err[0]
        r_turn_sigma = param_err[1]
        alpha_sigma = param_err[2]

        # positive rotation curve data
        pos_best_param, \
        pos_param_err, \
        pos_chi_square_rot = fit_data( depro_radii_fit,
                                      rot_vel_pos, rot_vel_pos_err,
                                      TRY_N)
        pos_v_max_best = pos_best_param[0]
        pos_r_turn_best = pos_best_param[1]
        pos_alpha_best = pos_best_param[2]
        pos_v_max_sigma = pos_param_err[0]
        pos_r_turn_sigma = pos_param_err[1]
        pos_alpha_sigma = pos_param_err[2]

        # negative rotation curve data
        neg_best_param, \
        neg_param_err, \
        neg_chi_square_rot = fit_data( depro_radii_fit,
                                      rot_vel_neg, rot_vel_neg_err,
                                      TRY_N)
        neg_v_max_best = neg_best_param[0]
        neg_r_turn_best = neg_best_param[1]
        neg_alpha_best = neg_best_param[2]
        neg_v_max_sigma = neg_param_err[0]
        neg_r_turn_sigma = neg_param_err[1]
        neg_alpha_sigma = neg_param_err[2]
        #######################################################################

        '''
        #######################################################################
        # Print statement to track the best fit parameters for the data file in
        #    question as well as the chi square (goodness of fit) statistic
        #----------------------------------------------------------------------
        print("Rot Curve Best Param:", best_param)
        print("Rot Curve Best Param (Positive):", pos_best_param)
        print("Rot Curve Best Param (Negative):", neg_best_param)
        print("Chi^{2}:", chi_square_rot)
        print("Chi^{2} (Pos):", pos_chi_square_rot)
        print("Chi^{2} (Neg):", neg_chi_square_rot)
        print("-----------------------------------------------------")
        #######################################################################
        '''
    # ~   ~   ~   ~   ~   ~   ~   ~   ~   ~   ~   ~   ~   ~   ~   ~   ~   ~   ~


    ###########################################################################
    # Create a dictionary to house each galaxy's data.
    #
    # NOTE: All entries are initialized to be -1s to indicate they are not
    #       found.
    #--------------------------------------------------------------------------
    row_data_dic = {'center_flux': center_flux,
                    'center_flux_error': center_flux_err,
                    'sMass_processed': sMass_processed,
                    'v_max_best': v_max_best,
                    'turnover_rad_best': r_turn_best,
                    'alpha_best': alpha_best,
                    'v_max_sigma': v_max_sigma,
                    'turnover_rad_sigma': r_turn_sigma,
                    'alpha_sigma': alpha_sigma,
                    'chi_square_rot': chi_square_rot,
                    'pos_v_max_best': pos_v_max_best,
                    'pos_turnover_rad_best': pos_r_turn_best,
                    'pos_alpha_best': pos_alpha_best,
                    'pos_v_max_sigma': pos_v_max_sigma,
                    'pos_turnover_rad_sigma': pos_r_turn_sigma,
                    'pos_alpha_sigma': pos_alpha_sigma,
                    'pos_chi_square_rot': pos_chi_square_rot,
                    'neg_v_max_best': neg_v_max_best,
                    'neg_turnover_rad_best': neg_r_turn_best,
                    'neg_alpha_best': neg_alpha_best,
                    'neg_v_max_sigma': neg_v_max_sigma,
                    'neg_turnover_rad_sigma': neg_r_turn_sigma,
                    'neg_alpha_sigma': neg_alpha_sigma,
                    'neg_chi_square_rot': neg_chi_square_rot}
    ###########################################################################

    return row_data_dic


def estimate_dark_matter( input_dict, rot_curve_file):
    """Estimate the total matter interior to a radius from the fitted v_max
    parameter and the last recorded radius for the galaxy. Then estimate the
    total dark matter interior to that radius by subtracting the stellar mass
    interior to that radius.

    @param:
        best_fit_param_table:
            astropy QTable containing the best fit parameters for each galaxy
            along with the errors associated with them and the chi-square
            goodness of fit statistic

        IMAGE_DIR:
            string representation of the file path that pictures are saved to

    @return:
        mass_estimate_table:
            astropy QTable with the stellar, dark matter, and total mass
            estimates interior to the outermost radius in the galaxy that was
            analyzed in this project
    """
    ###########################################################################
    # Gather the best fit 'v_max' parameter from the 'master_table'.
    #--------------------------------------------------------------------------
    v_max_best = input_dict['v_max_best'] * ( u.km / u.s)
    v_max_sigma = input_dict['v_max_sigma'] * ( u.km / u.s)
    ###########################################################################


    ###########################################################################
    # For each galaxy in the 'master_file,' calculate the total mass interior
    #    to the final radius recorded of the galaxy. Then, subtract the stellar
    #    mass interior to that radius to find the theoretical estimate for
    #    the dark matter interior to that radius.
    #
    # In addition, the errors associated with the total mass and dark matter
    #    mass interior to a radius are calculated.
    # ~   ~   ~   ~   ~   ~   ~   ~   ~   ~   ~   ~   ~   ~   ~   ~   ~   ~   ~
    rot_curve_table = ascii.read( rot_curve_file, format='ecsv')
    depro_dist = rot_curve_table['deprojected_distance'].value
    sMass_interior = rot_curve_table['sMass_interior'].value

    sMass_processed = sMass_interior[-1]


    if v_max_best == -1 or v_max_best == -100 or v_max_best == -999:
        gal_mass = np.nan
        gal_mass_err = np.nan
        theorized_dmMass = np.nan
        theorized_dmMass_err = np.nan

    else:
        depro_dist_end = np.abs( depro_dist[-1]) * ( u.kpc)
        depro_dist_end_m = depro_dist_end.to('m')
        v_max_best_m_per_s = v_max_best.to('m/s')
        v_max_sigma_m_per_s = v_max_sigma.to('m/s')

        gal_mass = v_max_best_m_per_s**2 * depro_dist_end_m / const.G
        gal_mass = gal_mass.to('M_sun')
        gal_mass /= u.M_sun  # strip 'gal_mass' of its units

        gal_mass_err = np.sqrt(
             ((2 * v_max_best_m_per_s * depro_dist_end_m) \
             / ( const.G * const.M_sun) )**2 \
             * v_max_sigma_m_per_s**2 \
          + ((-1 * v_max_best_m_per_s**2 * depro_dist_end_m) \
             / ( const.G**2 * const.M_sun) )**2 \
             * ( const.G.uncertainty * const.G.unit)**2 \
          + ((-1 * v_max_best_m_per_s**2 * depro_dist_end_m) \
             / ( const.G * const.M_sun**2) )**2 \
             * (const.M_sun.uncertainty * const.M_sun.unit)**2)

        theorized_dmMass = gal_mass - sMass_processed
        theorized_dmMass_err = gal_mass_err  # no error assumed in
                                             #   'sMass_processed'

    dmMass_to_sMass_ratio = theorized_dmMass / sMass_processed
    dmMass_to_sMass_ratio_err = theorized_dmMass_err / sMass_processed
    # ~   ~   ~   ~   ~   ~   ~   ~   ~   ~   ~   ~   ~   ~   ~   ~   ~   ~   ~


    ###########################################################################
    # Create a dictionary to house the mass estimate data.
    #--------------------------------------------------------------------------
    row_data_dict = {'total_mass': gal_mass.value,
               'total_mass_error': gal_mass_err.value,
               'dmMass': theorized_dmMass.value,
               'dmMass_error': theorized_dmMass_err.value,
               'sMass': sMass_processed,
               'dmMass_to_sMass_ratio': dmMass_to_sMass_ratio.value,
               'dmMass_to_sMass_ratio_error': dmMass_to_sMass_ratio_err.value}
    ###########################################################################

    return row_data_dict


def plot_fitted_rot_curve():
    # Make sure to take abs of deprojected distance in plotting
    depro_dist = np.abs()
    r_turn_best = input_dict['turnover_rad_best'] * ( u.kpc)
    alpha_best = input_dict['alpha_best']
    chi_square_rot = input_dict['chi_square_rot']
    rot_vel_data = rot_curve_table['rot_vel_avg'].value
    rot_vel_data_err = rot_curve_table['rot_vel_avg_error'].value

    if v_max_best != -1 and v_max_best != -100 and v_max_best != -999:
        ###################################################################
        # Plot the fitted rotation curve along with its errorbars. In addition,
        #    several statistics about the goodness of fit, and mass interior to
        #    the outermost radius recorded are displayed in the lower right
        #    side of the figure.
        #------------------------------------------------------------------
        fitted_rot_curve_fig = plt.figure(20)
        plt.errorbar( depro_dist, rot_vel_data,
                     yerr=rot_vel_data_err, fmt='o', color='purple',
                     markersize=4, capthick=1, capsize=3)

        plt.plot( np.linspace( 0, depro_dist[-1], 10000),
             rot_fit_func(np.linspace( 0, depro_dist[-1], 10000),
                          v_max_best.value,
                          r_turn_best.value,
                          alpha_best),
                          color='purple', linestyle='--')

        ax = fitted_rot_curve_fig.add_subplot(111)
        plt.tick_params( axis='both', direction='in')
        ax.yaxis.set_ticks_position('both')
        ax.xaxis.set_ticks_position('both')
        plt.ylabel(r'$V_{ROT}$ [$kms^{-1}$]')
        plt.xlabel(r'$d_{depro}$ [kpc]')
        plt.title( gal_id + ' Fitted Rotation Curve')

        textstr = '\n'.join((
                r'$\chi^{2}$: $%.3f$' % ( chi_square_rot, ),
                r'$m_{DM}$ [$M_{\odot}$]: $%9.2E$' % Decimal( theorized_dmMass.value, ),
                r'$m_{*}$ [$M_{\odot}$]: $%9.2E$' % Decimal( sMass_processed.value, ),
                r'$\frac{m_{*}}{m_{DM}}$: $%.3f$' % ( dmMass_to_sMass_ratio.value, )))
        props = dict( boxstyle='round', facecolor='cornsilk', alpha=0.6)

        ax.text(0.65, 0.34, textstr,
                verticalalignment='top', horizontalalignment='left',
                transform=ax.transAxes,
                color='black', fontsize=10, bbox=props)

        plt.savefig( IMAGE_DIR + '/fitted_rotation_curves/' + gal_id +\
                    '_fitted_rotation_curve.' + IMAGE_FORMAT,
                    format=IMAGE_FORMAT)
        plt.show()
        plt.close()
        ###################################################################