#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Created on Tues Jan 15 2019
@author: Jacob A. Smith
@version: 1.0

Extracts rotation curve data points from files written with rotation_curve_vX_X
and fits a function to the data given several parameters. A total mass for the
galaxy is then extracted from the v_max parameter and the stellar mass is then
subtracted to find the galaxy's dark matter mass.
"""
###############################
# Optional import statements  #
#    for diagnostics.         #
###############################
import matplotlib.pyplot as plt
import numpy as np
from decimal import Decimal

import glob, os.path

from scipy.optimize import curve_fit
from scipy.stats import norm
from scipy.stats import ks_2samp

from astropy.table import QTable, Column
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
            curve equation (given in m/s)

        r_turn:
            the radius at which the rotation curve trasitions from increasing
            to flat-body parameter for the rotation curve equation (given in
            kpc)

        alpha:
            the exponential parameter for the rotation curve equation
            (unitless)

    @return: the rotation curve equation with the given '@param' parameters and
             'depro_radius' data

    """
    return v_max * \
           (depro_radius / (r_turn**alpha + depro_radius**alpha)**(1/alpha))


def extract_matched_data( MASTER_FILE_NAME, CROSS_REF_FILE_NAMES):
    """Extract the vflag parameter {0, 1, 2, -9} that tells
    if the galaxy in question does not lie in a void {0}, does lie in a void
    {1}, is cut off by the MaNGA footprint {2}, or not found in the MaNGA
    footprint {-9}. Also extract the 'Mstar_NSA' parameter that gives
    information about a galaxy's stellar mass.

    Originally, the files in this project were cross referenced to Prof. Kelly
    Douglass' file via the link below. However, upon improvements to
    void_finder, new classifications wer given and the code has been updated.

    CROSS_REF_FILE_NAMES[0] .txt file obtained from:
    <http://www.pas.rochester.edu/~kdouglass/Research/
    kias1033_5_P-MJD-F_MPAJHU_Zdust_stellarMass_BPT_SFR_NSA_correctVflag.txt>

    @param:
        MASTER_FILE_NAME:
            string representation of the master file containing the best fit
            parameters along with their errors, the mass estimates obtained in
            estimate_dark_matter, and identifying information about each
            galaxy

        CROSS_REF_FILE_NAMES:
            string representations of the .txt files uses to extract the vflag
            parameter as obtained from void_finder.
            Prof. Kelly Douglass' file {index 0} was initially used to cross
            reference for vflag. However, with the updates to void_finder,
            indicies 1, 2, 3, and 4 are used.

    @return:
        vflag_master:
            master list containing each galaxy's 'vflag' parameter

        mstar_NSA_master:
            master list containing each galaxy's 'Mstar_NSA' parameter
    """
    master_table = ascii.read( MASTER_FILE_NAME,
                                   format='ecsv',
                                   include_names = ('NSA_plate',
                                                    'NSA_fiberID',
                                                    'NSA_MJD'))


    # ONEOINEQOH
    mstar_NSA = ascii.read( LOCAL_PATH + '\\nsa_v0_1_2.fits')


    # NOT FINISHED DO NOT USE
    doug_nc = ascii.read( CROSS_REF_FILE_NAMES[1],
                           include_names = ('MaNGA_plate',
                                            'MaNGA_fiberID',
                                            'MaNGA_data_release',
                                            'vflag'))
    doug_nf = ascii.read( CROSS_REF_FILE_NAMES[2],
                           include_names = ('MaNGA_plate',
                                            'MaNGA_fiberID',
                                            'MaNGA_data_release',
                                            'vflag'))
    doug_void_reclass = ascii.read( CROSS_REF_FILE_NAMES[3],
                           include_names = ('MaNGA_plate',
                                            'MaNGA_fiberID',
                                            'MaNGA_data_release',
                                            'vflag'))
    doug_wall_reclass = ascii.read( CROSS_REF_FILE_NAMES[4],
                           include_names = ('MaNGA_plate',
                                            'MaNGA_fiberID',
                                            'MaNGA_data_release',
                                            'vflag'))

    vflag_master = np.full( len( master_table), -1)
    mstar_NSA_master = np.full( len( master_table), -1, dtype=np.int64)

    for i in range( len( master_table)):
        target_plate = master_table['NSA_plate'][i]
        target_mjd = master_table['NSA_MJD'][i]
        target_fiberID = master_table['NSA_fiberID'][i]

        #######################################################################
        # The code below first matches via the 'target_plate' value. Of those
        #    matches, the next line matches via 'target_MJD.' Finally, of those
        #    galaxies in 'vflag_ref_table' that match both the galaxy in
        #    question's plate and MJD, the remaining line matches via
        #    'target_fiberID.'
        #
        # NOTE: At the conclusion of these three matching statements, only
        #       one galaxy match should be found within 'ref_table.'
        #######################################################################
        # MATCH PLATE
        plate_matches = ref_table['plate'] == target_plate
        # MATCH MJD
        mjd_matches = ref_table['MJD'] == target_mjd
        # MATCH FIBERID
        fiberID_matches = ref_table['fiberID'] == target_fiberID

        galaxy_matches = np.logical_and.reduce(( plate_matches, mjd_matches,
                                                fiberID_matches))

        #######################################################################
        # This statement pulls out the 'vflag' parameter from Prof. Kelly
        #    Douglass' file.
        #
        # NOTE: There are four "[0]" trailing the matched galaxy in
        #       'vflag_ref_table.' It is unclear why this must be done, but
        #       this code should return the 'vflag' parameter. It is
        #       theorized that when matching, additional tables are hashed
        #       together; thus to counteract this, four trailing "[0]" must
        #       be added to access 'vflag.'
        #######################################################################
        if len( ref_table[ galaxy_matches]) == 1:
            vflag_master[i] = ref_table[galaxy_matches]['vflag'][0]

        if np.isfinite( ref_table['Mstar_NSA'][galaxy_matches]):
            mstar_NSA_master[i] = ref_table[galaxy_matches]['Mstar_NSA'][0]

#        print( ref_table[galaxy_matches]['Mstar_NSA'][0])
#        print( ref_table[galaxy_matches]['vflag'][0])

    return vflag_master, mstar_NSA_master


def write_matched_data( vflag_list, mstar_NSA_list,
                       MASTER_FILE_NAME):
    """Writes the vflag parameter to the master file for each corresponding
    galaxy.

    @param:
        vflag_list:
            list containing all of the 'vflag' parameters for each galaxy

        mstar_NSA_list:
            list containing all the 'mstar_NSA' parameters for each galaxy

        LOCAL_PATH:
            string representation of the directory of the main script file

        MASTER_FILE_NAME:
            string representation of the name of the file which 'vflag_list'
            will be written to
    """
    master_table = ascii.read( MASTER_FILE_NAME, format='ecsv')

    ###########################################################################
    # NOTE: Because the add_column method will not overwrite existing columns
    #       with the same name, the columns are first removed from master_table
    #       if they exist and then added back with the updated values.
    #--------------------------------------------------------------------------
    del_col_names = ['vflag', 'Mstar_NSA']

    for col in del_col_names:
        try:
            master_table.remove_column( col)
        except KeyError:
            print("USER GENERATED KeyError: \n" + \
                  "Column name not found in del_col_names")
    ###########################################################################

    master_table.add_column( Column( vflag_list), name='vflag')
    master_table.add_column( Column( mstar_NSA_list) * u.M_sun,
                            name='Mstar_NSA')

    ascii.write(master_table,
                MASTER_FILE_NAME,
                format = 'ecsv',
                overwrite = True)


def fit_rot_curve_files( rot_curve_files, gal_stat_files,
                   TRY_N, ROT_CURVE_MASTER_FOLDER, IMAGE_DIR):
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

        ROT_CURVE_MASTER_FOLDER:
            string representation of the path of the folder containing all of
            the galaxy data for all of the galaxies in the MaNGA survey

        IMAGE_DIR:
            string representation of the file path that pictures are saved to

    @return:
        best_fit_param_table:
            astropy QTable containing the best fit parameters for each galaxy
            along with the errors associated with them and the chi-square
            goodness of fit statistic
    """
    ###########################################################################
    # Master arrays initialized to be empty.
    #--------------------------------------------------------------------------
    v_max_best_master = []
    r_turn_best_master = []
    alpha_best_master = []

    v_max_sigma_master = []
    r_turn_sigma_master = []
    alpha_sigma_master = []
    chi_square_rot_master = []


    pos_v_max_best_master = []
    pos_r_turn_best_master = []
    pos_alpha_best_master = []

    pos_v_max_sigma_master = []
    pos_r_turn_sigma_master = []
    pos_alpha_sigma_master = []
    pos_chi_square_rot_master = []


    neg_v_max_best_master = []
    neg_r_turn_best_master = []
    neg_alpha_best_master = []

    neg_v_max_sigma_master = []
    neg_r_turn_sigma_master = []
    neg_alpha_sigma_master = []
    neg_chi_square_rot_master = []
    ###########################################################################


    ###########################################################################
    # For each rotation curve data file, fit the data to the funciton given in
    #    rot_fit_func. More information is given within the loop.
    # ~  ~  ~  ~  ~  ~  ~  ~  ~  ~  ~  ~  ~  ~  ~  ~  ~  ~  ~  ~  ~  ~  ~  ~  ~
    for rot_file, gal_stat_file in zip( rot_curve_files, gal_stat_files):
        #######################################################################
        # Extract the gal_ID from the gal_stat_file,
        #----------------------------------------------------------------------
        gal_stat_table = ascii.read( gal_stat_file, format='ecsv')
        gal_ID = gal_stat_table['gal_ID'][0]
        print("gal_ID:", gal_ID)
        #######################################################################


        #######################################################################
        # Best fit parameters and errors in those parameters are initialized to
        # -1 to indicate that they start out as 'not found.' A value of -1 in
        # the master file therefore indicates that the galaxy was not fitted or
        # that there was otherwise insufficent data to find the best fit
        # parameters and their errors.
        #----------------------------------------------------------------------
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
        #######################################################################


        #######################################################################
        # Import the necessary data from the rotation curve files.
        #----------------------------------------------------------------------
        rot_data_table = ascii.read(rot_file, format='ecsv')
        depro_radii = rot_data_table['deprojected_distance'].value
        rot_vel_avg = rot_data_table['rot_vel_avg'].value
        rot_vel_avg_err = rot_data_table['rot_vel_avg_error'].value
        rot_vel_max = rot_data_table['max_velocity'].value
        rot_vel_max_err = rot_data_table['max_velocity_error'].value
        rot_vel_min = np.abs( rot_data_table['min_velocity'].value)
        rot_vel_min_err = rot_data_table['min_velocity_error'].value

        print("depro_radii:", depro_radii)
        print("rot_vel_avg:", rot_vel_avg)
        print("rot_vel_avg_err:", rot_vel_avg_err)
        print("rot_vel_max:", rot_vel_max)
        print("rot_vel_max_err:", rot_vel_max_err)
        print("rot_vel_min:", rot_vel_min)
        print("rot_vel_min_err:", rot_vel_min_err)
        #######################################################################


        #######################################################################
        # General information about the data file in question as well as a plot
        #    of the data before fitting.
        #----------------------------------------------------------------------
#        print( rot_file, ":\n\n", rot_data_table, "\n\n")
#        print("DATA TABLE INFORMATION \n",
#              'Columns:', rot_data_table.columns, '\n',
#              'Column Names:', rot_data_table.colnames, '\n',
#              'Meta Data:', rot_data_table.meta, '\n',
#              'Number of Rows:', len( rot_data_table))
#
#        rot_curve_decomposed_fig = plt.figure(10)
#        plt.errorbar( depro_radii, rot_vel_max,
#                     yerr=rot_vel_max_err, 'ro', ecolor='red')
#        plt.errorbar( depro_radii, rot_vel_min,
#                     yerr=rot_vel_min_err, 'bo', ecolor='blue')
#
#        ax = rot_curve_decomposed_fig.add_subplot(111)
#        plt.tick_params( axis='both', direction='in')
#        ax.yaxis.set_ticks_position('both')
#        ax.xaxis.set_ticks_position('both')
#
#        plt.ylabel(r'$V_{ROT}$ [$kms^{-1}$]')
#        plt.xlabel(r'$d_{depro}$ [kpc]')
#        plt.title( gal_ID + "Decomposed Rotation Curves")
#        plt.show()
        #######################################################################


        #######################################################################
        # Create a set of conditions to test for valid data:
        #
        # I.)  The data file should contain at least three data points so as to
        #      fit a rotation curve.
        # II.) The absolute maximum and absolute minimum of each data file
        #      should not be 0.
        #######################################################################
        if len( rot_data_table) >= 3:
            ###################################################################
            # NOTES:
            #------------------------------------------------------------------
            # Following from 'rot_fit_func,' the following first guesses of the
            #    parameters 'v_max,' 'r_turn,' and 'alpha' are described as
            #    such:
            #
            # v_max_guess / v_min_guess:
            #    the absolute maximum and absolute minimum (respectively) of
            #    the data file in question; first guess of the 'v_max'
            #    parameter
            #
            # r_turn_max_guess / r_turn_min_guess:
            #    the radius atwhich 'v_max' and 'v_min' are respectively found;
            #    first guess for 'r_turn' parameter
            #
            # alpha_guess: imperically-estimated, first guess of the 'alpha'
            #    parameter
            ###################################################################
            v_max_guess = max( rot_vel_avg)
            pos_v_max_guess = max( rot_vel_max)
            neg_v_max_guess = max( rot_vel_min)

            ###################################################################
            # If the initial guesses for the maximum rotational velocity are
            #    not 0, continue with the fitting process.
            #------------------------------------------------------------------
            if v_max_guess > 0 and pos_v_max_guess > 0 and neg_v_max_guess > 0:
                r_turn_guess = depro_radii[
                        np.argwhere( rot_vel_avg == v_max_guess)][0][0]
                pos_r_turn_guess = depro_radii[
                        np.argwhere( rot_vel_max == pos_v_max_guess)][0][0]
                neg_r_turn_guess = depro_radii[
                        np.argwhere( rot_vel_min == neg_v_max_guess)][0][0]

                alpha_guess = 2
                pos_alpha_guess = 2
                neg_alpha_guess = 2

                rot_guess = [ v_max_guess, r_turn_guess, alpha_guess]
                pos_rot_guess = [ pos_v_max_guess, pos_r_turn_guess,
                                 pos_alpha_guess]
                neg_rot_guess = [ neg_v_max_guess, neg_r_turn_guess,
                                 neg_alpha_guess]
                ###############################################################
                # Print statement to track the first guesses for the
                #    'rot_fit_func', and 'lum_fit_func' parameters.
                #--------------------------------------------------------------
#                print("Rot Parameter Guess:", rot_guess)
                ###############################################################


                ###############################################################
                # NOTES:
                #--------------------------------------------------------------
                # Each respective popt holds an array of the best fit
                #    parameters for the data file in question.
                #
                # Each respective pcov holds the covarience matrix of the
                #    parameters for the data file in question.
                #
                # bounds: imperically-devised limits for the 'rot_fit_func'
                #
                # max_nfev: the number of times the algorithm will try to fit
                #    the data
                #
                # loss='cauchy': "Severely weakens outliers influence, but may
                #    cause difficulties in optimization process"
                #    (see link below).
                #
                # <docs.scipy.org/doc/scipy/reference/generated/
                # scipy.optimize.least_squares.html
                # #scipy.optimize.least_squares>
                #
                #
                # If the program experiences a RuntimeError, it is assumed
                #    that scipy.optimize.curve_fit could not find the best
                #    fit parameters. The data given in the rotation curve
                #    file was otherwise satisfactory. Values of -999 are
                #    reported in instances where this RuntimeError occurs.
                ###############################################################

                ###############################################################
                # Average rotation curve
                #--------------------------------------------------------------
                try:
                    rot_popt, rot_pcov = curve_fit( rot_fit_func,
                            depro_radii, rot_vel_avg,
                            p0 = rot_guess,
                            sigma = rot_vel_avg_err,
                            bounds=( ( v_max_guess / 2,
                                      r_turn_guess / 4,
                                      0),
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
                      zip( depro_radii, rot_vel_avg, rot_vel_avg_err):
                          observed = velocity
                          expected = rot_fit_func(
                                  radius,
                                  v_max_best,
                                  r_turn_best,
                                  alpha_best)
                          error = vel_err
                          chi_square_rot += (observed - expected)**2 / error**2

                except RuntimeError:
                    v_max_best = -999
                    r_turn_best = -999
                    alpha_best = -999

                    v_max_sigma = -999
                    r_turn_sigma = -999
                    alpha_sigma = -999
                    chi_square_rot = -999
                ###############################################################

                ###############################################################
                # Positve rotation curve
                #--------------------------------------------------------------
                try:
                    pos_rot_popt, pos_rot_pcov = curve_fit( rot_fit_func,
                            depro_radii, rot_vel_max,
                            p0 = pos_rot_guess,
                            sigma = rot_vel_max_err,
                            bounds=( ( pos_v_max_guess / 2,
                                      pos_r_turn_guess / 4,
                                      0),
                                     ( pos_v_max_guess * 1.5,
                                      pos_r_turn_guess * 2,
                                      np.inf)),
                            max_nfev=TRY_N, loss='cauchy')

                    pos_rot_perr = np.sqrt( np.diag( pos_rot_pcov))

                    pos_v_max_best = pos_rot_popt[0]
                    pos_r_turn_best = pos_rot_popt[1]
                    pos_alpha_best = pos_rot_popt[2]

                    pos_v_max_sigma = pos_rot_perr[0]
                    pos_r_turn_sigma = pos_rot_perr[1]
                    pos_alpha_sigma = pos_rot_perr[2]


                    pos_chi_square_rot = 0
                    for radius, velocity, vel_err in \
                      zip( depro_radii, rot_vel_max, rot_vel_max_err):
                          observed = velocity
                          expected = rot_fit_func(
                                  radius,
                                  pos_v_max_best,
                                  pos_r_turn_best,
                                  pos_alpha_best)
                          error = vel_err
                          pos_chi_square_rot += (observed - expected)**2 \
                                                / error**2

                except RuntimeError:
                    pos_v_max_best = -999
                    pos_r_turn_best = -999
                    pos_alpha_best = -999

                    pos_v_max_sigma = -999
                    pos_r_turn_sigma = -999
                    pos_alpha_sigma = -999
                    pos_chi_square_rot = -999
                ###############################################################

                ###############################################################
                # Negative rotation curve
                #--------------------------------------------------------------
                try:
                    neg_rot_popt, neg_rot_pcov = curve_fit( rot_fit_func,
                            depro_radii, rot_vel_min,
                            p0 = neg_rot_guess,
                            sigma = rot_vel_min_err,
                            bounds=( ( neg_v_max_guess / 2,
                                      neg_r_turn_guess / 4,
                                      0),
                                     ( neg_v_max_guess * 1.5,
                                      neg_r_turn_guess * 2,
                                      np.inf)),
                            max_nfev=TRY_N, loss='cauchy')

                    neg_rot_perr = np.sqrt( np.diag( neg_rot_pcov))

                    neg_v_max_best = neg_rot_popt[0]
                    neg_r_turn_best = neg_rot_popt[1]
                    neg_alpha_best = neg_rot_popt[2]

                    neg_v_max_sigma = neg_rot_perr[0]
                    neg_r_turn_sigma = neg_rot_perr[1]
                    neg_alpha_sigma = neg_rot_perr[2]


                    neg_chi_square_rot = 0
                    for radius, velocity, vel_err in \
                      zip( depro_radii, rot_vel_min, rot_vel_min_err):
                          observed = velocity
                          expected = rot_fit_func(
                                  radius,
                                  neg_v_max_best,
                                  neg_r_turn_best,
                                  neg_alpha_best)
                          error = vel_err
                          neg_chi_square_rot += (observed - expected)**2 \
                                                / error**2

                except RuntimeError:
                    neg_v_max_best = -999
                    neg_r_turn_best = -999
                    neg_alpha_best = -999

                    neg_v_max_sigma = -999
                    neg_r_turn_sigma = -999
                    neg_alpha_sigma = -999
                    neg_chi_square_rot = -999
                ###############################################################


                ###############################################################
                # If a chi_square value is calculated to be infinity, set the
                #    chi_square to -50 so that it does not trigger issues in
                #    plotting a histogram of the values.
                ###############################################################
                if chi_square_rot == float('inf'):
                    chi_square_rot = -50
                if pos_chi_square_rot == float('inf'):
                    pos_chi_square_rot = -50
                if neg_chi_square_rot == float('inf'):
                    neg_chi_square_rot = -50
                ###############################################################


                ###############################################################
                # Print statement to track the best fit parameters for the data
                #    file in question as well as the chi square (goodness of
                #    fit) statistic.
                #--------------------------------------------------------------
                print("Rot Curve Best Param:", rot_popt)
                print("Rot Curve Best Param (Positive):", pos_rot_popt)
                print("Rot Curve Best Param (Negative):", neg_rot_popt)
                print(r"$\chi^{2}$:", chi_square_rot)
                print(r"$\chi^{2}$ (Positive):", pos_chi_square_rot)
                print(r"$\chi^{2}$ (Negative):", neg_chi_square_rot)
                print("-----------------------------------------------------")
                ###############################################################


            ###################################################################
            # If either the absolute maximum or the absolute minimum of the
            #    data file in question is zero, the best fit parameters and
            #    their errors are set to -100.
            #------------------------------------------------------------------
            elif v_max_guess <= 0 or pos_v_max_guess <= 0 or neg_v_max_guess <= 0:
                v_max_best = -100
                r_turn_best = -100
                alpha_best = -100
                pos_v_max_best = -100
                pos_r_turn_best = -100
                pos_alpha_best = -100
                neg_v_max_best = -100
                neg_r_turn_best = -100
                neg_alpha_best = -100

                v_max_sigma = -100
                r_turn_sigma = -100
                alpha_sigma = -100
                chi_square_rot = -100
                pos_v_max_sigma = -100
                pos_r_turn_sigma = -100
                pos_alpha_sigma = -100
                pos_chi_square_rot = -100
                neg_v_max_sigma = -100
                neg_r_turn_sigma = -100
                neg_alpha_sigma = -100
                neg_chi_square_rot = -100
            ###################################################################

        #######################################################################
        # If the data file in question has less than three data points, the
        #    best fit parameters along with their errors are left as
        #    initialized (-1).
        #----------------------------------------------------------------------
        elif len( rot_data_table) < 3:
            pass
        #######################################################################


        #######################################################################
        # Append the corresponding values to their respective arrays to
        #    write to the master_file.
        #----------------------------------------------------------------------
        v_max_best_master.append( v_max_best)
        r_turn_best_master.append( r_turn_best)
        alpha_best_master.append( alpha_best)
        pos_v_max_best_master.append( pos_v_max_best)
        pos_r_turn_best_master.append( pos_r_turn_best)
        pos_alpha_best_master.append( pos_alpha_best)
        neg_v_max_best_master.append( neg_v_max_best)
        neg_r_turn_best_master.append( neg_r_turn_best)
        neg_alpha_best_master.append( neg_alpha_best)

        v_max_sigma_master.append( v_max_sigma)
        r_turn_sigma_master.append( r_turn_sigma)
        alpha_sigma_master.append( alpha_sigma)
        chi_square_rot_master.append( chi_square_rot)
        pos_v_max_sigma_master.append( pos_v_max_sigma)
        pos_r_turn_sigma_master.append( pos_r_turn_sigma)
        pos_alpha_sigma_master.append( pos_alpha_sigma)
        pos_chi_square_rot_master.append( pos_chi_square_rot)
        neg_v_max_sigma_master.append( neg_v_max_sigma)
        neg_r_turn_sigma_master.append( neg_r_turn_sigma)
        neg_alpha_sigma_master.append( neg_alpha_sigma)
        neg_chi_square_rot_master.append( neg_chi_square_rot)
        #######################################################################
    # ~  ~  ~  ~  ~  ~  ~  ~  ~  ~  ~  ~  ~  ~  ~  ~  ~  ~  ~  ~  ~  ~  ~  ~  ~


    ###########################################################################
    # Convert the data arrays into Column objects to add to the master_file
    #    data table.
    #--------------------------------------------------------------------------
    v_max_best_col = Column( v_max_best_master)
    r_turn_best_col = Column( r_turn_best_master)
    alpha_best_col = Column( alpha_best_master)
    pos_v_max_best_col = Column( pos_v_max_best_master)
    pos_r_turn_best_col = Column( pos_r_turn_best_master)
    pos_alpha_best_col = Column( pos_alpha_best_master)
    neg_v_max_best_col = Column( neg_v_max_best_master)
    neg_r_turn_best_col = Column( neg_r_turn_best_master)
    neg_alpha_best_col = Column( neg_alpha_best_master)

    v_max_sigma_col = Column( v_max_sigma_master)
    r_turn_sigma_col = Column( r_turn_sigma_master)
    alpha_sigma_col = Column( alpha_sigma_master)
    chi_square_rot_col = Column( chi_square_rot_master)
    pos_v_max_sigma_col = Column( pos_v_max_sigma_master)
    pos_r_turn_sigma_col = Column( pos_r_turn_sigma_master)
    pos_alpha_sigma_col = Column( pos_alpha_sigma_master)
    pos_chi_square_rot_col = Column( pos_chi_square_rot_master)
    neg_v_max_sigma_col = Column( neg_v_max_sigma_master)
    neg_r_turn_sigma_col = Column( neg_r_turn_sigma_master)
    neg_alpha_sigma_col = Column( neg_alpha_sigma_master)
    neg_chi_square_rot_col = Column( neg_chi_square_rot_master)
    ###########################################################################


    ###########################################################################
    # Add the column objects to astropy QTables.
    #--------------------------------------------------------------------------
    best_fit_param_table = QTable( [v_max_best_col * ( u.km / u.s),
                                    v_max_sigma_col * ( u.km / u.s),
                                    r_turn_best_col * ( u.kpc),
                                    r_turn_sigma_col * ( u.kpc),
                                    alpha_best_col,
                                    alpha_sigma_col,
                                    chi_square_rot_col,
                                    pos_v_max_best_col * ( u.km / u.s),
                                    pos_v_max_sigma_col * ( u.km / u.s),
                                    pos_r_turn_best_col * ( u.kpc),
                                    pos_r_turn_sigma_col * ( u.kpc),
                                    pos_alpha_best_col,
                                    pos_alpha_sigma_col,
                                    pos_chi_square_rot_col,
                                    neg_v_max_best_col * ( u.km / u.s),
                                    neg_v_max_sigma_col * ( u.km / u.s),
                                    neg_r_turn_best_col * ( u.kpc),
                                    neg_r_turn_sigma_col * ( u.kpc),
                                    neg_alpha_best_col,
                                    neg_alpha_sigma_col,
                                    neg_chi_square_rot_col],
                           names = ['v_max_best',
                                    'v_max_sigma',
                                    'turnover_rad_best',
                                    'turnover_rad_sigma',
                                    'alpha_best',
                                    'alpha_sigma',
                                    'chi_square_rot',
                                    'pos_v_max_best',
                                    'pos_v_max_sigma',
                                    'pos_turnover_rad_best',
                                    'pos_turnover_rad_sigma',
                                    'pos_alpha_best',
                                    'pos_alpha_sigma',
                                    'pos_chi_square_rot',
                                    'neg_v_max_best',
                                    'neg_v_max_sigma',
                                    'neg_turnover_rad_best',
                                    'neg_turnover_rad_sigma',
                                    'neg_alpha_best',
                                    'neg_alpha_sigma',
                                    'neg_chi_square_rot'])
    ###########################################################################

    return best_fit_param_table


def write_best_params( best_fit_param_table, gal_stat_files,
                      MASTER_FILE_NAME, ROT_CURVE_MASTER_FOLDER):
    """Writes the list of galaxies along with the respective best fit
    parameters and galaxy statistics to a text file indicated in the
    'MASTER_FILE_NAME' variable.

    @param:
        gal_stat_files:
            string list of galaxies for which general statistical data exists

        best_fit_param_table:
            astropy QTable that contains:
                i.) MaNGA plate and fiberID for identification
               ii.) the best fit parameters for the max and min rotation
                       curves
              iii.) the best fit patameters for the max and min luminosity
                       curves

        LOCAL_PATH:
            string representation of the directory path of this file

        MASTER_FILE_NAME:
            string representation of the name of the file which
            'best_fit_param_table' will be written to

        ROT_CURVE_MASTER_FOLDER:
            string representation of the path of the folder containing all of
            the galaxy data for all of the galaxies in the MaNGA survey

    @return: function returns the boolean True upon completion
    """
    master_table = ascii.read( MASTER_FILE_NAME, format='ecsv')

    ###########################################################################
    # NOTE: Because the add_column method will not overwrite existing columns
    #       with the same name, the columns are first removed from master_table
    #       if they exist and then added back with the updated values.
    #--------------------------------------------------------------------------
    del_col_names = ['v_max_best',
                     'v_max_sigma',
                     'turnover_rad_best',
                     'turnover_rad_sigma',
                     'alpha_best',
                     'alpha_sigma',
                     'chi_square_rot',
                     'pos_v_max_best',
                     'pos_v_max_sigma',
                     'pos_turnover_rad_best',
                     'pos_turnover_rad_sigma',
                     'pos_alpha_best',
                     'pos_alpha_sigma',
                     'pos_chi_square_rot',
                     'neg_v_max_best',
                     'neg_v_max_sigma',
                     'neg_turnover_rad_best',
                     'neg_turnover_rad_sigma',
                     'neg_alpha_best',
                     'neg_alpha_sigma',
                     'neg_chi_square_rot',
                     'center_luminosity',
                     'center_luminosity_err',
                     'sMass_processed']

    for col in del_col_names:
        try:
            master_table.remove_column( col)
        except KeyError:
            print("USER GENERATED KeyError: \n" + \
                  "Column name" + col + "not found in del_col_names")
    ###########################################################################

    lum_center_master = []
    lum_center_err_master = []
    sMass_processed_master = []

    for data_file in gal_stat_files:
        gal_stat_table = ascii.read(data_file, format='ecsv')

        lum_center = gal_stat_table['center_luminosity'].value[0]
        lum_center_err = gal_stat_table['center_luminosity_error'].value[0]
        sMass_processed = gal_stat_table['sMass_processed'].value[0]

        lum_center_master.append( lum_center)
        lum_center_err_master.append( lum_center_err)
        sMass_processed_master.append( sMass_processed)

    master_table.add_column( best_fit_param_table['v_max_best'],
                            name='v_max_best')
    master_table.add_column( best_fit_param_table['v_max_sigma'],
                            name='v_max_sigma')
    master_table.add_column( best_fit_param_table['turnover_rad_best'],
                            name='turnover_rad_best')
    master_table.add_column( best_fit_param_table['turnover_rad_sigma'],
                            name='turnover_rad_sigma')
    master_table.add_column( best_fit_param_table['alpha_best'],
                            name='alpha_best')
    master_table.add_column( best_fit_param_table['alpha_sigma'],
                            name='alpha_sigma')
    master_table.add_column( best_fit_param_table['chi_square_rot'],
                            name='chi_square_rot')
    master_table.add_column( best_fit_param_table['pos_v_max_best'],
                            name='pos_v_max_best')
    master_table.add_column( best_fit_param_table['pos_v_max_sigma'],
                            name='pos_v_max_sigma')
    master_table.add_column( best_fit_param_table['pos_turnover_rad_best'],
                            name='pos_turnover_rad_best')
    master_table.add_column( best_fit_param_table['pos_turnover_rad_sigma'],
                            name='pos_turnover_rad_sigma')
    master_table.add_column( best_fit_param_table['pos_alpha_best'],
                            name='pos_alpha_best')
    master_table.add_column( best_fit_param_table['pos_alpha_sigma'],
                            name='pos_alpha_sigma')
    master_table.add_column( best_fit_param_table['pos_chi_square_rot'],
                            name='pos_chi_square_rot')
    master_table.add_column( best_fit_param_table['neg_v_max_best'],
                            name='neg_v_max_best')
    master_table.add_column( best_fit_param_table['neg_v_max_sigma'],
                            name='neg_v_max_sigma')
    master_table.add_column( best_fit_param_table['neg_turnover_rad_best'],
                            name='neg_turnover_rad_best')
    master_table.add_column( best_fit_param_table['neg_turnover_rad_sigma'],
                            name='neg_turnover_rad_sigma')
    master_table.add_column( best_fit_param_table['neg_alpha_best'],
                            name='neg_alpha_best')
    master_table.add_column( best_fit_param_table['neg_alpha_sigma'],
                            name='neg_alpha_sigma')
    master_table.add_column( best_fit_param_table['neg_chi_square_rot'],
                            name='neg_chi_square_rot')
    master_table.add_column( Column( lum_center_master),
                            name='center_luminosity')
    master_table.add_column( Column( lum_center_err_master),
                            name='center_luminosity_error')
    master_table.add_column( Column( sMass_processed_master),
                            name='sMass_processed')

    ascii.write(master_table,
                MASTER_FILE_NAME,
                format = 'ecsv',
                overwrite = True)


def estimate_dark_matter( best_fit_param_table,
                         rot_curve_files, gal_stat_files,
                         IMAGE_DIR):
    """Estimate the total matter interior to a radius from the fitted v_max
    parameter and the last recorded radius for the galaxy. Then estimate the
    total dark matter interior to that radius by subtracting the stellar mass
    interior to that radius.

    @param:
        best_fit_param_table:
            astropy QTable containing the best fit parameters for each galaxy
            along with the errors associated with them and the chi-square
            goodness of fit statistic

        rot_curve_files:
            a string list containing all of the rotation curve files within the
            'rotation_curve_vX_X' output folder

        gal_stat_files:
            a string list containing all of the galaxy statistic files within
            the 'rotation_curve_vX_X' output folder

        IMAGE_DIR:
            string representation of the file path that pictures are saved to

    @return:
        mass_estimate_table:
            astropy QTable with the stellar, dark matter, and total mass
            estimates interior to the outermost radius in the galaxy that was
            analyzed in this project
    """
    ###########################################################################
    # Initialize the arrays that will store the data.
    #--------------------------------------------------------------------------
    gal_mass_master = []
    gal_mass_err_master = []
    theorized_dmMass_master = []
    theorized_dmMass_err_master = []
    sMass_processed_master = []
    dm_to_stellar_mass_ratio_master = []
    dm_to_stellar_mass_ratio_err_master = []
    ###########################################################################


    ###########################################################################
    # Gather the necessary information from the 'best_fit_param_table'.
    #--------------------------------------------------------------------------
    v_max_best_list = best_fit_param_table['v_max_best']
    v_max_sigma_list = best_fit_param_table['v_max_sigma']
    r_turn_best_list = best_fit_param_table['turnover_rad_best']
    alpha_best_list = best_fit_param_table['alpha_best']
    chi_square_rot_list = best_fit_param_table['chi_square_rot']
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
    for v_max_best, v_max_sigma, r_turn_best, alpha_best, chi_square_rot, \
      rot_data_file, gal_stat_file in zip(
              v_max_best_list, v_max_sigma_list,
              r_turn_best_list, alpha_best_list,
              chi_square_rot_list,
              rot_curve_files, gal_stat_files):

        gal_stat_table = ascii.read( gal_stat_file, format='ecsv')
        gal_id = gal_stat_table['gal_ID'][0]

        rot_curve_table = ascii.read( rot_data_file, format='ecsv')
        depro_dist = rot_curve_table['deprojected_distance'].value
        rot_vel_data = rot_curve_table['rot_vel_avg'].value
        rot_vel_data_err = rot_curve_table['rot_vel_avg_error'].value
        sMass_interior = rot_curve_table['sMass_interior']

        sMass_processed = sMass_interior[-1]

        if v_max_best == -1 or v_max_best == -100 or v_max_best == -999:
            gal_mass = np.nan * u.M_sun
            gal_mass_err = np.nan * u.M_sun
            theorized_dmMass = np.nan * u.M_sun
            theorized_dmMass_err = np.nan * u.M_sun

        else:
            depro_dist_end = depro_dist[-1] * ( u.kpc)
            depro_dist_end_m = depro_dist_end.to('m')
            v_max_best_m_per_s = v_max_best.to('m/s')
            v_max_sigma_m_per_s = v_max_sigma.to('m/s')

            gal_mass = v_max_best_m_per_s**2 * depro_dist_end_m / const.G
            gal_mass = gal_mass.to('M_sun')
            gal_mass_err = np.sqrt(
                 ((2 * v_max_best_m_per_s * depro_dist_end_m) \
                 / ( const.G * const.M_sun) )**2 \
                 * v_max_sigma_m_per_s**2 \
              + ((-1 * v_max_best_m_per_s**2 * depro_dist_end_m) \
                 / ( const.G**2 * const.M_sun) )**2 \
                 * ( const.G.uncertainty * const.G.unit)**2 \
              + ((-1 * v_max_best_m_per_s**2 * depro_dist_end_m) \
                 / ( const.G * const.M_sun**2) )**2 \
                 * (const.M_sun.uncertainty * const.M_sun.unit)**2) \
              * ( u.M_sun)

            theorized_dmMass = gal_mass - sMass_processed
            theorized_dmMass_err = gal_mass_err  # no error assumed in
                                                 #   sMass_processed

        dmMass_to_sMass_ratio = theorized_dmMass / sMass_processed
        dmMass_to_sMass_ratio_err = theorized_dmMass_err / sMass_processed


        #######################################################################
        # Append the corresponding values to their respective arrays to
        #    write to the roatation curve file. The quantities are stirpped
        #    of their units at this stage in the algorithm because astropy
        #    Column objects cannot be created with quantities that have
        #    dimensions. The respective dimensions are added back when the
        #    Column objects are added to the astropy QTable.
        #----------------------------------------------------------------------
        gal_mass_master.append( gal_mass / u.M_sun)
        gal_mass_err_master.append( gal_mass_err / u.M_sun)
        theorized_dmMass_master.append( theorized_dmMass / u.M_sun)
        theorized_dmMass_err_master.append( theorized_dmMass_err / u.M_sun)
        sMass_processed_master.append( sMass_processed / u.M_sun)
        dm_to_stellar_mass_ratio_master.append( dmMass_to_sMass_ratio)
        dm_to_stellar_mass_ratio_err_master.append( dmMass_to_sMass_ratio_err)
        #######################################################################


        #######################################################################
        # Plot the fitted rotation curve along with its errorbars. In addition,
        #    several statistics about the goodness of fit, and mass interior to
        #    the outermost radius recorded are displayed in the lower right
        #    side of the figure.
        #----------------------------------------------------------------------
        fitted_rot_curve_fig = plt.figure(20)
        plt.errorbar( depro_dist, rot_vel_data,
                     yerr=rot_vel_data_err, fmt='o', color='purple',
                     markersize=4, capthick=1, capsize=3)

        plt.plot( np.linspace( 0, depro_dist[-1] / u.kpc, 10000),
                 rot_fit_func(np.linspace( 0, depro_dist[-1] / u.kpc, 10000),
                              v_max_best.value / ( u.km / u.s),
                              r_turn_best.value / ( u.kpc),
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

        plt.savefig( IMAGE_DIR + '\\fitted_rotation_curves\\' + gal_id +\
                    '_fitted_rotation_curve.png')
        plt.show()
        plt.close()
        #######################################################################
    # ~   ~   ~   ~   ~   ~   ~   ~   ~   ~   ~   ~   ~   ~   ~   ~   ~   ~   ~


    ###########################################################################
    # Convert the data arrays into Column objects to add to the rotation
    #    curve data table.
    #--------------------------------------------------------------------------
    gal_mass_col = Column( gal_mass_master)
    gal_mass_err_col = Column( gal_mass_err_master)
    theorized_dmMass_col = Column( theorized_dmMass_master)
    theorized_dmMass_err_col = Column( theorized_dmMass_err_master)
    sMass_col = Column( sMass_processed_master)
    dmMass_to_sMass_ratio_col = Column( dm_to_stellar_mass_ratio_master)
    dmMass_to_sMass_ratio_err_col = Column(
                                       dm_to_stellar_mass_ratio_err_master)
    ###########################################################################


    ###########################################################################
    # Add the column objects to astropy QTables.
    #--------------------------------------------------------------------------
    mass_estimate_table = QTable( [ gal_mass_col * (u.M_sun),
                                   gal_mass_err_col * (u.M_sun),
                                   theorized_dmMass_col * (u.M_sun),
                                   theorized_dmMass_err_col * (u.M_sun),
                                   sMass_col * (u.M_sun),
                                   dmMass_to_sMass_ratio_col,
                                   dmMass_to_sMass_ratio_err_col],
                          names = ['total_mass',
                                   'total_mass_error',
                                   'dmMass',
                                   'dmMass_error',
                                   'sMass',
                                   'dmMass_to_sMass_ratio',
                                   'dmMass_to_sMass_ratio_error'])
    ###########################################################################

    return mass_estimate_table


def write_mass_estimates( mass_estimate_table, MASTER_FILE_NAME):
    """Writes the mass estimates obtained in the estimate_dark_matter
    function. A try-except statementis included in the event that the master
    file already has data for these columns, and the data needs to overwritten.

    @param:
        mass_estimate_table:
            astropy QTable with the stellar, dark matter, and total mass
            estimates interior to the outermost radius in the galaxy that was
            analyzed in this project

        MASTER_FILE_NAME:
            string representation of the master file containing the best fit
            parameters along with their errors, the mass estimates obtained in
            estimate_dark_matter, and identifying information about each
            galaxy
    """
    master_table = ascii.read( MASTER_FILE_NAME, format='ecsv')

    ###########################################################################
    # NOTE: Because the add_column method will not overwrite existing columns
    #       with the same name, the columns are first removed from master_table
    #       if they exist and then added back with the updated values.
    #--------------------------------------------------------------------------
    del_col_names = ['total_mass',
                     'total_mass_error',
                     'dmMass',
                     'dmMass_error',
                     'sMass',
                     'dmMass_to_sMass_ratio',
                     'dmMass_to_sMass_ratio_error']

    for col in del_col_names:
        try:
            master_table.remove_column( col)
        except KeyError:
            print("USER GENERATED KeyError: \n" + \
                  "Column name not found in del_col_names")
    ###########################################################################

    master_table.add_column( mass_estimate_table['total_mass'])
    master_table.add_column( mass_estimate_table['total_mass_error'])
    master_table.add_column( mass_estimate_table['dmMass'])
    master_table.add_column( mass_estimate_table['dmMass_error'])
    master_table.add_column( mass_estimate_table['dmMass_to_sMass_ratio'])
    master_table.add_column( mass_estimate_table['dmMass_to_sMass_ratio_error'])

    ascii.write(master_table,
                MASTER_FILE_NAME,
                format = 'ecsv',
                overwrite = True)


def plot_mass_ratios( mass_estimate_table, MASTER_FILE_NAME, IMAGE_DIR):
    """Function to histogram the dark matter to stellar mass ratios.

    @param:
        mass_estimates_table:
            astropy QTable with the stellar, dark matter, and total mass
            estimates interior to the outermost radius in the galaxy that was
            analyzed in this project

        MASTER_FILE_NAME:
            string representation of the master file containing the best fit
            parameters along with their errors, the mass estimates obtained in
            estimate_dark_matter, and identifying information about each
            galaxy

        IMAGE_DIR:
            string representation of the file path that pictures are saved to
    """
    ###########################################################################
    # Hard-coded entry for the bins for the histrogram plots at the end of this
    #    function.
    #--------------------------------------------------------------------------
    BINS = np.linspace(0, 1000, 50)
    ###########################################################################


    ###########################################################################
    # Initialize the master arrays that will hold the wall and void mass ratios
    #    for plotting the histogram.
    #--------------------------------------------------------------------------
    dm_to_stellar_mass_ratio_wall = []
    dm_to_stellar_mass_ratio_void = []
    ###########################################################################


    ###########################################################################
    # Import 'vflag' from the 'master_file,' and gather the mass
    #    ratios from the 'mass_estimate_table' obtained in
    #    'estimate_dark_matter().'
    #--------------------------------------------------------------------------
    master_table = ascii.read( MASTER_FILE_NAME, format='ecsv')
    vflag_list = master_table['vflag'].data

    dm_to_stellar_mass_ratio_list = mass_estimate_table[
                                        'dmMass_to_sMass_ratio'].data
    ###########################################################################


    ###########################################################################
    # Separate the mass ratios according to wall or void.
    #--------------------------------------------------------------------------
    for vflag, mass_ratio in zip( vflag_list, dm_to_stellar_mass_ratio_list):
        if vflag == 0:
            dm_to_stellar_mass_ratio_wall.append( mass_ratio)

        elif vflag == 1:
            dm_to_stellar_mass_ratio_void.append( mass_ratio)
    ###########################################################################


    ###########################################################################
    # Calculate the mean, RMS, and standard deviation for the void, wall, and
    #    total distributions in the histogram below.
    #--------------------------------------------------------------------------
    ratio_mean = np.mean( dm_to_stellar_mass_ratio_list)
    ratio_wall_mean = np.mean( dm_to_stellar_mass_ratio_wall)
    ratio_void_mean = np.mean( dm_to_stellar_mass_ratio_void)
    ratio_stdev = np.std( dm_to_stellar_mass_ratio_list)
    ratio_wall_stdev = np.std( dm_to_stellar_mass_ratio_wall)
    ratio_void_stdev = np.std( dm_to_stellar_mass_ratio_void)
    ratio_rms = np.sqrt( np.mean( dm_to_stellar_mass_ratio_list**2))
    ratio_wall_rms = np.sqrt( np.mean( dm_to_stellar_mass_ratio_wall**2))
    ratio_void_rms = np.sqrt( np.mean( dm_to_stellar_mass_ratio_void**2))
    ###########################################################################


    ###########################################################################
    # Histogram the dark matter to stellar mass ratios as separated by wall
    #    versus void as well as the total distribution.
    #--------------------------------------------------------------------------
    dm_to_stellar_mass_hist = plt.figure()
    xmin, xmax = plt.xlim()
    x = np.linspace(xmin, xmax, 100)

    plt.hist( dm_to_stellar_mass_ratio_list,
             BINS, color='green', density=True, alpha=0.9)
    p = norm.pdf(x, ratio_mean, ratio_stdev)
    plt.plot(x, p, 'g--', linewidth=2)
    plt.axvline( ratio_mean, color='green', linestyle='-', linewidth=1.5)
    plt.axvline( ratio_mean + ratio_stdev,
                color='green', linestyle=':', linewidth=1)
    plt.axvline( ratio_mean - ratio_stdev,
                color='green', linestyle=':', linewidth=1)
    plt.axvline( ratio_mean + 2*ratio_stdev,
                color='green', linestyle=':', linewidth=1)
    plt.axvline( ratio_mean - 2*ratio_stdev,
                color='green', linestyle=':', linewidth=1)
    _, mean_ratio_ = plt.ylim()
    plt.text(ratio_mean + ratio_mean/10,
         mean_ratio_ - mean_ratio_/10,
         'Mean: {:.2f}'.format( ratio_mean))

    plt.hist( dm_to_stellar_mass_ratio_wall,
             BINS, color='black', density=True, alpha=0.8)
    p = norm.pdf(x, ratio_wall_mean, ratio_wall_stdev)
    plt.plot(x, p, 'k--', linewidth=2)
    plt.axvline( ratio_wall_mean, color='black', linestyle='-', linewidth=1.5)
    plt.axvline( ratio_wall_mean + ratio_wall_stdev,
                color='black', linestyle=':', linewidth=1)
    plt.axvline( ratio_wall_mean - ratio_wall_stdev,
                color='black', linestyle=':', linewidth=1)
    plt.axvline( ratio_wall_mean + 2*ratio_wall_stdev,
                color='black', linestyle=':', linewidth=1)
    plt.axvline( ratio_wall_mean - 2*ratio_wall_stdev,
                color='black', linestyle=':', linewidth=1)
    _, mean_wall_ratio_ = plt.ylim()
    plt.text(ratio_wall_mean + ratio_wall_mean/10,
         mean_wall_ratio_ - mean_wall_ratio_/10,
         'Mean: {:.2f}'.format( ratio_wall_mean))

    plt.hist( dm_to_stellar_mass_ratio_void,
             BINS, color='red', density=True, alpha=0.5)
    p = norm.pdf(x, ratio_void_mean, ratio_void_stdev)
    plt.plot(x, p, 'r--', linewidth=2)
    plt.axvline( ratio_void_mean, color='red', linestyle='-', linewidth=1.5)
    plt.axvline( ratio_void_mean + ratio_void_stdev,
                color='red', linestyle=':', linewidth=1)
    plt.axvline( ratio_void_mean - ratio_void_stdev,
                color='red', linestyle=':', linewidth=1)
    plt.axvline( ratio_void_mean + 2*ratio_void_stdev,
                color='red', linestyle=':', linewidth=1)
    plt.axvline( ratio_void_mean - 2*ratio_void_stdev,
                color='red', linestyle=':', linewidth=1)
    _, mean_void_ratio_ = plt.ylim()
    plt.text(ratio_void_mean + ratio_void_mean/10,
         mean_void_ratio_ - mean_void_ratio_/10,
         'Mean: {:.2f}'.format( ratio_void_mean))


    ax = dm_to_stellar_mass_hist.add_subplot(111)
    plt.tick_params( axis='both', direction='in')
    ax.yaxis.set_ticks_position('both')
    ax.xaxis.set_ticks_position('both')
    plt.ylabel(r'$Fraction_{galaxy}$')
    plt.xlabel(r'$\frac{m_{*}}{m_{DM}}$')
    plt.title(r'$\frac{m_{*}}{m_{DM}}$ Ratio Histogram')

    textstr = '\n'.join((
          r'STDEV: $%.2f$' % ( ratio_stdev, ),
          r'$STDEV_{wall}$: $%.2f$' % ( ratio_wall_stdev, ),
          r'$STDEV_{void}$: $%.2f$' % ( ratio_void_stdev, ),
          r'RMS: $%.2f$' % ( ratio_rms, ),
          r'$RMS_{wall}$: $%.2f$' % ( ratio_wall_rms, ),
          r'$RMS_{void}$: $%.2f$' % ( ratio_void_rms, )))

    props = dict( boxstyle='round', facecolor='cornsilk', alpha=0.6)

    ax.text(0.72, 0.95, textstr,
            verticalalignment='top', horizontalalignment='left',
            transform=ax.transAxes,
            color='black', fontsize=8, bbox=props)

    plt.savefig( IMAGE_DIR + '\\histograms\\dm_to_stellar_mass_ratio_hist.png')
    plt.show()
    plt.close()
    ###########################################################################


    ###########################################################################
    # Histogram the dark matter to stellar mass ratios as CDF separated by wall
    #    versus void distributions.
    #--------------------------------------------------------------------------
    cdf_range = max(
            max( dm_to_stellar_mass_ratio_wall),
            max( dm_to_stellar_mass_ratio_void))
    ks_stat, p_val = ks_2samp( dm_to_stellar_mass_ratio_wall,
                              dm_to_stellar_mass_ratio_void)

    dm_to_stellar_mass_cdf = plt.figure()
    plt.hist( dm_to_stellar_mass_ratio_wall, bins=1000, range=cdf_range,
             density=True, cumulative=True, histtype='step', color='black',
             linewidth=1.5)
    plt.hist( dm_to_stellar_mass_ratio_void, bins=1000, range=cdf_range,
             density=True, cumulative=True, histtype='step', color='red',
             linewidth=1.5)

    ax = dm_to_stellar_mass_cdf.add_subplot(111)
    plt.tick_params( axis='both', direction='in')
    ax.yaxis.set_ticks_position('both')
    ax.xaxis.set_ticks_position('both')
    plt.ylabel(r'$Fraction_{galaxy}$')
    plt.xlabel(r'$\frac{m_{*}}{m_{DM}}$')
    plt.title(r'$\frac{m_{*}}{m_{DM}}$ Ratio CDF')

    plt.text( 280, 0.15, "p-val: " + "{:.{}f}".format( p_val, 3))

    plt.savefig( IMAGE_DIR + '\\histograms\\dm_to_stellar_mass_ratio_hist.png')
    plt.show()
    plt.close()
    ###########################################################################


def analyze_rot_curve_discrep( rot_curve_files, gal_stat_files,
                      MASTER_FILE_NAME, ROT_CURVE_MASTER_FOLDER, IMAGE_DIR):
    """Analyze the discrepancies between minimum and maximum rotation curves.


    @param:
        rot_curve_files:
            a string list containing all of the data files within the
            'rotation_curve_vX_X' output folder

        gal_stats_files:
            string list of galaxies for which general statistical data exists

        MASTER_FILE_NAME:
            string representation of the name of the file which 'vflag_list'
            will be written to

        ROT_CURVE_MASTER_FOLDER:
            string representation of the path of the folder containing all of
            the galaxy data for all of the galaxies in the MaNGA survey
    """
    ###########################################################################
    # Hard-coded entry for the bins for the histrogram plots at the end of this
    #    function.
    #--------------------------------------------------------------------------
    BINS = np.linspace( -1000, 1000, 50)
    ###########################################################################


    ###########################################################################
    # Initialize the arrays to store the v_max differences, inclination angles,
    #    mass ratios, and the errors associed with them as available.
    #--------------------------------------------------------------------------
    v_max_diff_wall = []
    v_max_diff_wall_error = []
    v_max_diff_void = []
    v_max_diff_void_error = []
    v_max_diff_other = []
    v_max_diff_other_error = []

    inclin_angle_wall = []
    inclin_angle_void = []
    inclin_angle_other = []

    mass_ratio_wall = []
    mass_ratio_wall_error = []
    mass_ratio_void = []
    mass_ratio_void_error = []
    mass_ratio_other = []
    mass_ratio_other_error = []
    ###########################################################################


    ###########################################################################
    # Import the necessary data from the master_file.
    #--------------------------------------------------------------------------
    master_table = ascii.read( MASTER_FILE_NAME, format='ecsv')
    vflag_list = master_table['vflag']
    pos_v_max = master_table['pos_max_velocity_best']
    pos_v_max_error = master_table['pos_max_velocity_error']
    neg_v_max = master_table['neg_max_velocity_best']
    neg_v_max_error = master_table['neg_max_velocity_error']
    axes_ratios = master_table['NSA_b/a']
    dmMass_to_sMass_ratio = master_table['dmMass_to_sMass_ratio']
    mass_ratio_error = master_table['dmMass_to_sMass_ratio_error']
    ###########################################################################


    ###########################################################################
    # For each galaxy in the master_table, calculate the difference in the
    #    v_max parameter for the positive and negative rotation curves and the
    #    inclination angle of the galaxy. Add these quantities along with their
    #    errors as available and the stellar mass to dark matter mass ratio
    #    along with its error to arrays separated by the total distribution,
    #    voids, walls, galaxies on the edge of the MaNGA footprint, and
    #    galaxies not found within the MaNGA footprint.
    #--------------------------------------------------------------------------
    for vflag, pos_v_max, pos_v_max_err, neg_v_max, neg_v_max_err, b_over_a, \
    ratio, ratio_err in \
      zip( vflag_list, pos_v_max, pos_v_max_error, neg_v_max, neg_v_max_error,
          axes_ratios, dmMass_to_sMass_ratio, mass_ratio_error):

        v_max_diff = pos_v_max - neg_v_max
        v_max_diff_err = np.sqrt( pos_v_max_err**2 + neg_v_max_err**2)

        inc_angle = np.arccos( b_over_a)

        if vflag == 0:
            v_max_diff_wall.append( v_max_diff)
            v_max_diff_wall_error.append( v_max_diff_err)
            inclin_angle_wall.append( inc_angle)
            mass_ratio_wall.append( ratio)
            mass_ratio_wall_error.append( ratio_err)

        if vflag == 1:
            v_max_diff_void.append( v_max_diff)
            v_max_diff_void_error.append( v_max_diff_err)
            inclin_angle_void.append( inc_angle)
            mass_ratio_void.append( ratio)
            mass_ratio_void_error.append( ratio_err)

        if vflag == 2 or vflag == -9:
            v_max_diff_other.append( v_max_diff)
            v_max_diff_other_error.append( v_max_diff_err)
            inclin_angle_other.append( inc_angle)
            mass_ratio_other.append( ratio)
            mass_ratio_other_error.append( ratio_err)
    ###########################################################################


    ###########################################################################
    # Calculate the mean, RMS, and standard deviation for the void, wall, and
    #    total distributions in the histogram below.
    #--------------------------------------------------------------------------
    v_max_wall_mean = np.mean( v_max_diff_wall)
    v_max_void_mean = np.mean( v_max_diff_void)
    v_max_other_mean = np.mean( v_max_diff_other)
    v_max_wall_stdev = np.std( v_max_diff_wall)
    v_max_void_stdev = np.std( v_max_diff_void)
    v_max_other_stdev = np.std( v_max_diff_other)
    v_max_wall_rms = np.sqrt( np.mean( v_max_diff_wall**2))
    v_max_void_rms = np.sqrt( np.mean( v_max_diff_void**2))
    v_max_other_rms = np.sqrt( np.mean( v_max_diff_other**2))
    ###########################################################################


    ###########################################################################
    # Variables that are used in the resolution of the fitting of the gaussian
    #    are located directly below.
    #--------------------------------------------------------------------------
    xmin, xmax = plt.xlim()
    x = np.linspace(xmin, xmax, 100)
    ###########################################################################


    ###########################################################################
    # Plot the v_max_diff distribution and separate the distributions into
    #    walls, voids, and other.
    #--------------------------------------------------------------------------
    v_max_diff_hist = plt.figure()
    plt.title(r"$v_{max}$ Difference Histogram")

    plt.hist( v_max_diff_other, BINS, color='green', density=True, alpha=1.0)
    p = norm.pdf(x, v_max_other_mean, v_max_other_stdev)
    plt.plot(x, p, 'g--', linewidth=2)
    plt.axvline( v_max_other_mean, color='green', linestyle='-', linewidth=1.5)
    plt.axvline( v_max_other_mean + v_max_other_stdev,
                color='green', linestyle=':', linewidth=1)
    plt.axvline( v_max_other_mean - v_max_other_stdev,
                color='green', linestyle=':', linewidth=1)
    plt.axvline( v_max_other_mean + 2*v_max_other_stdev,
                color='green', linestyle=':', linewidth=1)
    plt.axvline( v_max_other_mean - 2*v_max_other_stdev,
                color='green', linestyle=':', linewidth=1)
    _, mean_v_max_other_ = plt.ylim()
    plt.text(v_max_other_mean + v_max_other_mean/10,
         mean_v_max_other_ - mean_v_max_other_/10,
         'Mean: {:.2f}'.format( v_max_other_mean))

    plt.hist( v_max_diff_wall, BINS, color='black', density=True, alpha=0.8)
    p = norm.pdf(x, v_max_wall_mean, v_max_wall_stdev)
    plt.plot(x, p, 'k--', linewidth=2)
    plt.axvline( v_max_wall_mean, color='black', linestyle='-', linewidth=1.5)
    plt.axvline( v_max_wall_mean + v_max_wall_stdev,
                color='black', linestyle=':', linewidth=1)
    plt.axvline( v_max_wall_mean - v_max_wall_stdev,
                color='black', linestyle=':', linewidth=1)
    plt.axvline( v_max_wall_mean + 2*v_max_wall_stdev,
                color='black', linestyle=':', linewidth=1)
    plt.axvline( v_max_wall_mean - 2*v_max_wall_stdev,
                color='black', linestyle=':', linewidth=1)
    _, mean_wall_v_max_ = plt.ylim()
    plt.text(v_max_wall_mean + v_max_wall_mean/10,
         mean_wall_v_max_ - mean_wall_v_max_/10,
         'Mean: {:.2f}'.format( v_max_wall_mean))

    plt.hist( v_max_diff_void, BINS, color='red', density=True, alpha=0.5)
    p = norm.pdf(x, v_max_void_mean, v_max_void_stdev)
    plt.plot(x, p, 'r--', linewidth=2)
    plt.axvline( v_max_void_mean, color='red', linestyle='-', linewidth=1.5)
    plt.axvline( v_max_void_mean + v_max_void_stdev,
                color='red', linestyle=':', linewidth=1)
    plt.axvline( v_max_void_mean - v_max_void_stdev,
                color='red', linestyle=':', linewidth=1)
    plt.axvline( v_max_void_mean + 2*v_max_void_stdev,
                color='red', linestyle=':', linewidth=1)
    plt.axvline( v_max_void_mean - 2*v_max_void_stdev,
                color='red', linestyle=':', linewidth=1)
    _, mean_void_v_max_ = plt.ylim()
    plt.text(v_max_void_mean + v_max_void_mean/10,
         mean_void_v_max_ - mean_void_v_max_/10,
         'Mean: {:.2f}'.format( v_max_void_mean))


    ax = v_max_diff_hist.add_subplot(111)
    plt.tick_params( axis='both', direction='in')
    ax.yaxis.set_ticks_position('both')
    ax.xaxis.set_ticks_position('both')
    plt.ylabel(r'$Fraction_{galaxy}$')
    plt.xlabel(r'$v_{max} Difference$ [km/s]')

    textstr = '\n'.join((
          r'$STDEV_{wall}$: $%.2f$' % ( v_max_wall_stdev, ),
          r'$STDEV_{void}$: $%.2f$' % ( v_max_void_stdev, ),
          r'STDEV: $%.2f$' % ( v_max_other_stdev, ),
          r'$RMS_{wall}$: $%.2f$' % ( v_max_wall_rms, ),
          r'$RMS_{void}$: $%.2f$' % ( v_max_void_rms, ),
          r'RMS: $%.2f$' % ( v_max_other_rms, )))

    props = dict( boxstyle='round', facecolor='cornsilk', alpha=0.6)

    ax.text(0.72, 0.95, textstr,
            verticalalignment='top', horizontalalignment='left',
            transform=ax.transAxes,
            color='black', fontsize=8, bbox=props)

    plt.savefig( IMAGE_DIR + '\\v_max_diff_hist.png' )
    plt.show()
    plt.close()
    ###########################################################################


    ###########################################################################
    # Plot the difference in the fitted v_max parameters from the positive and
    #    negative rotation curves against the inclination angle of the galaxy
    #    to see if there is any correlation.
    #
    # NOTE: Distributions are separated by wall, void, if the galaxy is cut off
    #       by the edge of the footprint, or if the galaxy is not found within
    #       the footprint.
    #--------------------------------------------------------------------------
    v_max_diff_vs_inclination_fig = plt.figure()
    plt.title(r"$v_{max}$ Difference VS Inclination Angle")
    plt.errorbar( inclin_angle_wall, v_max_diff_wall,
                 yerr=v_max_diff_wall_error, fmt='kv', ecolor='black')
    plt.errorbar( inclin_angle_void, v_max_diff_void,
                 yerr=v_max_diff_void_error, fmt='ro', ecolor='red')
    plt.errorbar( inclin_angle_other, v_max_diff_other,
                 yerr=v_max_diff_other_error,
                 fmt='go', ecolor='green', fillstyle='none')
#    plt.errorbar( inclination_angle, v_max_difference,
#                 yerr=v_max_difference_error , fmt='ko', ecolor='gray')

    ax = v_max_diff_vs_inclination_fig.add_subplot(111)
    plt.tick_params( axis='both', direction='in')
    ax.yaxis.set_ticks_position('both')
    ax.xaxis.set_ticks_position('both')
    plt.ylabel(r'$v_{max} Difference$ [km/s]')
    plt.xlabel(r'Inclination Angle [rad]')

    plt.savefig( IMAGE_DIR + '\\v_max_vs_inclination.png' )
    plt.show()
    plt.close()
    ###########################################################################


    ###########################################################################
    # Plot the difference in the fitted v_max parameters from the positive and
    #    negative rotation curves against the stellar mass to dark matter mass
    #    ratio to see if there is any correlation.
    #
    # NOTE: Distributions are separated by wall, void, if the galaxy is cut off
    #       by the edge of the footprint, or if the galaxy is not found within
    #       the footprint.
    #--------------------------------------------------------------------------
    v_max_diff_vs_mass_ratio_fig = plt.figure()
    plt.title(r"$v_{max}$ Difference VS Mass Ratio")
    plt.errorbar( mass_ratio_wall, v_max_diff_wall,
                 xerr=mass_ratio_wall_error, yerr=v_max_diff_wall_error,
                 fmt='kv', ecolor='black')
    plt.errorbar( mass_ratio_void, v_max_diff_void,
                 xerr=mass_ratio_void_error, yerr=v_max_diff_void_error,
                 fmt='ro', ecolor='red')
    plt.errorbar( mass_ratio_other, v_max_diff_other,
                 xerr=mass_ratio_other_error, yerr=v_max_diff_other_error,
                 fmt='go', ecolor='green', fillstyle='none')
#    plt.errorbar( mass_ratio, v_max_difference,
#                 xerr=mass_ratio_error, yerr=v_max_difference_error,
#                 fmt='ko', ecolor='gray')

    ax = v_max_diff_vs_mass_ratio_fig.add_subplot(111)
    plt.tick_params( axis='both', direction='in')
    ax.yaxis.set_ticks_position('both')
    ax.xaxis.set_ticks_position('both')
    plt.ylabel(r'$v_{max} Difference$ [km/s]')
    plt.xlabel('Mass Ratio')

    plt.savefig( IMAGE_DIR + '\\v_max_vs_mass_ratio.png' )
    plt.show()
    plt.close()
    ###########################################################################


def analyze_chi_square( MASTER_FILE_NAME, IMAGE_DIR):
    """Histogram the chi square values for the fit function parameters obtained
    in fitting the curves.

    Chi square values are separated into different figures by min/max rotation
    curve. Within each histogram each chi square distribution is also separated
    by wall versus void. The total distribution is also plotted for comparison.

    @param:
        MASTER_FILE_NAME:
            string representation of the name of the file which 'vflag_list'
            and the chi square values will be pulled from
    """
    ###########################################################################
    # Hard-coded entry for the bins for the histrogram plots at the end of this
    #    function.
    #--------------------------------------------------------------------------
    BINS = np.linspace( 0, 10000, 500)
    ###########################################################################


    ###########################################################################
    # Import the necessary data from the master_file.
    #--------------------------------------------------------------------------
    master_file = ascii.read( MASTER_FILE_NAME, format='ecsv')
    vflag_list = master_file['vflag']
    avg_chi_square_rot_master = master_file['chi_square_rot']
    pos_chi_square_rot_master = master_file['pos_chi_square_rot']
    neg_chi_square_rot_master = master_file['neg_chi_square_rot']
    ###########################################################################


    ###########################################################################
    # Initialize the arrays to store the chi square values.
    #
    # NOTE: chi square values are separated into difference histograms by the
    #       average of the two rotation curves and the positive and negative
    #       rotation curves.
    # NOTE: within these histograms, galaxies are separated by those in walls,
    #       those in voids, and other.
    #--------------------------------------------------------------------------
    avg_chi_square_rot_wall = []
    avg_chi_square_rot_void = []
    avg_chi_square_rot_other = []
    pos_chi_square_rot_wall = []
    pos_chi_square_rot_void = []
    pos_chi_square_rot_other = []
    neg_chi_square_rot_wall = []
    neg_chi_square_rot_void = []
    neg_chi_square_rot_other = []
    ###########################################################################


    ###########################################################################
    # For each galaxy in the master_table, separate the chi square values by
    #    if the galaxy is within a wall, a void, or other.
    #--------------------------------------------------------------------------
    for vflag, chi_square, pos_chi_square, neg_chi_square \
      in zip( vflag_list, avg_chi_square_rot_master,
             pos_chi_square_rot_master, neg_chi_square_rot_master):
        if vflag == 0:
            avg_chi_square_rot_wall.append( chi_square)
            pos_chi_square_rot_wall.append( pos_chi_square)
            neg_chi_square_rot_wall.append( neg_chi_square)

        if vflag == 1:
            avg_chi_square_rot_void.append( chi_square)
            pos_chi_square_rot_void.append( pos_chi_square)
            neg_chi_square_rot_void.append( neg_chi_square)

        if vflag == 2 or vflag == -9:
            avg_chi_square_rot_other.append( chi_square)
            pos_chi_square_rot_other.append( pos_chi_square)
            neg_chi_square_rot_other.append( neg_chi_square)
    ###########################################################################


    ###########################################################################
    # Calculate the mean, RMS, and standard deviation for the void, wall, and
    #    total distributions in the average, positive and negative, histograms
    #    below.
    #--------------------------------------------------------------------------
    avg_chi_square_rot_wall_mean = np.mean( avg_chi_square_rot_wall)
    avg_chi_square_rot_void_mean = np.mean( avg_chi_square_rot_void)
    avg_chi_square_rot_other_mean = np.mean( avg_chi_square_rot_other)
    avg_chi_square_rot_wall_stdev = np.std( avg_chi_square_rot_wall)
    avg_chi_square_rot_void_stdev = np.std( avg_chi_square_rot_void)
    avg_chi_square_rot_other_stdev = np.std( avg_chi_square_rot_other)
    avg_chi_square_rot_wall_rms = np.sqrt( np.mean(
            avg_chi_square_rot_wall**2))
    avg_chi_square_rot_void_rms = np.sqrt( np.mean(
            avg_chi_square_rot_void**2))
    avg_chi_square_rot_other_rms = np.sqrt( np.mean(
            avg_chi_square_rot_other**2))

    pos_chi_square_rot_wall_mean = np.mean( pos_chi_square_rot_wall)
    pos_chi_square_rot_void_mean = np.mean( pos_chi_square_rot_void)
    pos_chi_square_rot_other_mean = np.mean( pos_chi_square_rot_other)
    pos_chi_square_rot_wall_stdev = np.std( pos_chi_square_rot_wall)
    pos_chi_square_rot_void_stdev = np.std( pos_chi_square_rot_void)
    pos_chi_square_rot_other_stdev = np.std( pos_chi_square_rot_other)
    pos_chi_square_rot_wall_rms = np.sqrt( np.mean(
            pos_chi_square_rot_wall**2))
    pos_chi_square_rot_void_rms = np.sqrt( np.mean(
            pos_chi_square_rot_void**2))
    pos_chi_square_rot_other_rms = np.sqrt( np.mean(
            pos_chi_square_rot_other**2))

    neg_chi_square_rot_wall_mean = np.mean( neg_chi_square_rot_wall)
    neg_chi_square_rot_void_mean = np.mean( neg_chi_square_rot_void)
    neg_chi_square_rot_other_mean = np.mean( neg_chi_square_rot_other)
    neg_chi_square_rot_wall_stdev = np.std( neg_chi_square_rot_wall)
    neg_chi_square_rot_void_stdev = np.std( neg_chi_square_rot_void)
    neg_chi_square_rot_other_stdev = np.std( neg_chi_square_rot_other)
    neg_chi_square_rot_wall_rms = np.sqrt( np.mean(
            neg_chi_square_rot_wall**2))
    neg_chi_square_rot_void_rms = np.sqrt( np.mean(
            neg_chi_square_rot_void**2))
    neg_chi_square_rot_other_rms = np.sqrt( np.mean(
            neg_chi_square_rot_other**2))
    ###########################################################################


    ###########################################################################
    # Variables that are used in the fitting of the gaussian are located
    #    directly below.
    #--------------------------------------------------------------------------
    xmin, xmax = plt.xlim()
    x = np.linspace(xmin, xmax, 100)
    ###########################################################################


    ###########################################################################
    # Plot the chi square distribution for the average chi rotation curve and
    #    separate the distributions into walls and voids.
    #--------------------------------------------------------------------------
    avg_chi_square_rot_hist = plt.figure()
    plt.title("Chi Square Histogram")


    plt.hist( avg_chi_square_rot_other,
             BINS, color='green', density=True, alpha=1.0)
    p = norm.pdf(x,
                 avg_chi_square_rot_other_mean, avg_chi_square_rot_other_stdev)
    plt.plot(x, p, 'g--', linewidth=2)
    plt.axvline( avg_chi_square_rot_other_mean,
                color='green', linestyle='-', linewidth=1.5)
    plt.axvline(
            avg_chi_square_rot_other_mean + avg_chi_square_rot_other_stdev,
            color='green', linestyle=':', linewidth=1)
    plt.axvline(
            avg_chi_square_rot_other_mean - avg_chi_square_rot_other_stdev,
            color='green', linestyle=':', linewidth=1)
    plt.axvline(
            avg_chi_square_rot_other_mean + 2*avg_chi_square_rot_other_stdev,
            color='green', linestyle=':', linewidth=1)
    plt.axvline(
            avg_chi_square_rot_other_mean - 2*avg_chi_square_rot_other_stdev,
            color='green', linestyle=':', linewidth=1)
    _, mean_avg_chi_square_rot_other_ = plt.ylim()
    plt.text(avg_chi_square_rot_other_mean + avg_chi_square_rot_other_mean/10,
         mean_avg_chi_square_rot_other_ - mean_avg_chi_square_rot_other_/10,
         'Mean: {:.2f}'.format( avg_chi_square_rot_other_mean))


    plt.hist( avg_chi_square_rot_wall,
             BINS, color='black', density=True, alpha=0.8)
    p = norm.pdf(x,
                 avg_chi_square_rot_wall_mean, avg_chi_square_rot_wall_stdev)
    plt.plot(x, p, 'k--', linewidth=2)
    plt.axvline( avg_chi_square_rot_wall_mean,
                color='black', linestyle='-', linewidth=1.5)
    plt.axvline( avg_chi_square_rot_wall_mean + avg_chi_square_rot_wall_stdev,
                color='black', linestyle=':', linewidth=1)
    plt.axvline( avg_chi_square_rot_wall_mean - avg_chi_square_rot_wall_stdev,
                color='black', linestyle=':', linewidth=1)
    plt.axvline(
            avg_chi_square_rot_wall_mean + 2*avg_chi_square_rot_wall_stdev,
            color='black', linestyle=':', linewidth=1)
    plt.axvline(
            avg_chi_square_rot_wall_mean - 2*avg_chi_square_rot_wall_stdev,
            color='black', linestyle=':', linewidth=1)
    _, mean_wall_avg_chi_square_rot_ = plt.ylim()
    plt.text(avg_chi_square_rot_wall_mean + avg_chi_square_rot_wall_mean/10,
         mean_wall_avg_chi_square_rot_ - mean_wall_avg_chi_square_rot_/10,
         'Mean: {:.2f}'.format( avg_chi_square_rot_wall_mean))

    plt.hist( avg_chi_square_rot_void,
             BINS, color='red', density=True, alpha=0.5)
    p = norm.pdf(x,
                 avg_chi_square_rot_void_mean, avg_chi_square_rot_void_stdev)
    plt.plot(x, p, 'r--', linewidth=2)
    plt.axvline( avg_chi_square_rot_void_mean,
                color='red', linestyle='-', linewidth=1.5)
    plt.axvline( avg_chi_square_rot_void_mean + avg_chi_square_rot_void_stdev,
                color='red', linestyle=':', linewidth=1)
    plt.axvline( avg_chi_square_rot_void_mean - avg_chi_square_rot_void_stdev,
                color='red', linestyle=':', linewidth=1)
    plt.axvline(
            avg_chi_square_rot_void_mean + 2*avg_chi_square_rot_void_stdev,
            color='red', linestyle=':', linewidth=1)
    plt.axvline(
            avg_chi_square_rot_void_mean - 2*avg_chi_square_rot_void_stdev,
            color='red', linestyle=':', linewidth=1)
    _, mean_void_avg_chi_square_rot_ = plt.ylim()
    plt.text(avg_chi_square_rot_void_mean + avg_chi_square_rot_void_mean/10,
         mean_void_avg_chi_square_rot_ - mean_void_avg_chi_square_rot_/10,
         'Mean: {:.2f}'.format( avg_chi_square_rot_void_mean))


    ax = avg_chi_square_rot_hist.add_subplot(111)
    plt.tick_params( axis='both', direction='in')
    ax.yaxis.set_ticks_position('both')
    ax.xaxis.set_ticks_position('both')
    plt.ylabel(r'$Fraction_{galaxy}$')
    plt.xlabel('Chi Square (Goodness of Fit)')

    textstr = '\n'.join((
          r'STDEV: $%.2f$' % ( avg_chi_square_rot_other_stdev, ),
          r'$STDEV_{wall}$: $%.2f$' % ( avg_chi_square_rot_wall_stdev, ),
          r'$STDEV_{void}$: $%.2f$' % ( avg_chi_square_rot_void_stdev, ),
          r'RMS: $%.2f$' % ( avg_chi_square_rot_other_rms, ),
          r'$RMS_{wall}$: $%.2f$' % ( avg_chi_square_rot_wall_rms, ),
          r'$RMS_{void}$: $%.2f$' % ( avg_chi_square_rot_void_rms, )))

    props = dict( boxstyle='round', facecolor='cornsilk', alpha=0.6)

    ax.text(0.72, 0.95, textstr,
            verticalalignment='top', horizontalalignment='left',
            transform=ax.transAxes,
            color='black', fontsize=8, bbox=props)


    plt.savefig( IMAGE_DIR + '\\histograms\\avg_chi_square_hist.png' )
    plt.show()
    plt.close()
    ###########################################################################


    ###########################################################################
    # Plot the chi square distribution for the maximum chi rotation curve and
    #    separate the distributions into walls and voids.
    #--------------------------------------------------------------------------
    pos_chi_square_rot_hist = plt.figure()
    plt.title("Chi Square (Positive) Histogram")
    # -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -
    plt.hist( pos_chi_square_rot_other,
             BINS, color='purple', density=True, alpha=1.0)
    p = norm.pdf(x,
                 pos_chi_square_rot_other_mean, pos_chi_square_rot_other_stdev)
    plt.plot(x, p, color='purple', linestyle='--', linewidth=2)
    plt.axvline( pos_chi_square_rot_other_mean,
                color='purple', linestyle='-', linewidth=1.5)
    plt.axvline(
            pos_chi_square_rot_other_mean + pos_chi_square_rot_other_stdev,
            color='purple', linestyle=':', linewidth=1)
    plt.axvline(
            pos_chi_square_rot_other_mean - pos_chi_square_rot_other_stdev,
            color='purple', linestyle=':', linewidth=1)
    plt.axvline(
            pos_chi_square_rot_other_mean + 2*pos_chi_square_rot_other_stdev,
            color='purple', linestyle=':', linewidth=1)
    plt.axvline(
            pos_chi_square_rot_other_mean - 2*pos_chi_square_rot_other_stdev,
            color='purple', linestyle=':', linewidth=1)
    _, mean_pos_chi_square_rot_other_ = plt.ylim()
    plt.text(pos_chi_square_rot_other_mean + pos_chi_square_rot_other_mean/10,
         mean_pos_chi_square_rot_other_ - mean_pos_chi_square_rot_other_/10,
         'Mean: {:.2f}'.format( pos_chi_square_rot_other_mean))


    plt.hist( pos_chi_square_rot_wall,
             BINS, color='black', density=True, alpha=0.8)
    p = norm.pdf(x,
                 pos_chi_square_rot_wall_mean, pos_chi_square_rot_wall_stdev)
    plt.plot(x, p, 'k--', linewidth=2)
    plt.axvline( pos_chi_square_rot_wall_mean,
                color='black', linestyle='-', linewidth=1.5)
    plt.axvline( pos_chi_square_rot_wall_mean + pos_chi_square_rot_wall_stdev,
                color='black', linestyle=':', linewidth=1)
    plt.axvline( pos_chi_square_rot_wall_mean - pos_chi_square_rot_wall_stdev,
                color='black', linestyle=':', linewidth=1)
    plt.axvline(
            pos_chi_square_rot_wall_mean + 2*pos_chi_square_rot_wall_stdev,
            color='black', linestyle=':', linewidth=1)
    plt.axvline(
            pos_chi_square_rot_wall_mean - 2*pos_chi_square_rot_wall_stdev,
            color='black', linestyle=':', linewidth=1)
    _, mean_wall_pos_chi_square_rot_ = plt.ylim()
    plt.text(pos_chi_square_rot_wall_mean + pos_chi_square_rot_wall_mean/10,
         mean_wall_pos_chi_square_rot_ - mean_wall_pos_chi_square_rot_/10,
         'Mean: {:.2f}'.format( pos_chi_square_rot_wall_mean))


    plt.hist( pos_chi_square_rot_void,
             BINS, color='red', density=True, alpha=0.5)
    p = norm.pdf(x,
                 pos_chi_square_rot_void_mean, pos_chi_square_rot_void_stdev)
    plt.plot(x, p, 'r--', linewidth=2)
    plt.axvline( pos_chi_square_rot_void_mean,
                color='red', linestyle='-', linewidth=1.5)
    plt.axvline( pos_chi_square_rot_void_mean + pos_chi_square_rot_void_stdev,
                color='red', linestyle=':', linewidth=1)
    plt.axvline( pos_chi_square_rot_void_mean - pos_chi_square_rot_void_stdev,
                color='red', linestyle=':', linewidth=1)
    plt.axvline(
            pos_chi_square_rot_void_mean + 2*pos_chi_square_rot_void_stdev,
            color='red', linestyle=':', linewidth=1)
    plt.axvline(
            pos_chi_square_rot_void_mean - 2*pos_chi_square_rot_void_stdev,
            color='red', linestyle=':', linewidth=1)
    _, mean_void_pos_chi_square_rot_ = plt.ylim()
    plt.text(pos_chi_square_rot_void_mean + pos_chi_square_rot_void_mean/10,
         mean_void_pos_chi_square_rot_ - mean_void_pos_chi_square_rot_/10,
         'Mean: {:.2f}'.format( pos_chi_square_rot_void_mean))


    ax = pos_chi_square_rot_hist.add_subplot(111)
    plt.tick_params( axis='both', direction='in')
    ax.yaxis.set_ticks_position('both')
    ax.xaxis.set_ticks_position('both')
    plt.ylabel(r'$Fraction_{galaxy}$')
    plt.xlabel('Chi Square (Goodness of Fit)')

    textstr = '\n'.join((
          r'STDEV: $%.2f$' % ( pos_chi_square_rot_other_stdev, ),
          r'$STDEV_{wall}$: $%.2f$' % ( pos_chi_square_rot_wall_stdev, ),
          r'$STDEV_{void}$: $%.2f$' % ( pos_chi_square_rot_void_stdev, ),
          r'RMS: $%.2f$' % ( pos_chi_square_rot_other_rms, ),
          r'$RMS_{wall}$: $%.2f$' % ( pos_chi_square_rot_wall_rms, ),
          r'$RMS_{void}$: $%.2f$' % ( pos_chi_square_rot_void_rms, )))

    props = dict( boxstyle='round', facecolor='cornsilk', alpha=0.6)

    ax.text(0.72, 0.95, textstr,
            verticalalignment='top', horizontalalignment='left',
            transform=ax.transAxes,
            color='black', fontsize=8, bbox=props)


    plt.savefig( IMAGE_DIR + '\\histograms\\pos_chi_square_hist.png' )
    plt.show()
    plt.close()
    ###########################################################################


    ###########################################################################
    # Plot the chi square distribution for the minimum chi rotation curve and
    #    separate the distributions into walls and voids.
    #--------------------------------------------------------------------------
    neg_chi_square_rot_hist = plt.figure()
    plt.title("Chi Square (Negative) Histogram")
    # -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -
    plt.hist( neg_chi_square_rot_other,
             BINS, color='blue', density=True, alpha=1.0)
    p = norm.pdf(x,
                 neg_chi_square_rot_other_mean, neg_chi_square_rot_other_stdev)
    plt.plot(x, p, 'g--', linewidth=2)
    plt.axvline( neg_chi_square_rot_other_mean,
                color='blue', linestyle='-', linewidth=1.5)
    plt.axvline(
            neg_chi_square_rot_other_mean + neg_chi_square_rot_other_stdev,
            color='blue', linestyle=':', linewidth=1)
    plt.axvline(
            neg_chi_square_rot_other_mean - neg_chi_square_rot_other_stdev,
            color='blue', linestyle=':', linewidth=1)
    plt.axvline(
            neg_chi_square_rot_other_mean + 2*neg_chi_square_rot_other_stdev,
            color='blue', linestyle=':', linewidth=1)
    plt.axvline(
            neg_chi_square_rot_other_mean - 2*neg_chi_square_rot_other_stdev,
            color='blue', linestyle=':', linewidth=1)
    _, mean_neg_chi_square_rot_other_ = plt.ylim()
    plt.text(neg_chi_square_rot_other_mean + neg_chi_square_rot_other_mean/10,
         mean_neg_chi_square_rot_other_ - mean_neg_chi_square_rot_other_/10,
         'Mean: {:.2f}'.format( neg_chi_square_rot_other_mean))


    plt.hist( neg_chi_square_rot_wall,
             BINS, color='black', density=True, alpha=0.8)
    p = norm.pdf(x,
                 neg_chi_square_rot_wall_mean, neg_chi_square_rot_wall_stdev)
    plt.plot(x, p, 'k--', linewidth=2)
    plt.axvline( neg_chi_square_rot_wall_mean,
                color='black', linestyle='-', linewidth=1.5)
    plt.axvline( neg_chi_square_rot_wall_mean + neg_chi_square_rot_wall_stdev,
                color='black', linestyle=':', linewidth=1)
    plt.axvline( neg_chi_square_rot_wall_mean - neg_chi_square_rot_wall_stdev,
                color='black', linestyle=':', linewidth=1)
    plt.axvline(
            neg_chi_square_rot_wall_mean + 2*neg_chi_square_rot_wall_stdev,
            color='black', linestyle=':', linewidth=1)
    plt.axvline(
            neg_chi_square_rot_wall_mean - 2*neg_chi_square_rot_wall_stdev,
            color='black', linestyle=':', linewidth=1)
    _, mean_wall_neg_chi_square_rot_ = plt.ylim()
    plt.text(neg_chi_square_rot_wall_mean + neg_chi_square_rot_wall_mean/10,
         mean_wall_neg_chi_square_rot_ - mean_wall_neg_chi_square_rot_/10,
         'Mean: {:.2f}'.format( neg_chi_square_rot_wall_mean))


    plt.hist( neg_chi_square_rot_void,
             BINS, color='red', density=True, alpha=0.5)
    p = norm.pdf(x,
                 neg_chi_square_rot_void_mean, neg_chi_square_rot_void_stdev)
    plt.plot(x, p, 'r--', linewidth=2)
    plt.axvline( neg_chi_square_rot_void_mean,
                color='red', linestyle='-', linewidth=1.5)
    plt.axvline( neg_chi_square_rot_void_mean + neg_chi_square_rot_void_stdev,
                color='red', linestyle=':', linewidth=1)
    plt.axvline( neg_chi_square_rot_void_mean - neg_chi_square_rot_void_stdev,
                color='red', linestyle=':', linewidth=1)
    plt.axvline(
            neg_chi_square_rot_void_mean + 2*neg_chi_square_rot_void_stdev,
            color='red', linestyle=':', linewidth=1)
    plt.axvline(
            neg_chi_square_rot_void_mean - 2*neg_chi_square_rot_void_stdev,
            color='red', linestyle=':', linewidth=1)
    _, mean_void_neg_chi_square_rot_ = plt.ylim()
    plt.text(neg_chi_square_rot_void_mean + neg_chi_square_rot_void_mean/10,
         mean_void_neg_chi_square_rot_ - mean_void_neg_chi_square_rot_/10,
         'Mean: {:.2f}'.format( neg_chi_square_rot_void_mean))


    ax = neg_chi_square_rot_hist.add_subplot(111)
    plt.tick_params( axis='both', direction='in')
    ax.yaxis.set_ticks_position('both')
    ax.xaxis.set_ticks_position('both')
    plt.ylabel(r'$Fraction_{galaxy}$')
    plt.xlabel('Chi Square (Goodness of Fit)')

    textstr = '\n'.join((
          r'STDEV: $%.2f$' % ( neg_chi_square_rot_other_stdev, ),
          r'$STDEV_{wall}$: $%.2f$' % ( neg_chi_square_rot_wall_stdev, ),
          r'$STDEV_{void}$: $%.2f$' % ( neg_chi_square_rot_void_stdev, ),
          r'RMS: $%.2f$' % ( neg_chi_square_rot_other_rms, ),
          r'$RMS_{wall}$: $%.2f$' % ( neg_chi_square_rot_wall_rms, ),
          r'$RMS_{void}$: $%.2f$' % ( neg_chi_square_rot_void_rms, )))

    props = dict( boxstyle='round', facecolor='cornsilk', alpha=0.6)

    ax.text(0.72, 0.95, textstr,
            verticalalignment='top', horizontalalignment='left',
            transform=ax.transAxes,
            color='black', fontsize=8, bbox=props)


    plt.savefig( IMAGE_DIR + '\\histograms\\neg_chi_square_hist.png' )
    plt.show()
    plt.close()
    ###########################################################################