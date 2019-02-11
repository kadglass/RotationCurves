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

from scipy.optimize import curve_fit
from scipy.stats import norm
from scipy.stats import ks_2samp

from astropy.table import QTable, Column
from astropy.io import ascii
import astropy.units as u
import astropy.constants as const


GAL_STAT_INDICATOR = "_gal_stat_data.txt"
ROT_CURVE_INDICATOR = "_rot_curve_data.txt"


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


def initialize_master_table( master_table, col_names):
    """Initialize the master_table to contain all of the columns listed in
    col_names but with fill values of -1.

    @param:
        master_table:
            astropy QTable containing identifying information about each galaxy

        col_names:
            an array of strings containing the names of all of the column data
            to be calculated in this script

    @return:
        master_table:
            astropy QTable containing the best fit parameters, the mass
            estimates, environmental classification, and the associated errors
            as well as the identifying information about each galaxy
    """
    for name in col_names:
        try:
            master_table.remove_column( name)

        except KeyError:
            print("USER-GENERATED KEY ERROR: \n" \
                  + "   Column '" + name + "' not found.")

        master_table.add_column( Column( np.full( len( master_table), -1)),
                                name=name)

    return master_table


def pull_matched_data( master_table, ref_table, col_match_names):
    """Match the master_table to a reference table via the

    @param:
        master_table:
            astropy QTable containing the best fit parameters, the mass
            estimates, environmental classification, and the associated errors
            as well as the identifying information about each galaxy

        ref_table:
            astropy QTable containing identifying information on the galaxy
            in question and the vflag parameter associated with the galaxy

        col_match_names:
            array of strings containing the column names of the data to be
            pulled from the ref_table

    @return:
        master_table:
            astropy QTable containing each galaxy's matched parameter in
            addition to the information originally contained in the table
    """
    ###########################################################################
    # Match each entry in 'master_table' to 'ref_table' according to the
    #    'MaNGA_plate' and 'MaNGA_fiberID.' The matched parameter is then
    #    pulled from 'ref_table' and added to the 'master_table.'
    #
    # NOTE: In cases where the 'ref_table' does not contain all of the
    #       'MaNGA_plate' and 'MaNGA_fiberID' match combinations found in the
    #       'master_table,' the return of 'match_catalogs()' will have some
    #       empty return statements because of the lack of matches. The
    #       try-except statement below catches this exception and prints a
    #       message detailing what happened.
    #--------------------------------------------------------------------------
    for i in range( len( master_table)):
        galaxy_matches = match_catalogs( master_table, ref_table, i)

        try:
            for col_name in col_match_names:
                master_table[i][ col_name] = galaxy_matches[0][ col_name]
        except IndexError:
            print("USER-CAUGHT INDEX ERROR: \n",
                  "'galaxy_matches' has zero length (i.e. no matches were " \
                  + "found in the 'ref_table' for " \
                  + master_table[i]['MaNGA_plate'] + "-" \
                  + master_table[i]['MaNGA_fiberID'] + ")")
    ###########################################################################

    return master_table


def match_catalogs( catalog_a, catalog_b, entry):
    """Finds the matches in 'catalog_b' according to the 'entry' data found in
    the data fields of 'catalog_a' specified in the 'match_catagories' array.

    @param:
        catalog_a:
            astropy QTable in which the target data is to be taken from

        catalog_b:
            astropy QTable containing potential matches according to the
            'MaNGA_plate' and 'MaNGA_fiberID' in the selected row's data fields

        entry:
            the row in which the specific 'MaNGA_plate' and 'MaNGA_fiberID'
            are to be used in matching to catalog_b

    @return:
        cat_b_matches:
            astropy QTable containg the matches found in catalog_b
    """
    def match_catalogs_sub( catalog_a, catalog_b, match_criteria, entry):
        """Recursive sub-function of match_catalogs() to match catalog_b
        according to the data in one of catalog_a's entries.
        """
        n = len( match_criteria) - 1

        target_data = catalog_a[ match_criteria[n]][ entry]

        for row in catalog_b:
            if row[ match_criteria[n]] != target_data:
                row['match_flag'] = False

        #######################################################################
        # If there are no more catagories to be matched, return an astropy
        #    QTable containg all the matches found in 'catalog_b.' Otherwise,
        #    recall the function with the matches already found and match
        #    according to the next set of criteria.
        #----------------------------------------------------------------------
        if n == 0:
            return catalog_b
        else:
            return match_catalogs_sub( catalog_a, catalog_b,
                                  match_criteria[0: n], entry)
        #######################################################################


    # ! # ! # ! # ! # ! # ! # ! # ! # ! # ! # ! # ! # ! # ! # ! # ! # ! # ! # !
    # 'match_catalogs()' is a general function to match two astropy Tables
    #    given a set of 'match_criteria.' At the time this function was
    #    written, the 'MaNGA_plate' and 'MaNGA_fiberID' were sufficent to match
    #    the galaxies one-to-one. However, if the user desires to match two
    #    astropy Tables via some other criteria, the user only need to set
    #    'match_criteria' to an n-length array of column name strings.
    #--------------------------------------------------------------------------
    match_criteria = ['MaNGA_plate', 'MaNGA_fiberID']
    # ! # ! # ! # ! # ! # ! # ! # ! # ! # ! # ! # ! # ! # ! # ! # ! # ! # ! # !


    ###########################################################################
    # Initialize a flag variable column to track the matches found and add it
    #    to 'catalog_b.' There exists a try-except statement to delete any
    #    residual 'match_flag' Column. If this column is not found in
    #    catalog_b, then continue the function as normal.
    #--------------------------------------------------------------------------
    match_flag_array = np.full( len( catalog_b), True)
    try:
        catalog_b.remove_column('match_flag')
    except KeyError:
        pass

    catalog_b.add_column( Column( match_flag_array), name='match_flag')
    ###########################################################################


    ###########################################################################
    # Initial call to the recursive 'match_catalogs_sub()' sub-function. The
    #    output of this function, 'cat_b_matches' contains all of the entries
    #    found in 'catalog_b' but with the added 'match_flag' column to
    #    reference the matches found.
    #
    # 'matches' extracts the rows of 'cat_b_matches' where 'match_flag' equals
    #    True.
    #
    # Finally, the 'match_flag' column is then removed from the 'matches'
    #    table before returning.
    #--------------------------------------------------------------------------
    cat_b_matches = match_catalogs_sub( catalog_a, catalog_b, match_criteria,
                                       entry)

    matches = cat_b_matches[ cat_b_matches['match_flag']]
    matches.remove_column('match_flag')
    ###########################################################################

    return matches


def build_vflag_ref_table( CROSS_REF_FILE_NAMES):
    """Compile the environmental classifications and identifying data from the
    various 'void_finder' output files.

    ATTN: Originally, the files in this project were cross referenced to Prof.
    Kelly Douglass' file via the link below. However, upon improvements to
    void_finder, new classifications wer given and the code has been updated.

    CROSS_REF_FILE_NAMES[0] .txt file obtained from:
    <http://www.pas.rochester.edu/~kdouglass/Research/
    kias1033_5_P-MJD-F_MPAJHU_Zdust_stellarMass_BPT_SFR_NSA_correctVflag.txt>

    @param:
        CROSS_REF_FILE_NAMES:
            string representations of the text files uses to extract the vflag
            parameter as obtained from void_finder

    @return:
        vflag_ref_table:
            astropy QTable containing the compiled

    """
    ###########################################################################
    # Initialize the 'vflag_ref_table' to contain the 'MaNGA_plate,'
    #    'MaNGA_fiberID,' and 'vflag' columns.
    #--------------------------------------------------------------------------
    vflag_ref_table = QTable( names=('MaNGA_plate', 'MaNGA_fiberID', 'vflag'),
                             dtype = ( int, int, int))
    ###########################################################################


    ###########################################################################
    # Read in each of the 'void_finder' output files.
    #--------------------------------------------------------------------------
    doug_not_classified = ascii.read( CROSS_REF_FILE_NAMES[1],
                           include_names = ('MaNGA_plate', 'MaNGA_fiberID',
                                            'vflag'), format='ecsv')
    doug_not_found = ascii.read( CROSS_REF_FILE_NAMES[2],
                           include_names = ('MaNGA_plate', 'MaNGA_fiberID',
                                            'vflag'), format='ecsv')
    doug_void_reclass = ascii.read( CROSS_REF_FILE_NAMES[3],
                           include_names = ('MaNGA_plate', 'MaNGA_fiberID',
                                            'vflag'), format='ecsv')
    doug_wall_reclass = ascii.read( CROSS_REF_FILE_NAMES[4],
                           include_names = ('MaNGA_plate', 'MaNGA_fiberID',
                                            'vflag'), format='ecsv')
    ###########################################################################


    ###########################################################################
    # For each row in each of the 'void_finder' output files, append that row
    #    onto the end of the 'vflag_ref_table.'
    #--------------------------------------------------------------------------
    for row in doug_not_classified:
        vflag_ref_table.add_row( row)

    for row in doug_not_found:
        vflag_ref_table.add_row( row)

    for row in doug_void_reclass:
        vflag_ref_table.add_row( row)

    for row in doug_wall_reclass:
        vflag_ref_table.add_row( row)
    ###########################################################################

    return vflag_ref_table


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
    MaNGA_plate_master = []
    MaNGA_fiberID_master = []

    center_flux_master = []
    center_flux_err_master = []
    sMass_processed_master = []

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
        # Extract the data from the gal_stat_file, and append the
        #    'MaNGA_plate,' 'MaNGA_fiberID,' 'center_flux,' and
        #    'center_flux_error,' data to their respective master arrays
        #----------------------------------------------------------------------
        gal_stat_table = ascii.read( gal_stat_file, format='ecsv')
        gal_ID = gal_stat_table['gal_ID'][0]
        print("gal_ID:", gal_ID)

        center_flux = gal_stat_table['center_flux'][0].value
        center_flux_err = gal_stat_table['center_flux_error'][0].value

        MaNGA_plate_master.append( gal_ID[ 0: gal_ID.find('-')] )
        MaNGA_fiberID_master.append( gal_ID[ gal_ID.find('-') + 1:])
        center_flux_master.append( center_flux)
        center_flux_err_master.append( center_flux_err)
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

#        print("depro_radii:", depro_radii)
#        print("rot_vel_avg:", rot_vel_avg)
#        print("rot_vel_avg_err:", rot_vel_avg_err)
#        print("rot_vel_max:", rot_vel_max)
#        print("rot_vel_max_err:", rot_vel_max_err)
#        print("rot_vel_min:", rot_vel_min)
#        print("rot_vel_min_err:", rot_vel_min_err)
        #######################################################################


        #######################################################################
        # Extract the total stellar mass processed for the galaxy from the last
        #    data point in the 'sMass_interior' column for the galaxy and
        #    append that value to the
        #----------------------------------------------------------------------
        sMass_interior = rot_data_table['sMass_interior'].value
        sMass_processed = sMass_interior[ -1]

        sMass_processed_master.append( sMass_processed)

#        print("sMass_processed:", sMass_processed)
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
#        plt.title( gal_ID + " Decomposed Rotation Curves")
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
                print("Rot Parameter Guess:", rot_guess)
                print("POS Rot Parameter Guess:", pos_rot_guess)
                print("NEG Rot Parameter Guess:", neg_rot_guess)
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
#                print("Rot Curve Best Param:", rot_popt)
#                print("Rot Curve Best Param (Positive):", pos_rot_popt)
#                print("Rot Curve Best Param (Negative):", neg_rot_popt)
#                print("Chi^{2}:", chi_square_rot)
#                print("Chi^{2} (Pos):", pos_chi_square_rot)
#                print("Chi^{2} (Neg):", neg_chi_square_rot)
#                print("-----------------------------------------------------")
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
    MaNGA_plate_col = Column( MaNGA_plate_master)
    MaNGA_fiberID_col = Column( MaNGA_fiberID_master)

    center_flux_col = Column( center_flux_master)
    center_flux_err_col = Column( center_flux_err_master)
    sMass_processed_col = Column( sMass_processed_master)

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
    best_fit_param_table = QTable( [MaNGA_plate_col,
                                    MaNGA_fiberID_col,
                                    center_flux_col,
                                    center_flux_err_col,
                                    sMass_processed_col,
                                    v_max_best_col,
                                    v_max_sigma_col,
                                    r_turn_best_col,
                                    r_turn_sigma_col,
                                    alpha_best_col,
                                    alpha_sigma_col,
                                    chi_square_rot_col,
                                    pos_v_max_best_col,
                                    pos_v_max_sigma_col,
                                    pos_r_turn_best_col,
                                    pos_r_turn_sigma_col,
                                    pos_alpha_best_col,
                                    pos_alpha_sigma_col,
                                    pos_chi_square_rot_col,
                                    neg_v_max_best_col,
                                    neg_v_max_sigma_col,
                                    neg_r_turn_best_col,
                                    neg_r_turn_sigma_col,
                                    neg_alpha_best_col,
                                    neg_alpha_sigma_col,
                                    neg_chi_square_rot_col],
                           names = ['MaNGA_plate',
                                    'MaNGA_fiberID',
                                    'center_flux',
                                    'center_flux_error',
                                    'sMass_processed',
                                    'v_max_best',
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


def estimate_dark_matter( master_table,
                         IMAGE_FORMAT, IMAGE_DIR):
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
    # Initialize the arrays that will store the data.
    #--------------------------------------------------------------------------
    MaNGA_plate_master = []
    MaNGA_fiberID_master = []

    gal_mass_master = []
    gal_mass_err_master = []
    theorized_dmMass_master = []
    theorized_dmMass_err_master = []
    sMass_processed_master = []
    dm_to_stellar_mass_ratio_master = []
    dm_to_stellar_mass_ratio_err_master = []
    ###########################################################################


    ###########################################################################
    # Gather the necessary information from the 'master_table'.
    #--------------------------------------------------------------------------
    MaNGA_plate_list = master_table['MaNGA_plate']
    MaNGA_fiberID_list = master_table['MaNGA_fiberID']

    v_max_best_list = master_table['v_max_best']
    v_max_sigma_list = master_table['v_max_sigma']
    r_turn_best_list = master_table['turnover_rad_best']
    alpha_best_list = master_table['alpha_best']
    chi_square_rot_list = master_table['chi_square_rot']
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
      plate, fiberID in zip(
              v_max_best_list, v_max_sigma_list,
              r_turn_best_list, alpha_best_list,
              chi_square_rot_list,
              MaNGA_plate_list, MaNGA_fiberID_list):

        gal_stat_table = ascii.read( str( plate) + "-" + str( fiberID) \
                                    + GAL_STAT_INDICATOR, format='ecsv')
        gal_id = gal_stat_table['gal_ID'][0]

        rot_curve_table = ascii.read( str( plate) + "-" + str( fiberID) \
                                    + ROT_CURVE_INDICATOR, format='ecsv')
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
        #    write to the master file. The quantities are stirpped of their
        #    units at this stage in the algorithm because astropy Column
        #    objects cannot be created with quantities that have dimensions.
        #    The respective dimensions are added back when the Column objects
        #    are added to the astropy QTable.
        #----------------------------------------------------------------------
        MaNGA_plate_master.append( plate)
        MaNGA_fiberID_master.append( fiberID)

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

        plt.savefig( IMAGE_DIR + '/fitted_rotation_curves/' + gal_id +\
                    '_fitted_rotation_curve.' + IMAGE_FORMAT,
                    format=IMAGE_FORMAT)
        plt.show()
        plt.close()
        #######################################################################
    # ~   ~   ~   ~   ~   ~   ~   ~   ~   ~   ~   ~   ~   ~   ~   ~   ~   ~   ~


    ###########################################################################
    # Convert the data arrays into Column objects to add to the rotation
    #    curve data table.
    #--------------------------------------------------------------------------
    MaNGA_plate_col = Column( MaNGA_plate_master)
    MaNGA_fiberID_col = Column( MaNGA_fiberID_master)

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
    mass_estimate_table = QTable( [ MaNGA_plate_col,
                                   MaNGA_fiberID_col,
                                   gal_mass_col * (u.M_sun),
                                   gal_mass_err_col * (u.M_sun),
                                   theorized_dmMass_col * (u.M_sun),
                                   theorized_dmMass_err_col * (u.M_sun),
                                   sMass_col * (u.M_sun),
                                   dmMass_to_sMass_ratio_col,
                                   dmMass_to_sMass_ratio_err_col],
                          names = ['MaNGA_plate',
                                   'MaNGA_fiberID',
                                   'total_mass',
                                   'total_mass_error',
                                   'dmMass',
                                   'dmMass_error',
                                   'sMass',
                                   'dmMass_to_sMass_ratio',
                                   'dmMass_to_sMass_ratio_error'])
    ###########################################################################

    return mass_estimate_table


def plot_mass_ratios( master_table, IMAGE_FORMAT, IMAGE_DIR):
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
    # Import 'vflag' and 'dmMass_to_sMass_ratio' from the 'master_table.'
    #--------------------------------------------------------------------------
    vflag_list = master_table['vflag'].data
    dm_to_stellar_mass_ratio_list = master_table['dmMass_to_sMass_ratio'].data
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
#    ratio_mean = np.mean( dm_to_stellar_mass_ratio_list)
#    ratio_stdev = np.std( dm_to_stellar_mass_ratio_list)
#    ratio_rms = np.sqrt( np.mean( dm_to_stellar_mass_ratio_list**2))

    ratio_wall_mean = np.mean( dm_to_stellar_mass_ratio_wall)
    ratio_wall_stdev = np.std( dm_to_stellar_mass_ratio_wall)
    ratio_wall_rms = np.sqrt( np.mean( dm_to_stellar_mass_ratio_wall**2))

    ratio_void_mean = np.mean( dm_to_stellar_mass_ratio_void)
    ratio_void_stdev = np.std( dm_to_stellar_mass_ratio_void)
    ratio_void_rms = np.sqrt( np.mean( dm_to_stellar_mass_ratio_void**2))
    ###########################################################################


    ###########################################################################
    # Histogram the dark matter to stellar mass ratios as separated by wall
    #    versus void as well as the total distribution.
    #--------------------------------------------------------------------------
    dm_to_stellar_mass_hist = plt.figure()
    xmin, xmax = plt.xlim()
    x = np.linspace(xmin, xmax, 100)

#    plt.hist( dm_to_stellar_mass_ratio_list,
#             BINS, color='green', density=True, alpha=0.9)
#    p = norm.pdf(x, ratio_mean, ratio_stdev)
#    plt.plot(x, p, 'g--', linewidth=2)
#    plt.axvline( ratio_mean, color='green', linestyle='-', linewidth=1.5)
#    plt.axvline( ratio_mean + ratio_stdev,
#                color='green', linestyle=':', linewidth=1)
#    plt.axvline( ratio_mean - ratio_stdev,
#                color='green', linestyle=':', linewidth=1)
#    plt.axvline( ratio_mean + 2*ratio_stdev,
#                color='green', linestyle=':', linewidth=1)
#    plt.axvline( ratio_mean - 2*ratio_stdev,
#                color='green', linestyle=':', linewidth=1)
#    _, mean_ratio_ = plt.ylim()
#    plt.text(ratio_mean + ratio_mean/10,
#         mean_ratio_ - mean_ratio_/10,
#         'Mean: {:.2f}'.format( ratio_mean))

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
#          r'STDEV: $%.2f$' % ( ratio_stdev, ),
          r'$STDEV_{wall}$: $%.2f$' % ( ratio_wall_stdev, ),
          r'$STDEV_{void}$: $%.2f$' % ( ratio_void_stdev, ),
#          r'RMS: $%.2f$' % ( ratio_rms, ),
          r'$RMS_{wall}$: $%.2f$' % ( ratio_wall_rms, ),
          r'$RMS_{void}$: $%.2f$' % ( ratio_void_rms, )))

    props = dict( boxstyle='round', facecolor='cornsilk', alpha=0.6)

    ax.text(0.72, 0.95, textstr,
            verticalalignment='top', horizontalalignment='left',
            transform=ax.transAxes,
            color='black', fontsize=8, bbox=props)

    plt.savefig( IMAGE_DIR + '/histograms/dm_to_stellar_mass_ratio_hist.' \
                + IMAGE_FORMAT,
                format=IMAGE_FORMAT)
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

    plt.savefig( IMAGE_DIR + '/histograms/dm_to_stellar_mass_ratio_hist.' \
                + IMAGE_FORMAT,
                format=IMAGE_FORMAT)
    plt.show()
    plt.close()
    ###########################################################################


def analyze_rot_curve_discrep( master_table, IMAGE_FORMAT, IMAGE_DIR):
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

        elif vflag == 1:
            v_max_diff_void.append( v_max_diff)
            v_max_diff_void_error.append( v_max_diff_err)
            inclin_angle_void.append( inc_angle)
            mass_ratio_void.append( ratio)
            mass_ratio_void_error.append( ratio_err)

        elif vflag == 2 or vflag == -9:
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

    plt.savefig( IMAGE_DIR + '/histograms/v_max_diff_hist.' + IMAGE_FORMAT,
                format=IMAGE_FORMAT)
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

    plt.savefig( IMAGE_DIR + '/v_max_vs_inclination.' + IMAGE_FORMAT,
                format=IMAGE_FORMAT)
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

    plt.savefig( IMAGE_DIR + '/v_max_vs_mass_ratio.' + IMAGE_FORMAT,
                format=IMAGE_FORMAT)
    plt.show()
    plt.close()
    ###########################################################################


def analyze_chi_square( master_table, IMAGE_FORMAT, IMAGE_DIR):
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
    vflag_list = master_table['vflag']
    avg_chi_square_rot_master = master_table['chi_square_rot']
    pos_chi_square_rot_master = master_table['pos_chi_square_rot']
    neg_chi_square_rot_master = master_table['neg_chi_square_rot']
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

        elif vflag == 1:
            avg_chi_square_rot_void.append( chi_square)
            pos_chi_square_rot_void.append( pos_chi_square)
            neg_chi_square_rot_void.append( neg_chi_square)

        elif vflag == 2 or vflag == -9:
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


    plt.savefig( IMAGE_DIR + '/histograms/avg_chi_square_hist.' + IMAGE_FORMAT,
                format=IMAGE_FORMAT)
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


    plt.savefig( IMAGE_DIR + '/histograms/pos_chi_square_hist.' + IMAGE_FORMAT,
                format=IMAGE_FORMAT)
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


    plt.savefig( IMAGE_DIR + '/histograms/neg_chi_square_hist.' + IMAGE_FORMAT,
                format=IMAGE_FORMAT)
    plt.show()
    plt.close()
    ###########################################################################