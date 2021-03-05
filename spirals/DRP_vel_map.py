'''
Analyzes mainly face-on galaxies to determine the kinematics of the galaxy.

To download the MaNGA .fits files used to calculate the kinematics for these 
galaxies, see the instructions for each data release via the following links:

http://www.sdss.org/dr14/manga/manga-data/data-access/
http://www.sdss.org/dr15/manga/manga-data/data-access/
http://www.sdss.org/dr16/manga/manga-data/data-access/
'''


################################################################################
# IMPORT MODULES
#-------------------------------------------------------------------------------
import numpy as np
import numpy.ma as ma

import matplotlib.pyplot as plt

import warnings
warnings.simplefilter('ignore', np.RankWarning)

import os.path

from astropy.io import fits
from astropy.table import Table, Column

from DRP_vel_map_functions import find_vel_map, mass_newton

from DRP_rotation_curve_plottingFunctions import plot_rband_image, \
                                                 plot_Ha_vel

from DRP_vel_map_plottingFunctions import plot_rot_curve, \
                                          plot_diagnostic_panel, \
                                          plot_residual, \
                                          plot_chi2
################################################################################






################################################################################
################################################################################
################################################################################


def fit_vel_map(Ha_vel, 
                Ha_vel_ivar, 
                Ha_vel_mask, 
                r_band, 
                r_band_ivar,
                axis_ratio, 
                phi_EofN_deg, 
                z, 
                gal_ID,
                fit_function, 
                IMAGE_DIR=None, 
                IMAGE_FORMAT='eps', 
                num_masked_gal=0):
    '''
    Determine the values of the parameters that best reproduce the galaxy's 
    H-alpha velocity map.


    Parameters:
    ===========

    Ha_vel : numpy array of shape (n,n)
        H-alpha velocity field data

    Ha_vel_ivar : numpy array of shape (n,n)
        Inverse variance in the H-alpha velocity field data

    Ha_vel_mask : numpy array of shape (n,n)
        Bitmask for the H-alpha velocity map

    r_band : numpy array of shape (n,n)
        r-band flux data

    r_band_ivar : numpy array of shape (n,n)
        Inverse variance in the r-band flux data

    axis_ratio : float
        Ratio of the galaxy's minor axis to major axis as obtained via an 
        elliptical sersic fit of the galaxy

    phi_EofN_deg : float
        Angle (east of north) of rotation in the 2-D, observational plane.  
        Units are degrees 

        NOTE: east is 'left' per astronomy convention

    z : float
        Galaxy redshift as calculated by the shift in H-alpha flux

    gal_ID : string
        [PLATE]-[IFU]

    fit_function : string
        Determines which function to use for the velocity.  Options are 'BB' and 
        'tanh'.

    IMAGE_DIR : string
        File path to which pictures of the fitted rotation curves are saved.  
        Default value is None (do not save images).

    IMAGE_FORMAT : string
        Saved image file format.  Default format is eps.

    num_masked_gal : float
        Cumulative number of completely masked galaxies seen so far.  Default 
        value is 0.


    Returns:
    ========

    param_outputs : dictionary
        Contains the best-fit parameter values:
          - v_sys [km/s] : systemic velocity of the galaxy
          - ba : axis ratio of galaxy
          - x0 : x-coordinate of central pixel of galaxy rotation
          - y0 : y-coordinate of central pixel of galaxy rotation
          - phi [deg] : orientation angle of galaxy's major axis (east of north)
          - velocity function parameters (depends on fit_function value)

    num_masked_gal : float
        Cumulative number of completely masked galaxies
    '''

    ############################################################################
    # Apply mask to all data arrays
    #---------------------------------------------------------------------------
    num_masked_spaxels = np.sum(Ha_vel_mask) - np.sum(r_band == 0)
    frac_masked_spaxels = num_masked_spaxels/np.sum(r_band != 0)

    mr_band = ma.array( r_band, mask=Ha_vel_mask)
    mr_band_ivar = ma.array( r_band_ivar, mask=Ha_vel_mask)

    mHa_vel = ma.array( Ha_vel, mask=Ha_vel_mask)
    mHa_vel_ivar = ma.array( Ha_vel_ivar, mask=Ha_vel_mask)
    '''
    #---------------------------------------------------------------------------
    # Show the mask.  Yellow points represent masked data points.
    #---------------------------------------------------------------------------
    plt.figure(1)
    plt.imshow( Ha_vel_mask)
    plt.show()
    plt.close()
    '''
    ############################################################################
    

    ############################################################################
    # DIAGNOSTICS:
    #---------------------------------------------------------------------------
    # Plot r-band image
    #---------------------------------------------------------------------------
    plot_rband_image(r_band, 
                     gal_ID, 
                     IMAGE_DIR=IMAGE_DIR, 
                     IMAGE_FORMAT=IMAGE_FORMAT)
    '''
    if IMAGE_DIR is None:
        plt.show()
    '''
    #---------------------------------------------------------------------------
    # Plot H-alpha velocity field before systemic redshift subtraction.  Galaxy 
    # velocities vary from file to file, so vmin and vmax will have to be 
    # manually adjusted for each galaxy before reshift subtraction.
    #---------------------------------------------------------------------------
    plot_Ha_vel(Ha_vel, 
                gal_ID, 
                IMAGE_DIR=IMAGE_DIR, 
                FOLDER_NAME='/unmasked_Ha_vel/', 
                IMAGE_FORMAT=IMAGE_FORMAT, 
                FILENAME_SUFFIX='_Ha_vel_raw.')
    '''
    if IMAGE_DIR is None:
        plt.show()
    '''
    ############################################################################


    ############################################################################
    # Determine initial guess for the optical center via the max luminosity in 
    # the r-band.
    #---------------------------------------------------------------------------
    center_guess = np.unravel_index(ma.argmax(mr_band), mr_band.shape)
    
    i_center_guess = center_guess[0]
    j_center_guess = center_guess[1]
    ############################################################################


    ############################################################################
    # Set the initial guess for the systemic velocity to be equal to the 
    # velocity at the initially-guessed center spaxel.
    #---------------------------------------------------------------------------
    sys_vel_guess = mHa_vel[center_guess]
    ############################################################################


    ############################################################################
    # Set the initial guess for the inclination angle equal that given by the 
    # measured axis ratio.
    #---------------------------------------------------------------------------
    inclination_angle_guess = np.arccos(axis_ratio)
    ############################################################################


    ############################################################################
    # Adjust the domain of the rotation angle (phi) from 0-pi to 0-2pi, where it 
    # always points through the positive velocity semi-major axis.
    #---------------------------------------------------------------------------
    # Find spaxel along semi-major axis
    delta_x = int(j_center_guess*0.4)
    delta_y = int(delta_x/np.tan(phi_EofN_deg*np.pi/180.))
    semi_major_axis_spaxel = tuple(np.subtract(center_guess, (-delta_y, delta_x)))

    # Check value along semi-major axis
    if mHa_vel[semi_major_axis_spaxel] < 0:
        phi_guess = phi_EofN_deg*np.pi/180. + np.pi
    else:
        phi_guess = phi_EofN_deg*np.pi/180.
    ############################################################################


    ############################################################################
    # Find the global max and global min of 'masked_Ha_vel' to use in graphical
    # analysis.
    #
    # NOTE: If the entire data array is masked, 'global_max' and 'global_min'
    #       cannot be calculated. It has been found that if the
    #       'inclination_angle' is 0 degrees, the entire 'Ha_vel' array is
    #       masked. An if-statement tests this case, and sets 'unmasked_data'
    #       to False if there is no max/min in the array.
    #---------------------------------------------------------------------------
    global_max = np.max(mHa_vel)
    global_min = np.min(mHa_vel)

    unmasked_data = True

    if np.isnan(global_max):
        unmasked_data = False
        global_max = 0.1
        global_min = -0.1
    ############################################################################


    ############################################################################
    # If 'unmasked_data' was set to False because all of the 'Ha_vel' data is
    # masked after correcting for the angle of inclination, set all of the 
    # best-fit values to be nan.
    #---------------------------------------------------------------------------
    if not unmasked_data:
        param_outputs = {'v_sys': np.nan,  'v_sys_err': np.nan,
                         'ba': np.nan,     'ba_err': np.nan,
                         'x0': np.nan,     'x0_err': np.nan,
                         'y0': np.nan,     'y0_err': np.nan,
                         'phi': np.nan,    'phi_err': np.nan, 
                         'r_turn': np.nan, 'r_turn_err': np.nan, 
                         'v_max': np.nan,  'v_max_err': np.nan}

        if fit_function == 'BB':
            param_outputs['alpha'] = np.nan
            param_outputs['alpha_err'] = np.nan

        num_masked_gal += 1

        print("ALL DATA POINTS FOR THE GALAXY ARE MASKED!!!")
    ############################################################################


    ############################################################################
    # If there is unmasked data in the data array, fit the velocity map.
    #---------------------------------------------------------------------------
    else:
        print('Fitting velocity map')
        param_outputs, best_fit_map, scale = find_vel_map(mHa_vel, 
                                                          mHa_vel_ivar, 
                                                          z, 
                                                          i_center_guess, 
                                                          j_center_guess,
                                                          sys_vel_guess, 
                                                          inclination_angle_guess, 
                                                          phi_guess, 
                                                          fit_function)

        if param_outputs is not None:
            ####################################################################
            # Mask best-fit map
            #-------------------------------------------------------------------
            mbest_fit_map = ma.array(best_fit_map, mask=mHa_vel.mask)
            ####################################################################


            ####################################################################
            # Plot the best-fit H-alpha velocity field
            #-------------------------------------------------------------------
            plot_Ha_vel(mbest_fit_map, 
                        gal_ID, 
                        IMAGE_DIR=IMAGE_DIR, 
                        FOLDER_NAME='/fitted_velocity_fields/', 
                        FILENAME_SUFFIX='_fitted_vel_field.', 
                        IMAGE_FORMAT=IMAGE_FORMAT)

            if IMAGE_DIR is None:
                plt.show()
            ####################################################################


            ####################################################################
            # Plot the residual velocity map between the best-fit and the data
            #-------------------------------------------------------------------
            plot_residual(mbest_fit_map, 
                          mHa_vel,
                          gal_ID, 
                          IMAGE_DIR=IMAGE_DIR, 
                          FOLDER_NAME='/residuals/', 
                          FILENAME_SUFFIX='_residual.', 
                          IMAGE_FORMAT=IMAGE_FORMAT)

            if IMAGE_DIR is None:
                plt.show()
            ####################################################################


            ####################################################################
            # Plot the chi2 map of the best-fit model
            #-------------------------------------------------------------------
            plot_chi2(mbest_fit_map, 
                      mHa_vel,
                      mHa_vel_ivar, 
                      gal_ID, 
                      IMAGE_DIR=IMAGE_DIR, 
                      FOLDER_NAME='/chi2/', 
                      FILENAME_SUFFIX='_chi2.', 
                      IMAGE_FORMAT=IMAGE_FORMAT)

            if IMAGE_DIR is None:
                plt.show()
            ####################################################################


            ####################################################################
            # Plot H-alpha velocity field with redshift subtracted.
            #-------------------------------------------------------------------
            plot_Ha_vel(mHa_vel - param_outputs['v_sys'], 
                        gal_ID, 
                        IMAGE_DIR=IMAGE_DIR, 
                        FOLDER_NAME='/masked_Ha_vel/', 
                        FILENAME_SUFFIX='_Ha_vel_field.', 
                        IMAGE_FORMAT=IMAGE_FORMAT)

            if IMAGE_DIR is None:
                plt.show()
            ####################################################################


            ####################################################################
            # Plot 1D rotation curve with best-fit
            #-------------------------------------------------------------------
            plot_rot_curve(mHa_vel, 
                           mHa_vel_ivar,
                           param_outputs, 
                           scale,
                           gal_ID, 
                           fit_function,
                           IMAGE_DIR=IMAGE_DIR, 
                           IMAGE_FORMAT=IMAGE_FORMAT)

            if IMAGE_DIR is None:
                plt.show()
            ####################################################################

            '''
            ####################################################################
            # Plot cumulative mass as a function of deprojected radius.
            #-------------------------------------------------------------------
            plot_mass_curve(mHa_vel, 
                            mHa_vel_ivar, 
                            param_outputs,
                            gal_ID,
                            IMAGE_DIR=IMAGE_DIR, 
                            IMAGE_FORMAT=IMAGE_FORMAT)

            if IMAGE_DIR is None:
                plt.show()
            ####################################################################
            '''

            ####################################################################
            # Plot a two by two paneled image containging 
            #   - the r-band image
            #   - the masked H-alpha velocity array, 
            #   - the best-fit H-alpha velocity array, 
            #   - the best-fit rotation curve along with the mass rotation curves
            #-------------------------------------------------------------------
            plot_diagnostic_panel(r_band, 
                                  mHa_vel, 
                                  mHa_vel_ivar, 
                                  mbest_fit_map,
                                  param_outputs,
                                  scale, 
                                  gal_ID, 
                                  fit_function,
                                  IMAGE_DIR=IMAGE_DIR, 
                                  IMAGE_FORMAT=IMAGE_FORMAT)

            if IMAGE_DIR is None:
                plt.show()
            ####################################################################
    ############################################################################


    return param_outputs, num_masked_gal




################################################################################
################################################################################
################################################################################




def estimate_total_mass(v_max, v_max_err, R90, z):
    '''
    Estimate the total mass interior to each given radius from the fitted 
    v_max parameter.


    Parameters:
    ===========

    v_max : float
        Best-fit maximum velocity for the galaxy [km/s]

    v_max_err : float
        Uncertainty in the best-fit maximum velocity [km/s]

    R90 : float
        90% light radius [arcsec]

    z : float
        Galaxy redshift


    Returns:
    ========

    M90 : float
        Logarithm of the total mass corresponding to the 90% light radius 
        [log(Msun)]
    '''


    ############################################################################
    # Calculate masses
    #---------------------------------------------------------------------------
    M90 = mass_newton(v_max, v_max_err, R90, z)
    ############################################################################

    return {'M90':np.log10(M90)}









