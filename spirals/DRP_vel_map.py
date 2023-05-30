'''
Analyzes disk galaxies to determine the kinematics of the galaxy.

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

from dark_matter_mass_v1 import rot_fit_BB, rot_fit_tanh

from DRP_vel_map_functions import find_vel_map, \
                                  mass_newton, \
                                  find_phi, \
                                  build_map_mask

from DRP_rotation_curve_plottingFunctions import plot_rband_image, \
                                                 plot_vel

from DRP_vel_map_plottingFunctions import plot_rot_curve, \
                                          plot_diagnostic_panel, \
                                          plot_residual, \
                                          plot_residual_norm, \
                                          plot_chi2, \
                                          plot_Ha_sigma
################################################################################




################################################################################
# Constants
#-------------------------------------------------------------------------------
H_0 = 100      # Hubble's Constant in units of h km/s/Mpc
c = 299792.458 # Speed of light in units of km/s

q0 = 0.2 # Nominal disk thickness
################################################################################






################################################################################
################################################################################
################################################################################


def fit_vel_map(vel, 
                vel_ivar, 
                vel_mask, 
                Ha_sigma, 
                Ha_sigma_ivar, 
                Ha_sigma_mask, 
                Ha_flux, 
                Ha_flux_ivar, 
                Ha_flux_mask, 
                r_band, 
                r_band_ivar,
                axis_ratio, 
                phi_EofN_deg, 
                z, 
                gal_ID,
                fit_function, 
                V_type='Ha', 
                IMAGE_DIR=None, 
                IMAGE_FORMAT='eps', 
                num_masked_gal=0):
    '''
    Determine the values of the parameters that best reproduce the galaxy's 
    H-alpha velocity map.


    Parameters:
    ===========

    vel : numpy array of shape (n,n)
        Velocity map to be fit [km/s]

    vel_ivar : numpy array of shape (n,n)
        Inverse variance in the velocity map

    vel_mask : numpy array of shape (n,n)
        Bitmask for the velocity map

    Ha_sigma : numpy array of shape (n,n)
        H-alpha line width data

    Ha_sigma_ivar : numpy array of shape (n,n)
        Inverse variance in the H-alpha line width data

    Ha_sigma_mask : numpy array of shape (n,n)
        Bitmask for the H-alpha line width map

    Ha_flux : numpy array of shape (n,n)
        H-alpha flux field data

    Ha_flux_ivar : numpy array of shape (n,n)
        Inverse variance of the H-alpha flux field data

    Ha_flux_mask : numpy array of shape (n,n)
        Bitmask for the H-alpha flux map

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

    V_type : string
        Type of velocity map being fit.  Used for plot titles, file names, etc.

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
    num_masked_spaxels = np.sum(vel_mask) - np.sum(r_band == 0)
    frac_masked_spaxels = num_masked_spaxels/np.sum(r_band != 0)

    mr_band = ma.array( r_band, mask=vel_mask)
    mr_band_ivar = ma.array( r_band_ivar, mask=vel_mask)

    mvel = ma.array( vel, mask=vel_mask)
    mvel_ivar = ma.array( vel_ivar, mask=vel_mask)

    mHa_flux = ma.array( Ha_flux, mask=Ha_flux_mask + vel_mask)
    mHa_flux_ivar = ma.array( Ha_flux_ivar, mask=Ha_flux_mask + vel_mask)

    mHa_sigma = ma.array( Ha_sigma, mask=vel_mask + Ha_sigma_mask)
    mHa_sigma_ivar = ma.array( Ha_sigma_ivar, mask=vel_mask + Ha_sigma_mask)
    '''
    #---------------------------------------------------------------------------
    # Show the mask.  Yellow points represent masked data points.
    #---------------------------------------------------------------------------
    plt.figure()
    plt.imshow(vel_mask)
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
    # Plot velocity field before systemic redshift subtraction.  Galaxy 
    # velocities vary from file to file, so vmin and vmax will have to be 
    # manually adjusted for each galaxy before reshift subtraction.
    #---------------------------------------------------------------------------
    plot_vel(vel, 
             gal_ID, 
             V_type=V_type, 
             IMAGE_DIR=IMAGE_DIR, 
             FOLDER_NAME='/unmasked_' + V_type + '_vel/', 
             IMAGE_FORMAT=IMAGE_FORMAT, 
             FILENAME_SUFFIX='_' + V_type + '_vel_raw.')
    '''
    if IMAGE_DIR is None:
        plt.show()
    '''
    #---------------------------------------------------------------------------
    # Plot H-alpha line width
    #---------------------------------------------------------------------------
    plot_Ha_sigma(mHa_sigma, 
                  gal_ID, 
                  IMAGE_DIR=IMAGE_DIR, 
                  FOLDER_NAME='/Ha_sigma/', 
                  IMAGE_FORMAT=IMAGE_FORMAT, 
                  FILENAME_SUFFIX='_Ha_sigma.')
    ############################################################################


    ############################################################################
    # Determine initial guess for the optical center via the max luminosity in 
    # the r-band.
    #---------------------------------------------------------------------------
    center_guess = np.unravel_index(ma.argmax(mr_band), mr_band.shape)

    if gal_ID == '8613-12701':
        center_guess = (40,35)
    elif gal_ID == '8134-3701':
        center_guess = (22,22)
    elif gal_ID in ['8252-6103']:
        center_guess = (27,27)
    elif gal_ID in ['8447-9102', '9037-9102']:
        center_guess = (32,32)
    elif gal_ID in ['8249-12702', '8940-12701', '8941-12703', '8950-12705', 
                    '9488-12702', '8333-12702', '9181-12701']:
        center_guess = (37,37)
    elif gal_ID in ['7958-12703']:
        center_guess = (39,39)
    elif gal_ID in ['8588-12702']:
        center_guess = (40,40)

    #print(center_guess)
    
    i_center_guess = center_guess[0]
    j_center_guess = center_guess[1]
    ############################################################################


    ############################################################################
    # Set the initial guess for the systemic velocity to be equal to the 
    # velocity at the initially-guessed center spaxel.
    #---------------------------------------------------------------------------
    sys_vel_guess = mvel[center_guess]

    if (sys_vel_guess is ma.masked) or (gal_ID == '8940-12701'):
        sys_vel_guess = 0.

    #print(sys_vel_guess)
    ############################################################################


    ############################################################################
    # Set the initial guess for the inclination angle equal that given by the 
    # measured axis ratio.
    #---------------------------------------------------------------------------
    cosi2 = (axis_ratio**2 - q0**2)/(1 - q0**2)

    if cosi2 < 0:
        cosi2 = 0

    inclination_angle_guess = np.arccos(np.sqrt(cosi2))
    #inclination_angle_guess = np.arccos(axis_ratio)
    #print(axis_ratio, inclination_angle_guess)
    ############################################################################


    ############################################################################
    # Adjust the domain of the rotation angle (phi) from 0-pi to 0-2pi, where it 
    # always points through the positive velocity semi-major axis.
    #---------------------------------------------------------------------------
    phi_guess = find_phi(center_guess, phi_EofN_deg, mvel)


    if gal_ID in ['8134-6102', '9864-6102']:
        phi_guess += 0.25*np.pi

    elif gal_ID in ['8932-12704', '8252-6103', '9500-6102']:
        phi_guess -= 0.25*np.pi

    elif gal_ID in ['8613-12703', '8726-1901', '8615-1901', '8325-9102', 
                    '8274-6101', '9027-12705', '9868-12702', '8135-1901', 
                    '7815-1901', '8568-1901', '8989-1902', '8458-3701', 
                    '9000-1901', '9037-3701', '8456-6101']:
        phi_guess += 0.5*np.pi

    elif gal_ID in ['9864-3702', '8601-1902']:
        phi_guess -= 0.5*np.pi

    elif gal_ID in ['9502-12702']:
        phi_guess += 0.75*np.pi

    elif gal_ID in ['9029-12705', '8137-3701', '8618-3704', '8323-12701', 
                    '8942-3703', '8333-12701', '8615-6103', '9486-3704', 
                    '8937-1902', '9095-3704', '8466-1902', '9508-3702', 
                    '8727-3703', '8341-12704', '8655-6103', '8335-6101']:
        phi_guess += np.pi

    elif gal_ID in ['8082-1901', '8078-3703', '8551-1902', '9039-3703', 
                    '8624-1902', '8948-12702', '8443-6102', '8259-1901']:
        phi_guess += 1.5*np.pi

    elif gal_ID in ['8241-12705', '8326-6102', '9505-6103', '8256-6104']:
        phi_guess += 1.75*np.pi

    elif gal_ID in ['8655-1902', '7960-3701', '9864-9101', '8588-3703']:
        phi_guess = phi_EofN_deg*np.pi/180.


    phi_guess = phi_guess%(2*np.pi)

    #print(phi_EofN_deg, phi_guess*180/np.pi)
    ############################################################################


    ############################################################################
    # Find the global max and global min of the masked velocity map to use in 
    # graphical analysis.
    #
    # NOTE: If the entire array is masked, 'global_max' and 'global_min'
    #       cannot be calculated. It has been found that if the
    #       'inclination_angle' is 0 degrees, the entire array is masked. An 
    #       if-statement tests this case, and sets 'unmasked_data' to False if 
    #       there is no max/min in the array.
    #---------------------------------------------------------------------------
    global_max = ma.max(mvel)
    global_min = ma.min(mvel)

    unmasked_data = True

    if np.isnan(global_max) or (global_max is ma.masked):
        unmasked_data = False
        global_max = 0.1
        global_min = -0.1
    ############################################################################


    ############################################################################
    # If 'unmasked_data' was set to False because the entire velocity map is
    # masked after correcting for the angle of inclination, set the output to 
    # None.
    #---------------------------------------------------------------------------
    if not unmasked_data:
        param_outputs = None

        fit_flag = np.nan

        num_masked_gal += 1

        print("ALL DATA POINTS FOR THE GALAXY ARE MASKED!!!", flush=True)
    ############################################################################


    ############################################################################
    # If there is unmasked data in the data array, fit the velocity map.
    #---------------------------------------------------------------------------
    else:
        print(gal_ID, 'fitting velocity map', flush=True)
        param_outputs, best_fit_map, scale, fit_flag = find_vel_map(gal_ID, 
                                                                    mvel, 
                                                                    mvel_ivar, 
                                                                    mHa_sigma,
                                                                    mHa_flux, 
                                                                    mHa_flux_ivar,
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
            best_mask = build_map_mask(gal_ID, 
                                       fit_flag, 
                                       mvel, 
                                       mHa_flux, 
                                       mHa_flux_ivar, 
                                       mHa_sigma)

            mbest_fit_map = ma.array(best_fit_map, mask=best_mask)
            ####################################################################


            ####################################################################
            # Plot the best-fit velocity field
            #-------------------------------------------------------------------
            plot_vel(mbest_fit_map, 
                     gal_ID, 
                     V_type=V_type, 
                     model=True,
                     IMAGE_DIR=IMAGE_DIR, 
                     FOLDER_NAME='/fitted_velocity_fields/', 
                     FILENAME_SUFFIX='_fitted_' + V_type + '_vel_field.', 
                     IMAGE_FORMAT=IMAGE_FORMAT)

            if IMAGE_DIR is None:
                plt.show()
            ####################################################################


            ####################################################################
            # Plot the residual velocity map between the best-fit and the data
            #-------------------------------------------------------------------
            plot_residual(mbest_fit_map, 
                          mvel,
                          gal_ID, 
                          IMAGE_DIR=IMAGE_DIR, 
                          FOLDER_NAME='/residuals/', 
                          FILENAME_SUFFIX='_' + V_type + '_residual.', 
                          IMAGE_FORMAT=IMAGE_FORMAT)

            if IMAGE_DIR is None:
                plt.show()
            ####################################################################


            ####################################################################
            # Plot the normalized residual velocity map between the best-fit and 
            # the data
            #-------------------------------------------------------------------
            plot_residual_norm(mbest_fit_map, 
                               mvel,
                               gal_ID, 
                               IMAGE_DIR=IMAGE_DIR, 
                               FOLDER_NAME='/residuals_norm/', 
                               FILENAME_SUFFIX='_' + V_type + '_residual_norm.', 
                               IMAGE_FORMAT=IMAGE_FORMAT)

            if IMAGE_DIR is None:
                plt.show()
            ####################################################################


            ####################################################################
            # Plot the chi2 map of the best-fit model
            #-------------------------------------------------------------------
            plot_chi2(mbest_fit_map, 
                      mvel,
                      mvel_ivar, 
                      gal_ID, 
                      IMAGE_DIR=IMAGE_DIR, 
                      FOLDER_NAME='/chi2/', 
                      FILENAME_SUFFIX='_' + V_type + '_chi2.', 
                      IMAGE_FORMAT=IMAGE_FORMAT)

            if IMAGE_DIR is None:
                plt.show()
            ####################################################################


            ####################################################################
            # Plot velocity field with the systemic velocity subtracted.
            #-------------------------------------------------------------------
            plot_vel(ma.array(mvel, mask=best_mask) - param_outputs['v_sys'], 
                     gal_ID, 
                     V_type=V_type, 
                     IMAGE_DIR=IMAGE_DIR, 
                     FOLDER_NAME='/masked_' + V_type + '_vel/', 
                     FILENAME_SUFFIX='_' + V_type + '_vel_field.', 
                     IMAGE_FORMAT=IMAGE_FORMAT)

            if IMAGE_DIR is None:
                plt.show()
            ####################################################################


            ####################################################################
            # Plot 1D rotation curve with best-fit
            #-------------------------------------------------------------------
            Rmax = plot_rot_curve(ma.array(mvel, mask=best_mask), 
                                  ma.array(mvel_ivar, mask=best_mask),
                                  param_outputs, 
                                  scale,
                                  gal_ID, 
                                  fit_function,
                                  IMAGE_DIR=IMAGE_DIR, 
                                  IMAGE_FORMAT=IMAGE_FORMAT, 
                                  FILENAME_SUFFIX='_' + V_type)

            param_outputs['Rmax'] = Rmax

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
            #   - the masked velocity array, 
            #   - the best-fit velocity array, 
            #   - the best-fit rotation curve
            #-------------------------------------------------------------------
            plot_diagnostic_panel(r_band, 
                                  ma.array(mvel, mask=best_mask), 
                                  ma.array(mvel_ivar, mask=best_mask), 
                                  mbest_fit_map,
                                  param_outputs,
                                  scale, 
                                  gal_ID, 
                                  fit_function,
                                  V_type=V_type, 
                                  IMAGE_DIR=IMAGE_DIR, 
                                  IMAGE_FORMAT=IMAGE_FORMAT)

            if IMAGE_DIR is None:
                plt.show()
            ####################################################################
    ############################################################################


    return param_outputs, num_masked_gal, fit_flag




################################################################################
################################################################################
################################################################################




def estimate_total_mass(params, r, z, fit_function, gal_ID):
    '''
    Estimate the total mass interior to each given radius from the parameters.


    Parameters:
    ===========

    params : list
        Best-fit values for the given velocity function, in the order of input 
        to the velocity function.

    r : float
        Radius at which to calculate the total mass [arcsec]

    z : float
        Galaxy redshift

    fit_function : string
        Represents which fit function to use to calculate the velocity at the 
        given r.

    gal_ID : string
        <plate>-<IFU>


    Returns:
    ========

    M, Merr : float
        Logarithm of the total mass and error corresponding to the mass within r
        [log(Msun)]
    '''


    ############################################################################
    # Convert r from arcsec to kpc
    #---------------------------------------------------------------------------
    dist_to_galaxy_Mpc = c*z/H_0
    dist_to_galaxy_kpc = dist_to_galaxy_Mpc*1000

    r_kpc = dist_to_galaxy_kpc*np.tan(r*(1./60)*(1./60)*(np.pi/180))
    ############################################################################


    ############################################################################
    # Calculate velocity at given radius
    #---------------------------------------------------------------------------
    hess = np.load('DRP_map_Hessians/' + gal_ID + '_Hessian.npy')

    N_samples = 10000

    v_samples = np.zeros(N_samples, dtype=float)

    if fit_function == 'BB':
        v = rot_fit_BB(r_kpc, params)

        try:

            param_samples = np.random.multivariate_normal(mean=params, 
                                                          cov=0.5*hess[-3:,-3:], 
                                                          size=N_samples)

            for i in range(N_samples):
                
                if np.all(param_samples[i] > 0):
                    v_samples[i] = rot_fit_BB(r_kpc, param_samples[i])

            v_err = np.std(v_samples[np.isfinite(v_samples)])

        except np.linalg.LinAlgError:
            v_err = np.nan
    
    elif fit_function == 'tanh':
        v = rot_fit_tanh(r_kpc, params)

        param_samples = np.random.multivariate_normal(mean=params, 
                                                      cov=0.5*hess[-2:,-2:], 
                                                      size=N_samples)

        for i in range(N_samples):

            if np.all(param_samples[i] > 0):
                v_samples[i] = rot_fit_tanh(r_kpc, param_samples[i])

        v_err = np.std(v_samples[np.isfinite(v_samples)])
    
    else:
        print('Fit function not known.  Please update estimate_total_mass function.')
    ############################################################################


    ############################################################################
    # Calculate masses
    #---------------------------------------------------------------------------
    M, M_err = mass_newton(v, v_err, r_kpc)
    ############################################################################

    return {'M':np.log10(M), 'M_err':np.log10(M_err)}










