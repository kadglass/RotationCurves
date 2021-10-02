
import numpy as np
import numpy.ma as ma

from astropy.table import Table

from scipy.optimize import minimize, curve_fit

import matplotlib.pyplot as plt

from disk_mass_functions import find_mass_curve, chi2_mass

from Pipe3D_rotation_curve_plottingFunctions import plot_sMass_image

from disk_mass_plotting_functions import plot_fitted_disk_rot_curve

import sys
sys.path.insert(1, '/Users/kellydouglass/Documents/Research/Rotation_curves/Yifan_Zhang/RotationCurve/2D_RC/main/')
from rotation_curve_functions import disk_vel




################################################################################
################################################################################
################################################################################

def calc_mass_curve(sMass_density, 
                    r_band,
                    map_mask, 
                    x0, 
                    y0,
                    ba, 
                    phi, 
                    z, 
                    gal_ID, 
                    IMAGE_DIR=None, 
                    IMAGE_FORMAT='eps'):
    '''
    Extract the mass as a function of radius from the stellar mass density map, 
    and convert this to a rotation curve (velocity as a function of radius).


    PARAMETERS
    ==========

    sMass_density : numpy array of shape (n,n)
        Stellar mass density map in units of log(Msun/spaxel^2)

    r_band : numpy array of shape (n,n)
        r-band flux data

    map_mask : numpy array of shape (n,n)
        Bitmask for the stellar mass density map

    x0, y0 : float or None
        Kinematic center of the galaxy.  If None, then need to find and use the 
        brighest location on the galaxy as the center.

    ba : float
        Ratio of the galaxy's minor axis to major axis

    phi : float
        Angle (east of north) of rotation in the 2-D, observational plane

        NOTE: east is 'left' per astronomy convention

    z : float
        Galaxy redshift

    gal_ID : string
        [PLATE]-[IFU]

    IMAGE_DIR : string
        File path to which various produced images are saved.  
        Default value is None (do not save images).

    IMAGE_FORMAT : string
        Saved image file format.  Default format is eps.


    RETURNS
    =======

    mass_vel_table : astropy table
        Data table containing M(r) and v(r) for the stellar mass of the galaxy.
    '''

    ############################################################################
    # Mask the maps
    #---------------------------------------------------------------------------
    sMass_mask = map_mask + np.isnan(sMass_density)

    msMass_density = ma.array(sMass_density.value, mask=sMass_mask)

    mr_band = ma.array(r_band, mask=sMass_mask)
    ############################################################################


    ############################################################################
    # Plot the stellar mass density map
    #---------------------------------------------------------------------------
    plot_sMass_image(msMass_density, 
                     gal_ID, 
                     IMAGE_DIR=IMAGE_DIR, 
                     IMAGE_FORMAT=IMAGE_FORMAT)

    if IMAGE_DIR is None:
        plt.show()
    ############################################################################


    ############################################################################
    # If necessary, determine optical center via the max luminosity in the 
    # r-band.
    #---------------------------------------------------------------------------
    if x0 is None:
        optical_center = np.unravel_index(ma.argmax(mr_band), mr_band.shape)
        '''
        i_center = center_guess[0]
        j_center = center_guess[1]
        '''
    else:
        optical_center = np.array([int(y0), int(x0)])
        '''
        i_center = int(y0)
        j_center = int(x0)
        '''
    ############################################################################


    ############################################################################
    # If all of the data is masked, return null values for everything
    #---------------------------------------------------------------------------
    if np.sum(map_mask == 0) == 0:

        mass_vel_table = Table()

        mass_vel_table['radius'] = [np.NaN]
        mass_vel_table['M_star'] = [np.NaN]
        mass_vel_table['star_vel'] = [np.NaN]
        mass_vel_table['star_vel_err'] = [np.NaN]

        print('ALL DATA POINTS FOR THE GALAXY ARE MASKED!')
    ############################################################################


    ############################################################################
    # If there is unmasked data in the map, execute the function as normal.
    #---------------------------------------------------------------------------
    else:
        mass_vel_table = find_mass_curve(z, 
                                         map_mask,
                                         msMass_density, 
                                         optical_center, 
                                         phi, 
                                         ba)
    ############################################################################

    return mass_vel_table






################################################################################
################################################################################
################################################################################

def fit_mass_curve(data_table, gal_ID, IMAGE_DIR=None, IMAGE_FORMAT='eps'):
    '''
    Fit the stellar mass rotation curve to the disk velocity function.


    PARAMETERS
    ==========

    data_table : astropy table
        Contains the deprojected radius and the mass within that radius

    gal_ID : string
        [PLATE]-[IFU]

    IMAGE_DIR : string
        File path to which various produced images are saved.  
        Default value is None (do not save images).

    IMAGE_FORMAT : string
        Saved image file format.  Default format is eps.


    RETURNS
    =======

    best_fit_values : dictionary
        Values (and uncertainties) of the best-fit parameters
    '''


    ############################################################################
    # Set up initial guesses for the best-fit parameters
    #---------------------------------------------------------------------------
    # Central disk mass density [M_sol/pc^2]
    Sigma_disk_guess = 1000.

    # Disk scale radius [kpc]
    R_disk_guess = 1.

    param_guesses = [Sigma_disk_guess, R_disk_guess]
    ############################################################################


    ############################################################################
    # Set up bounds for the best-fit parameters
    #---------------------------------------------------------------------------
    # Central disk mass density [M_sol/pc^2]
    Sigma_disk_min = 0.
    Sigma_disk_max = 1e4
    Sigma_disk_bounds = (Sigma_disk_min, Sigma_disk_max)

    # Disk scale radius [kpc]
    R_disk_min = 0.
    R_disk_max = 10. 
    R_disk_bounds = (R_disk_min, R_disk_max)

    param_bounds = [Sigma_disk_bounds, R_disk_bounds]
    ############################################################################


    ############################################################################
    # Find the best-fit parameters
    #---------------------------------------------------------------------------
    try:
        popt, pconv = curve_fit(disk_vel, 
                                data_table['radius'], 
                                data_table['star_vel'], 
                                p0=param_guesses,
                                sigma=data_table['star_vel_err'])

        #-----------------------------------------------------------------------
        # Determine uncertainties in the fitted parameters
        #-----------------------------------------------------------------------
        perr = np.sqrt(np.diag(pconv))
        #-----------------------------------------------------------------------


        #-----------------------------------------------------------------------
        # Unpack results
        #-----------------------------------------------------------------------
        chi2 = chi2_mass(popt, 
                         data_table['radius'], 
                         data_table['star_vel'], 
                         data_table['star_vel_err'])

        best_fit_values = {'Sigma_disk': popt[0], 
                           'Sigma_disk_err': perr[0], 
                           'R_disk': popt[1], 
                           'R_disk_err': perr[1], 
                           'chi2_disk': chi2}
        #-----------------------------------------------------------------------


        #-----------------------------------------------------------------------
        # Plot data and best-fit curve
        #-----------------------------------------------------------------------
        plot_fitted_disk_rot_curve(gal_ID, 
                                   data_table, 
                                   best_fit_values, 
                                   chi2,
                                   IMAGE_DIR=IMAGE_DIR, 
                                   IMAGE_FORMAT=IMAGE_FORMAT)

        if IMAGE_DIR is None:
            plt.show()
        #-----------------------------------------------------------------------
        
    except RuntimeError:
        print(gal_ID, 'fit did not converge.', flush=True)

        best_fit_values = None
    '''
    result = minimize(chi2_mass, 
                      param_guesses, 
                      method='L-BFGS-B',
                      args=(data_table['radius'], 
                            data_table['star_vel'], 
                            data_table['star_vel_err']), 
                      bounds=param_bounds, 
                      options={'disp':101})

    if result.success:
        print(gal_ID, 'successfully fit!')

        #-----------------------------------------------------------------------
        # Determine uncertainties in the fitted parameters
        #-----------------------------------------------------------------------
        fit_param_err = np.sqrt(np.diag(result.hess_inv.todense()))
        #-----------------------------------------------------------------------


        #-----------------------------------------------------------------------
        # Unpack results
        #-----------------------------------------------------------------------
        best_fit_values = {'Sigma_disk': result.x[0], 
                           'Sigma_disk_err': fit_param_err[0], 
                           'R_disk': result.x[1], 
                           'R_disk_err': fit_param_err[1]}
        #-----------------------------------------------------------------------


        #-----------------------------------------------------------------------
        # Plot data and best-fit curve
        #-----------------------------------------------------------------------
        plot_fitted_disk_rot_curve(gal_ID, 
                                   data_table, 
                                   best_fit_values, 
                                   result.fun,
                                   IMAGE_DIR=IMAGE_DIR, 
                                   IMAGE_FORMAT=IMAGE_FORMAT)

        if IMAGE_DIR is None:
            plt.show()
        #-----------------------------------------------------------------------
    else:
        print(gal_ID, 'fit did not converge.')

        best_fit_values = None
    '''
    ############################################################################

    return best_fit_values



