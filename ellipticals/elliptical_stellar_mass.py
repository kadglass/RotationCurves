import numpy as np
import matplotlib.pyplot as plt
import matplotlib

from scipy.optimize import curve_fit
from astropy.table import Table

from elliptical_plottingFunctions import plot_stellar_mass
from elliptical_stellar_mass_functions import *

import sys
sys.path.insert(1, '/Users/nityaravi/Documents/Github/RotationCurves/spirals/')
from DRP_rotation_curve_functions import calc_stellar_mass


H_0 = 100      # Hubble's Constant in units of h km/s/Mpc
c = 299792.458 # Speed of light in units of km/s
G = 4.30091E-6 # Gravitation constant in units of (km/s)^2 kpc/Msun

MANGA_FIBER_DIAMETER = 2*(1/60)*(1/60)*(np.pi/180) # angular fiber diameter (2") in radians
MANGA_SPAXEL_SIZE = 0.5*(1/60)*(1/60)*(np.pi/180)  # spaxel size (0.5") in radians


def calc_mass_curve(sMass_density,
                    sMass_density_err,
                    flux,
                    ba, 
                    phi, 
                    z):
    '''
    Extract the mass as a function of radius from the stellar mass density map, 
    and convert this to a rotation curve (velocity as a function of radius).
    Modified from spiral galaxy functions

    PARAMETERS
    ==========

    sMass_density : numpy array of shape (n,n)
        Stellar mass density map in units of log(Msun/spaxel^2)
        
    sMass_density_err : numpy array of shape (n,n)
        stellar mass density error map in log(Msun/spaxel^2)

    flux : numpy array of shape (n,n)
        g-band weighted mean flux data

    ba : float
        Ratio of the galaxy's minor axis to major axis

    phi : float
        Angle (east of north) of rotation in the 2-D, observational plane

        NOTE: east is 'left' per astronomy convention

    z : float
        Galaxy redshift


    RETURNS
    =======

    mass_table : astropy table
        Data table containing M(r) for the stellar mass of the galaxy.
    '''

    ############################################################################
    # Mask the maps
    #---------------------------------------------------------------------------
    sMass_mask = np.isnan(sMass_density)

    msMass_density = ma.array(sMass_density, mask=sMass_mask)

    sMass_err_mask = np.isnan(sMass_density_err)
    msMass_density_err = ma.array(sMass_density_err, mask=sMass_err_mask)

    mflux = ma.array(flux, mask=sMass_mask)
    ############################################################################

    # find center
    optical_center = np.unravel_index(ma.argmax(mflux), mflux.shape)
        
    ############################################################################


    ############################################################################
    # If all of the data is masked, return null values for everything
    #---------------------------------------------------------------------------
    if np.sum(sMass_mask == 0) == 0:

        mass_vel_table = Table()

        mass_vel_table['radius'] = [np.NaN]
        mass_vel_table['M_star'] = [np.NaN]
        mass_vel_table['M_star_err'] = [np.NaN]

        print('ALL DATA POINTS FOR THE GALAXY ARE MASKED!')
    ############################################################################


    ############################################################################
    # If there is unmasked data in the map, execute the function as normal.
    #---------------------------------------------------------------------------
    else:
        mass_table = find_mass_curve(z,
                                     sMass_mask,
                                     msMass_density,
                                     msMass_density_err,
                                     optical_center, 
                                     phi, 
                                     ba)
    ############################################################################

    return mass_table



def find_mass_curve(z, 
                    map_mask, 
                    msMass_density, 
                    msMass_density_err,
                    optical_center, 
                    phi, 
                    ba):
    '''
    Measure the rotation curve for the disk component of the galaxy based on the 
    stellar mass density map.
    Modified from spiral galaxy functions


    PARAMETERS
    ==========

    z : float
        Redshift of galaxy

    map_mask : numpy array of shape (n,n)
        Boolean array where true values represent spaxels which are masked

    msMass_density : numpy array of shape (n,n)
        Masked stellar mass density map

    msMass_density_err : numpy array of shape (n,n)
        Masked stellar mass density error map 

    optical_center : tuple of shape (2,1)
        Array coordinates of the kinematic center of the galaxy

    phi: float
        angle (east of north) of rotation in the 2-D observational plane
        NOTE: East is 'left' per astronomy convention

    ba : float
        b/a Sersic axis ratio for galaxy


    RETURNS
    =======

    data_table : astropy table
        Table of output data, including the deprojected radius and M(r)
    '''


    ############################################################################
    # Convert pixel distance to physical distances.
    #---------------------------------------------------------------------------
    dist_to_galaxy_Mpc = c*z/H_0
    dist_to_galaxy_kpc = dist_to_galaxy_Mpc*1000

    pix_scale_factor = dist_to_galaxy_kpc*np.tan(MANGA_SPAXEL_SIZE)
    ############################################################################


    ############################################################################
    # Create a meshgrid for all coordinate points based on the dimensions of
    # the stellar mass density numpy array.
    #---------------------------------------------------------------------------
    array_length = msMass_density.shape[0]  # y-coordinate distance
    array_width = msMass_density.shape[1]  # x-coordinate distance

    X_RANGE = np.arange(0, array_width, 1)
    Y_RANGE = np.arange(0, array_length, 1)
    X_COORD, Y_COORD = np.meshgrid( X_RANGE, Y_RANGE)
    ############################################################################


    ############################################################################
    # Initialization code to draw the elliptical annuli.
    #---------------------------------------------------------------------------
    phi_elip = (90 - phi)*np.pi/180.

    x_diff = X_COORD - optical_center[1]
    y_diff = Y_COORD - optical_center[0]

    ellipse = (x_diff*np.cos(phi_elip) - y_diff*np.sin(phi_elip))**2 \
              + (x_diff*np.sin(phi_elip) + y_diff*np.cos(phi_elip))**2 \
              / ba**2
    ############################################################################


    ############################################################################
    # Initialize the lists for the columns of the output table
    #---------------------------------------------------------------------------
    radius = []

    sMass_interior_curve = []
    sMass_interior_curve_err = []
    ############################################################################


    ############################################################################
    # Initialize the stellar mass surface density interior to an annulus to
    # be 0 solar masses.
    #---------------------------------------------------------------------------
    sMass_interior = 0.
    sMass_interior_err2 = 0.
    sMass_interior_err = 0.
    ############################################################################


    ############################################################################
    #---------------------------------------------------------------------------
    dR = 2
    R = 2

    while True:

        deproj_dist_kpc = R*pix_scale_factor

        ########################################################################
        # Define an eliptical annulus and check if there is at least one point 
        # within the mask.
        #-----------------------------------------------------------------------
        pix_between_annuli = np.logical_and( (R - dR)**2 <= ellipse, 
                                             ellipse < R**2)

        if not np.any( map_mask[ pix_between_annuli] == 0):
            break
        ########################################################################


        ########################################################################
        # Extract the stellar mass interior to that annulus.
        #-----------------------------------------------------------------------
        sMass_interior, sMass_interior_err2 = calc_stellar_mass(sMass_interior,
                                            sMass_interior_err2,
                                           10**msMass_density, 
                                           10**msMass_density_err,
                                           pix_between_annuli)

        sMass_interior_err = np.sqrt(sMass_interior_err2)
        
        ########################################################################
    

        ########################################################################
        # Append the corresponding values to their respective arrays to write to 
        # the output data file.
        #-----------------------------------------------------------------------
        if np.isfinite(sMass_interior):
            radius.append( deproj_dist_kpc)

            sMass_interior_curve.append( np.log10(sMass_interior))
            sMass_interior_curve_err.append( np.log10(sMass_interior_err))
        ########################################################################


        ########################################################################
        # Increment the radius of the annulus R by dR
        #-----------------------------------------------------------------------
        R += dR
        ########################################################################


    ############################################################################
    # Build output data table
    #---------------------------------------------------------------------------
    data_table = Table()

    data_table['radius'] = radius

    data_table['M_star'] = sMass_interior_curve
    data_table['M_star_err'] = sMass_interior_curve_err
    
    ############################################################################


    return data_table


def fit_mass_curve(data_table, gal_ID, COV_DIR='', IMAGE_DIR=None, IMAGE_FORMAT='png'):
    '''
    Fit the stellar mass distribution to the exponential sphere model.


    PARAMETERS
    ==========

    data_table : astropy tables
        Contains the deprojected radius and the mass within that radius

    gal_ID : string
        [PLATE]-[IFU]
    
    COV_DIR : string
        path to directory for covariance matrices. default is none so covariance
        matrix will save in current directory

    IMAGE_DIR : string
        File path to which various produced images are saved.  
        Default value is None (do not save images).

    IMAGE_FORMAT : string
        Saved image file format.  Default format is png.


    RETURNS
    =======

    best_fit_values : dictionary
        Values (and uncertainties) of the best-fit parameters
    '''


    ############################################################################
    # Set up initial guesses for the best-fit parameters
    #---------------------------------------------------------------------------
    
    rho_c_guess =  10**9.

    R_scale_guess = 1.
        
    param_guesses = [rho_c_guess, R_scale_guess]

    ############################################################################


    ############################################################################
    # Set up bounds for the best-fit parameters
    #---------------------------------------------------------------------------
    # Bulge central density [M_sol/kpc^3] 
    rho_c_min = 0.
    rho_c_max = 1e14

    # Bulge scale radius [kpc]
    R_scale_min = 0.
    R_scale_max = 100.


    param_bounds = ([rho_c_min, R_scale_min], 
                        [rho_c_max, R_scale_max])
    ############################################################################


    ############################################################################
    # Find the best-fit parameters
    #---------------------------------------------------------------------------
    try:
        popt, pcov = curve_fit(exponential_sphere, 
                                    data_table['radius'], 
                                    10**data_table['M_star'], 
                                    p0=param_guesses,
                                    bounds=param_bounds,
                                    sigma=10**data_table['M_star_err']
                                    )


        #-----------------------------------------------------------------------
        # Determine uncertainties in the fitted parameters
        #-----------------------------------------------------------------------
        np.save(COV_DIR + gal_ID + '_cov.npy', pcov) 

        perr = np.sqrt(np.diag(pcov))
        #-----------------------------------------------------------------------


        #-----------------------------------------------------------------------
        # Unpack results
        #-----------------------------------------------------------------------
        chi2 = chi2_mass(popt, 
                     data_table['radius'], 
                     data_table['M_star'], 
                     data_table['M_star_err'])




        best_fit_values = {'rho_c' : popt[0],
                        'rho_c_err' : perr[0],
                        'R_scale' : popt[1],
                        'R_scale_err' : perr[1], 
                        'chi2_M_star': chi2}


        #-----------------------------------------------------------------------


        #-----------------------------------------------------------------------
        # Plot data and best-fit curve
        #-----------------------------------------------------------------------
        
        plot_stellar_mass(gal_ID,
                              data_table,
                              best_fit_values,
                              chi2,
                              COV_DIR,
                              IMAGE_DIR,
                              IMAGE_FORMAT)
        
    #-----------------------------------------------------------------------
        
    except RuntimeError:
        print(gal_ID, 'fit did not converge.', flush=True)

        best_fit_values = None
 
    ############################################################################

    return best_fit_values


