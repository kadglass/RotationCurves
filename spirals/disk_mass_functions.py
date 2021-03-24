
import numpy as np

from astropy.table import Table
import astropy.units as u

from DRP_rotation_curve_functions import calc_stellar_mass, calc_velocity

import sys
sys.path.insert(1, '/Users/kellydouglass/Documents/Research/Rotation_curves/Yifan_Zhang/RotationCurve')
from rotation_curve_functions import disk_vel



################################################################################
# Constants
#-------------------------------------------------------------------------------
H_0 = 100      # Hubble's Constant in units of h km/s/Mpc
c = 299792.458 # Speed of light in units of km/s
G = 4.30091E-6 # Gravitation constant in units of (km/s)^2 kpc/Msun

MANGA_FIBER_DIAMETER = 2*(1/60)*(1/60)*(np.pi/180) # angular fiber diameter (2") in radians
MANGA_SPAXEL_SIZE = 0.5*(1/60)*(1/60)*(np.pi/180)  # spaxel size (0.5") in radians
################################################################################



################################################################################
################################################################################
################################################################################

def find_mass_curve(z, 
                    map_mask, 
                    msMass_density, 
                    optical_center, 
                    phi, 
                    ba):
    '''
    Measure the rotation curve for the disk component of the galaxy based on the 
    stellar mass density map.


    PARAMETERS
    ==========

    z : float
        Redshift of galaxy

    map_mask : numpy array of shape (n,n)
        Boolean array where true values represent spaxels which are masked

    msMass_density : numpy array of shape (n,n)
        Masked stellar mass density map

    optical_center : tuple of shape (2,1)
        Array coordinates of the kinematic center of the galaxy

    phi_EofN_deg : float
        angle (east of north) of rotation in the 2-D observational plane
        NOTE: East is 'left' per astronomy convention

    axis_ratio : float
        b/a Sersic axis ratio for galaxy


    RETURNS
    =======

    data_table : astropy table
        Table of output data, including the deprojected radius, M(r), and v(r)
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
    rot_curve_dist = []

    sMass_interior_curve = []
    sVel_rot_curve = []
    sVel_rot_curve_err = []
    ############################################################################


    ############################################################################
    # Initialize the stellar mass surface density interior to an annulus to
    # be 0 solar masses.
    #---------------------------------------------------------------------------
    sMass_interior = 0.
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
        sMass_interior = calc_stellar_mass(sMass_interior, 
                                           10**msMass_density, 
                                           pix_between_annuli)

        sVel_rot, sVel_rot_err = calc_velocity(sMass_interior*u.Msun, 0*u.Msun, 
                                               deproj_dist_kpc*u.kpc, 0*u.kpc)
        ########################################################################


        ########################################################################
        # Append the corresponding values to their respective arrays to write to 
        # the output data file.
        #-----------------------------------------------------------------------
        if np.isfinite(sMass_interior) and np.isfinite(sVel_rot):
            rot_curve_dist.append( deproj_dist_kpc)
            #rot_curve_dist_err.append( deproj_dist_kpc_err)

            sMass_interior_curve.append( np.log10(sMass_interior))
            sVel_rot_curve.append( sVel_rot.value)
            sVel_rot_curve_err.append( sVel_rot_err.value)
        ########################################################################


        ########################################################################
        # Increment the radius of the annulus R by dR
        #-----------------------------------------------------------------------
        R += dR
        ########################################################################
    ############################################################################


    ############################################################################
    # Build output data table
    #---------------------------------------------------------------------------
    data_table = Table()

    data_table['radius'] = rot_curve_dist
    #data_table['radius_err'] = rot_curve_dist_err

    data_table['M_star'] = sMass_interior_curve
    data_table['star_vel'] = sVel_rot_curve
    data_table['star_vel_err'] = sVel_rot_curve_err
    ############################################################################


    return data_table




################################################################################
################################################################################
################################################################################



#def calc_velocity( mass, mass_err, radius, radius_err):
    '''
    Calculate the velocity corresponding to the mass interior to a given 
    radius assuming circular motion:
                    v(r) = sqrt( G * M(r) / r)


    Parameters:
    ===========

    mass : float
        Total stellar mass interior to previous annuli

    mass_err : float
        Uncertainty in the total mass

    radius : float
        Radius of current annulus

    radius_err : float
        Uncertainty in the radius


    Returns:
    ========

    velocity : float
        Velocity (in units of km/s) corresponding to total stellar mass within 
        radius, assuming circular motion:
                            v(r) = sqrt( G * M(r) / r)

    velocity_err : float
        Uncertainty in velocity

    '''

    ############################################################################
    # Calculate velocity corresponding to mass within given radius
    #---------------------------------------------------------------------------
 #   velocity = np.sqrt(G*mass/radius)
    ############################################################################


    ############################################################################
    # Calculate velocity error
    #---------------------------------------------------------------------------
#    velocity_err = 0.5 * velocity * np.sqrt( (mass_err / mass)**2 \
#                                             + (radius_err / radius)**2)
    ############################################################################

#    return velocity, velocity_err




################################################################################
################################################################################
################################################################################

def chi2_mass(params, r, v_data_mass, v_data_mass_err):
    '''
    Calculate the reduced chi2 of the disk velocity curve

    (Written by Yifan Zhang)


    PARAMETERS
    ==========

    params : list
        Best-fit parameter values for the disk velocity function

    r : numpy ndarray of shape (n,)
        Deprojected radii, units of kpc

    v_data_mass : numpy ndarray of shape (n,)
        Velocity due to the mass within a radius r, units of km/s

    v_data_mass_err : numpy ndarray of shape (n,)
        Uncertainty in the velocities due to the mass within a radius r, in 
        units of km/s


    RETURNS
    =======

    n_chi2 : float
        Reduced chi2 of the fit
    '''

    #v_model= disk_vel(params, r)
    v_model = disk_vel(r, params[0], params[1])

    chi2 = np.sum((v_data_mass - v_model)**2/v_data_mass_err**2)

    n_chi2 = chi2/(len(r) - len(params))
    
    return n_chi2



