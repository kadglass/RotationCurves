from astropy.table import QTable, Column
import astropy.constants as const
import astropy.units as u

import numpy as np
import numpy.ma as ma

import math



################################################################################
# Declare constants
#-------------------------------------------------------------------------------
H_0 = 100 * (u.km / ( u.s * u.Mpc))  # Hubble's Constant in units of km /s /Mpc

MANGA_FIBER_DIAMETER = 2*(1/60)*(1/60)*(np.pi/180) # angular fiber diameter (2") in radians
MANGA_SPAXEL_SIZE = 0.5*(1/60)*(1/60)*(np.pi/180)  # spaxel size (0.5") in radians
################################################################################



################################################################################
################################################################################
################################################################################


def build_mask( Ha_vel_mask, sMass_density):
    '''
    Build mask for galaxy.


    Parameters:
    ===========

    Ha_vel_mask : numpy ndarray of shape (n,n)
        Bitmask for the H-alpha velocity map

    sMass_density : numpy ndarray of shape (n,n)
        Stellar mass density map


    Returns:
    ========

    mask : numpy boolean ndarray of shape (n,n)
        Mask map where true values are to be masked.
    '''


    ############################################################################
    # Mask all spaxels with bit > 0
    #---------------------------------------------------------------------------
    Ha_vel_mask_boolean = Ha_vel_mask > 0
    ############################################################################


    ############################################################################
    # Mask all spaxels with NaN values in the stellar mass density estimate
    #---------------------------------------------------------------------------
    # Locate np.nan values in sMass_density array
    sMass_density_boolean = np.isnan( sMass_density)
    ############################################################################


    ############################################################################
    # Combine mask arrays
    #---------------------------------------------------------------------------
    mask = np.logical_or( Ha_vel_mask_boolean, sMass_density_boolean)
    ############################################################################

    return mask





################################################################################
################################################################################
################################################################################



def find_rot_curve( z, 
                    data_mask, 
                    r_band, r_band_ivar, 
                    Ha_vel, mHa_vel, mHa_vel_ivar, 
                    msMass_density, 
                    optical_center, 
                    phi_EofN_deg, 
                    axis_ratio):
    '''
    Measure the rotation curves for the galaxy based on the unmasked H-alpha 
    velocity data.


    Parameters
    ==========

    z : float
        Redshift of galaxy

    data_mask : numpy array of shape (n,n)
        Boolean array where true values represent spaxels which are masked

    r_band : numpy array of shape (n,n)
        r-band flux map

    r_band_ivar : numpy array of shape (n,n)
        Inverse variances of the r-band flux map

    Ha_vel : numpy array of shape (n,n)
        H-alpha velocity map

    mHa_vel : numpy array of shape (n,n)
        Masked H-alpha velocity map

    mHa_vel_ivar : numpy array of shape (n,n)
        Masked inverse variances in H-alpha velocity map

    msMass_density : numpy array of shape (n,n)
        Masked stellar mass density map

    optical_center : tuple of shape (2,1)
        Array coordinates of the optical center of the galaxy

    phi_EofN_deg : float
        angle (east of north) of rotation in the 2-D observational plane
        NOTE: East is 'left' per astronomy convention

    axis_ratio : float
        b/a Sersic axis ratio for galaxy


    Returns
    =======

    list_dict : dictionary
        Dictionary of output lists

    flux_center : float
        r-band flux of spaxel at galaxy's optical center

    mvel_contour_plot : numpy array of shape (n,n)
        H-alpha velocity map masking all spaxels not included in annuli
    '''


    ############################################################################
    # Extract the flux from the center of the galaxy.
    #---------------------------------------------------------------------------
    flux_center = r_band[ optical_center]
    flux_center_err = 1/np.sqrt(r_band_ivar[ optical_center])

    '''
    print("flux_center:", flux_center)
    print("flux_center_error:", flux_center_error)
    '''
    ############################################################################


    ############################################################################
    # Find the first data point along the galaxy's semi-major axis where
    # 'r_band' equals zero, therefore finding the point to signal to stop
    # collecting data for the rotation curve. Set this point to -999 as a flag.
    #---------------------------------------------------------------------------
    r_band = flag_data_ends( r_band, phi_EofN_deg, optical_center)
    ############################################################################


    ############################################################################
    # Convert pixel distance to physical distances in units of both
    # kiloparsecs and centimeters.
    #---------------------------------------------------------------------------
    dist_to_galaxy_kpc = ( z * const.c.to('km/s') / H_0).to('kpc')

    pix_scale_factor = dist_to_galaxy_kpc * np.tan( MANGA_SPAXEL_SIZE)

    '''
    print("z:", z)
    print("dist_to_galaxy_kpc:", dist_to_galaxy_kpc)
    print("dist_to_galaxy_kpc_err:", dist_to_galaxy_kpc_err)
    print("dist_to_galaxy_cm:", dist_to_galaxy_cm)
    print("dist_to_galaxy_cm_err:", dist_to_galaxy_cm_err)
    print("pix_scale_factor:", pix_scale_factor)
    print("pix_scale_factor_err:", pix_scale_factor_err)
    '''
    ############################################################################


    ############################################################################
    # Create a meshgrid for all coordinate points based on the dimensions of
    # the H-alpha velocity numpy array.
    #---------------------------------------------------------------------------
    array_length = mHa_vel.shape[0]  # y-coordinate distance
    array_width = mHa_vel.shape[1]  # x-coordinate distance

    X_RANGE = np.arange(0, array_width, 1)
    Y_RANGE = np.arange(0, array_length, 1)
    X_COORD, Y_COORD = np.meshgrid( X_RANGE, Y_RANGE)
    ############################################################################


    ############################################################################
    # Initialization code to draw the elliptical annuli and to normalize the
    # 2D-arrays for the max and min velocity so as to check for anomalous data.
    #---------------------------------------------------------------------------
    phi_elip = math.radians( 90 - ( phi_EofN_deg / u.deg)) * u.rad

    x_diff = X_COORD - optical_center[1]
    y_diff = Y_COORD - optical_center[0]

    ellipse = ( x_diff*np.cos( phi_elip) - y_diff*np.sin( phi_elip))**2 \
              + ( x_diff*np.sin( phi_elip) + y_diff*np.cos( phi_elip))**2 \
              / ( axis_ratio)**2
    ############################################################################


    ############################################################################
    # Initialize the rotation curve lists that will store the rotation curve
    # data.
    #---------------------------------------------------------------------------
    rot_curve_dist = []

    rot_curve_max_vel = []
    rot_curve_max_vel_err = []
    rot_curve_min_vel = []
    rot_curve_min_vel_err = []
    rot_curve_vel_avg = []
    rot_curve_vel_avg_err = []
    rot_curve_vel_diff = []
    rot_curve_vel_diff_err = []

    totMass_interior_curve = []
    totMass_interior_curve_err = []

    sMass_interior_curve = []
    sVel_rot_curve = []
    sVel_rot_curve_err = []

    dmMass_interior_curve = []
    dmMass_interior_curve_err = []
    dmVel_rot_curve = []
    dmVel_rot_curve_err = []
    ############################################################################


    ############################################################################
    # Put all lists inside a dictionary for cleaner function calls.
    #---------------------------------------------------------------------------
    list_dict = {'radius':rot_curve_dist, 'M_star':sMass_interior_curve,
                 'max_vel':rot_curve_max_vel, 'max_vel_err':rot_curve_max_vel_err,
                 'min_vel':rot_curve_min_vel, 'min_vel_err':rot_curve_min_vel_err,
                 'avg_vel':rot_curve_vel_avg, 'avg_vel_err':rot_curve_vel_avg_err,
                 'vel_diff':rot_curve_vel_diff, 'vel_diff_err':rot_curve_vel_diff_err,
                 'M_tot':totMass_interior_curve, 'M_tot_err':totMass_interior_curve_err,
                 'star_vel':sVel_rot_curve, 'star_vel_err':sVel_rot_curve_err,
                 'DM':dmMass_interior_curve, 'DM_err':dmMass_interior_curve_err,
                 'DM_vel':dmVel_rot_curve, 'DM_vel_err':dmVel_rot_curve_err}
    ############################################################################


    ############################################################################
    # Initialize the stellar mass surface density interior to an annulus to
    #    be 0 solar masses.
    #---------------------------------------------------------------------------
    sMass_interior = 0 * u.M_sun
    ############################################################################


    ############################################################################
    #---------------------------------------------------------------------------
    dR = 2
    R = 2

    valid_data = True

    while valid_data:

        deproj_dist_kpc = R * pix_scale_factor

        ########################################################################
        # Define an eliptical annulus and check if either of the edge points 
        # are within that annulus.
        #
        # NOTE: Although if the edge point is within 'pix_between_annuli' and 
        #       thus 'valid_data' is set to False, the current iteration of the 
        #       loop still completes as intended.
        #-----------------------------------------------------------------------
        pix_between_annuli = np.logical_and( (R - dR)**2 <= ellipse, 
                                             ellipse < R**2)

        if np.any( r_band[ pix_between_annuli] == -999):
            valid_data = False
        ########################################################################


        ########################################################################
        # Extract the maximum and minimum velocity at an annulus and the stellar 
        # mass interior to that annulus.
        #-----------------------------------------------------------------------
        list_dict = find_velocity_extrema( pix_between_annuli, deproj_dist_kpc, 
                                           mHa_vel, mHa_vel_ivar, 
                                           msMass_density, sMass_interior, 
                                           list_dict)
        ########################################################################


        ########################################################################
        # Increment the radius of the annulus R by dR
        #-----------------------------------------------------------------------
        R += dR
        ########################################################################


    ############################################################################
    # Add the pixels of the H-alpha velocity field analyzed in the current 
    # iteration of the algorithm to an image that plots all the pixels analyzed 
    # for a given galaxy.
    #---------------------------------------------------------------------------
    contour_boolean = ellipse >= R*R
    contour_mask = np.logical_or( data_mask, contour_boolean)

    masked_vel_contour_plot = ma.masked_where( contour_mask, Ha_vel)
    ############################################################################

    return list_dict, flux_center, flux_center_err, masked_vel_contour_plot



################################################################################
################################################################################
################################################################################




def flag_data_ends( r_band, phi, center):
    '''
    Find the first data point along the galaxy's semi-major axis where the 
    r-band flux equals zero, therefore finding the point to signal to stop
    collecting data for the rotation curve.  Set this point to -999 as a flag.


    Parameters:
    ===========

    r_band : numpy array of shape (n,n)
        r-band flux map

    phi : float
        angle (east of north) of rotation in the 2-D observational plane
        NOTE: East is 'left' per astronomy convention

    center : numpy array of shape (2,1)
        Array coordinates of the optical center of the galaxy


    Returns:
    ========

    r_band : numpy array of shape (n,n)
        r-band flux map
    '''

    array_length = r_band.shape[0]  # y-coordinate distance
    array_width = r_band.shape[1]  # x-coordinate distance

    x_center = center[1]
    y_center = center[0]

    phi_edge = phi.to('rad')
    slope = -1 * (np.cos( phi_edge) / np.sin( phi_edge))
    y_intercept = y_center - slope * x_center

    #---------------------------------------------------------------------------
    x_edge_pos = x_center
    y_edge_pos = y_center

    edge_pos = False

    while not edge_pos:
        x_temp_pos = x_edge_pos + 1
        y_temp_pos = int( slope * x_temp_pos + y_intercept)

        if ( x_temp_pos >= array_width) or ( y_temp_pos < 0) or ( y_temp_pos >= array_length) or ( r_band[ y_temp_pos, x_temp_pos] == 0):
            edge_pos = True

        else:
            x_edge_pos = x_temp_pos
            y_edge_pos = y_temp_pos
    #---------------------------------------------------------------------------
    #---------------------------------------------------------------------------
    x_edge_neg = x_center
    y_edge_neg = y_center

    edge_neg = False

    while not edge_neg:
        x_temp_neg = x_edge_neg - 1
        y_temp_neg = int( slope * x_temp_neg + y_intercept)

        if ( x_temp_neg < 0) or ( y_temp_neg < 0) or ( y_temp_neg >= array_length) or ( r_band[ y_temp_neg, x_temp_neg] == 0):
            edge_neg = True

        else:
            x_edge_neg = x_temp_neg
            y_edge_neg = y_temp_neg
    #---------------------------------------------------------------------------

    r_band[ y_edge_pos, x_edge_pos] = -999
    r_band[ y_edge_neg, x_edge_neg] = -999


    return r_band



################################################################################
################################################################################
################################################################################



def find_velocity_extrema( pix_between_annuli, deproj_dist, mHa_vel, 
                           mHa_vel_ivar, msMass_density, sMass_interior, 
                           list_dict):
    '''
    Locate the minimum and maximum velocity values on each annulus


    Parameters:
    ===========

    pix_between_annuli : numpy array of shape (n,n)
        Boolean array, where true values indicate spaxels that are on the 
        current annulus

    deproj_dist : float
        Deprojected radius of current annulus in units of kpc

    mHa_vel : numpy array of shape (n,n)
        Masked Halpha velocity map

    mHa_vel_ivar : numpy array of shape (n,n)
        Masked inverse variances in H-alpha velocity map

    msMass_density : numpy array of shape (n,n)
        Masked stellar mass density map

    sMass_interior : float
        Stellar mass interior to previous annuli


    Returns:
    ========

    list_dict : dictionary
        Contains lists of all the output values
    '''


    deproj_dist_m = deproj_dist.to('m')


    ############################################################################
    # Find the coordinates of the max/min velocity for a given annulus.
    #
    # If there is no max/min velocity point (i.e. all eligible data points are 
    # masked), then skip to the next annulus.
    #
    # If there is more than one max or more than one min velocity at the given 
    # annulus, then use the one closest to the semi-major axis.
    #---------------------------------------------------------------------------
    max_vel_point = np.argwhere( mHa_vel[ pix_between_annuli].max() == mHa_vel)
    min_vel_point = np.argwhere( mHa_vel[ pix_between_annuli].min() == mHa_vel)
    ############################################################################


    ############################################################################
    # Only the length of 'max_vel_point' is tested because if the maximum of 
    # 'masked_Ha_vel' returns nothing, the minimum will also return nothing 
    # because there are only masked values.
    #---------------------------------------------------------------------------
    if len( max_vel_point) == 0:

        print('ALL DATA POINTS IN', deproj_dist, 'ANNULUS ARE MASKED!!!')

    else:
        ########################################################################
        # NOTE: The max/min velocity coordinates are extracted in the [0][j] 
        #       fashion because sometimes there can be more than one point that 
        #       contains the max/min velocity at that annulus.  Thus, the first 
        #       point with the maximum velocity is extracted and separated into 
        #       its x- and y-coordinates.
        #-----------------------------------------------------------------------
        max_vel_point_x = max_vel_point[0][1]
        max_vel_point_y = max_vel_point[0][0]

        min_vel_point_x = min_vel_point[0][1]
        min_vel_point_y = min_vel_point[0][0]


        max_vel = mHa_vel[max_vel_point_y, max_vel_point_x] * ( u.km / u.s)
        max_vel_ivar = mHa_vel_ivar[max_vel_point_y, max_vel_point_x] * ( u.s / u.km)**2

        min_vel = mHa_vel[min_vel_point_y, min_vel_point_x] * ( u.km / u.s)
        min_vel_ivar = mHa_vel_ivar[min_vel_point_y, min_vel_point_x] * ( u.s / u.km)**2
        ########################################################################


        ########################################################################
        # Calculate the average rotational velocity at an annulus, its error, 
        # difference in the rotational velocities, its error, and the total mass 
        # interior to that annulus along with its error.
        #-----------------------------------------------------------------------
        avg_vel, avg_vel_err, vel_diff, vel_diff_err = calc_avg_vel( max_vel, 
                                                                     min_vel, 
                                                                     max_vel_ivar, 
                                                                     min_vel_ivar)

        mass_interior, mass_interior_err = calc_mass_interior( avg_vel, 
                                                               avg_vel_err,
                                                               deproj_dist_m, 0*u.m)
        ########################################################################


        ########################################################################
        # Calculate the stellar rotational velocity at a radius, its error, and 
        # the stellar mass interior to that radius.
        #-----------------------------------------------------------------------
        sMass_interior = calc_stellar_mass( sMass_interior, 
                                            msMass_density, 
                                            pix_between_annuli)

        sVel_rot, sVel_rot_err = calc_velocity( sMass_interior, 0*u.M_sun, 
                                                deproj_dist_m, 0*u.m)
        ########################################################################


        ########################################################################
        # Calculate the rotational velocities at a radius and the dark matter 
        # mass interior to that radius along with the associated errors.
        #-----------------------------------------------------------------------
        dmMass_interior, dmMass_interior_err = calc_dark_matter( mass_interior, 
                                                                 mass_interior_err, 
                                                                 sMass_interior)

        dmVel_rot, dmVel_rot_err = calc_velocity( dmMass_interior, 
                                                  dmMass_interior_err, 
                                                  deproj_dist_m, 0*u.m)
        ########################################################################

        
        ########################################################################
        # Append the corresponding values to their respective arrays to write to 
        # the roatation curve file.  The quantities are stripped of their units 
        # at this stage in the algorithm because astropy Column objects cannot 
        # be created with quantities that have dimensions.  The respective 
        # dimensions are added back when the Column objects are added to the 
        # astropy QTable.
        #-----------------------------------------------------------------------
        list_dict['radius'].append( deproj_dist.value)
        #list_dict['radius_err'].append( deproj_dist_kpc_err.value)

        list_dict['max_vel'].append( max_vel.value)
        list_dict['max_vel_err'].append( 1/np.sqrt(max_vel_ivar.value))
        list_dict['min_vel'].append( min_vel.value)
        list_dict['min_vel_err'].append( 1/np.sqrt(min_vel_ivar.value))
        list_dict['avg_vel'].append( avg_vel.value)
        list_dict['avg_vel_err'].append( avg_vel_err.value)
        list_dict['vel_diff'].append( vel_diff.value)
        list_dict['vel_diff_err'].append( vel_diff_err.value)

        list_dict['M_tot'].append( mass_interior.value)
        list_dict['M_tot_err'].append( mass_interior_err.value)

        list_dict['M_star'].append( sMass_interior.value)
        list_dict['star_vel'].append( sVel_rot.value)
        list_dict['star_vel_err'].append( sVel_rot_err.value)

        list_dict['DM'].append( dmMass_interior.value)
        list_dict['DM_err'].append( dmMass_interior_err.value)
        list_dict['DM_vel'].append( dmVel_rot.value)
        list_dict['DM_vel_err'].append( dmVel_rot_err.value)
        ########################################################################
    
    return list_dict



################################################################################
################################################################################
################################################################################



def calc_avg_vel( max_vel, min_vel, max_vel_ivar, min_vel_ivar):
    '''
    Calculate the average rotational velocity at an annulus, its inverse 
    variance, and the difference in the rotational velocities and its error.

    Parameters:
    ===========

    max_vel : float
        Maximum H-alpha velocity within the annulus

    min_vel : float
        Minimum H-alpha velocity within the annulus

    max_vel_ivar : float
        Inverse variance of the maximum H-alpha velocity

    min_vel_ivar : float
        Inverse variance of the minimum H-alpha velocity


    Returns:
    ========

    avg_vel : float
        Average extreme Halpha velocity within the annulus

    avg_vel_err : float
        Error in the average extreme Halpha velocity

    vel_diff : float
        Difference between the magnitudes of the minimum and maximum velocities

    vel_diff_err : float
        Error in the velocity difference

    '''

    ############################################################################
    # Average velocity at annulus
    #---------------------------------------------------------------------------
    avg_vel = 0.5 * ( max_vel + abs( min_vel))
    avg_vel_err = 0.5*np.sqrt((1/max_vel_ivar) + (1/min_vel_ivar))
    ############################################################################


    ############################################################################
    # Difference between the magnitudes of the minimum and maximum velocities
    #---------------------------------------------------------------------------
    vel_diff = abs( max_vel - abs( min_vel))
    vel_diff_err = np.sqrt((1/max_vel_ivar) + (1/min_vel_ivar))
    ############################################################################


    return avg_vel, avg_vel_err, vel_diff, vel_diff_err



################################################################################
################################################################################
################################################################################



def calc_mass_interior(velocity, velocity_err, radius, radius_err):
    '''
    Calculate the mass interior to the given radius assuming circular motion:
    M(r) = v(r)^2 * r / G


    Parameters:
    ===========

    velocity : float
        Rotational velocity at given radius

    velocity_err : float
        Uncertainty in the rotational velocity

    radius : float
        Radius within which to calculate mass

    radius_err : float
        Uncertainty in the radius


    Returns:
    ========

    mass : float
        Mass interior to radius in units of solar masses

    mass_err : float
        Uncertainty in the mass in units of solar masses

    '''

    
    mass = velocity.to('m/s')**2 * radius / const.G
    mass = mass.to('M_sun')

    mass_err = mass * np.sqrt( (2 * velocity_err / velocity)**2 \
                               + (radius_err / radius)**2 \
                               + (const.G.uncertainty * const.G.unit / const.G)**2)

    return mass, mass_err



################################################################################
################################################################################
################################################################################



def calc_stellar_mass(mass, mass_err2,  sMass_density_array, sMass_density_array_err, annulus_spaxels):
    '''
    Calculate the stellar mass interior to a radius.

    NOTE: 'sMass_density_array,' in units of log10( M_sun / spaxel**2), is data 
          pulled from the Pipe3d datacube.  Because it is in units proportional 
          to log( spaxel**(-2)) and in the system of spaxels a spaxel can be 
          given units of one, 'sMass_density_array' is essentially in units of 
          log10( M_sun).  To find the stellar mass interior to a radius that 
          satisfies the 'annulus_spaxels' condition, 10 is raised to the 
          power of 'sMass_density_array' for a given spaxel.


    Parameters:
    ===========

    mass : float
        Total stellar mass interior to previous annuli

    sMass_density_array : numpy array of shape (n,n)
        Masked array of the stellar mass density

    annulus_spaxels : numpy boolean array of shape (n,n)
        Values are true for those spaxels which are part of the current annulus


    Returns:
    ========

    mass : float
        Total stellar mass interior to current annulus


    '''

    ############################################################################
    # Calculate stellar mass interior to current annulus
    #---------------------------------------------------------------------------
    for spaxel in sMass_density_array[ annulus_spaxels]:
        try:
            if np.isfinite(spaxel.physical):
                mass += spaxel.physical
        except AttributeError:
            if np.isfinite(spaxel):
                mass += spaxel
    
    for spaxel in sMass_density_array_err[ annulus_spaxels]:
        try:
            if np.isfinite(spaxel.physical):
                mass_err2 += spaxel.physical**2
        except AttributeError:
            if np.isfinite(spaxel):
                mass_err2 += spaxel**2
  

    ############################################################################

    return mass, mass_err2



################################################################################
################################################################################
################################################################################



def calc_velocity( mass, mass_err, radius, radius_err):
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
    velocity = np.sqrt( const.G * ( mass.to('kg')) / radius)
    velocity = velocity.to('km/s')
    ############################################################################

    ############################################################################
    # Calculate velocity error
    #---------------------------------------------------------------------------
    velocity_err = 0.5 * velocity * np.sqrt( (const.G.uncertainty * const.G.unit / const.G)**2 \
                                             + (mass_err / mass)**2 \
                                             + (radius_err / radius)**2)

 

    velocity_err = velocity_err.to('km/s')
    ############################################################################

    return velocity, velocity_err



################################################################################
################################################################################
################################################################################



def calc_dark_matter( tot_mass, tot_mass_err, stellar_mass):
    '''
    Calculate the rotational velocities at a radius and the dark matter mass 
    interior to that radius along with the errors associated with them.
    

    Parameters:
    ===========

    tot_mass : float
        Total mass interior to current radius

    tot_mass_err : float
        Uncertainty in the total interior mass

    stellar_mass : float
        Stellar mass interior to current radius


    Returns:
    ========

    DM_mass : float
        Dark matter mass in units of solar masses

    DM_mass_err : float
        Uncertainty in the dark matter mass

    '''

    ############################################################################
    # Calculate dark matter mass interior to current annulus
    #---------------------------------------------------------------------------
    DM_mass = tot_mass - stellar_mass
    ############################################################################


    ############################################################################
    # Calculate the uncertainty in the dark matter mass
    #
    # NOTE: There is no error given for sMass_density_interior, so the error in 
    #       the dark matter mass is the same as the error in the total mass.
    #---------------------------------------------------------------------------
    DM_mass_err = tot_mass_err
    ############################################################################

    return DM_mass, DM_mass_err



################################################################################
################################################################################
################################################################################



def put_data_in_QTable( lists, gal_ID, 
                        center_flux, center_flux_err, 
                        frac_masked_spaxels):
    '''
    Unpacks the lists dictionary and creates a QTable of the data.


    Parameters:
    ===========

    lists : dictionary
        Dictionary of output data lists 

    gal_ID : string
        MaNGA galaxy ID information in the format [plate]-[IFU]

    center_flux : float
        r-band flux in galaxy's luminous center spaxel

    center_flux_err : float
        Uncertainty in the r-band flux at the galaxy's luminous center spaxel

    frac_masked_spaxels : float
        Fraction of spaxels which were masked


    Returns:
    ========

    data_table : Astropy table
        Table with all the lists as columns

    '''

    ############################################################################
    # Convert lists in dictionary to Astropy table column objects
    #---------------------------------------------------------------------------
    dist_col = Column( lists['radius'])
    #dist_err_col = Column( lists['radius_err'])
    max_vel_col = Column( lists['max_vel'])
    max_vel_err_col = Column( lists['max_vel_err'])
    min_vel_col = Column( lists['min_vel'])
    min_vel_err_col = Column( lists['min_vel_err'])
    rot_curve_vel_avg_col = Column( lists['avg_vel'])
    rot_curve_vel_avg_err_col = Column( lists['avg_vel_err'])
    rot_curve_vel_diff_col = Column( lists['vel_diff'])
    rot_curve_vel_diff_err_col = Column( lists['vel_diff_err'])

    totMass_interior_col = Column( lists['M_tot'])
    totMass_interior_err_col = Column( lists['M_tot_err'])

    sMass_interior_col = Column( lists['M_star'])
    sVel_rot_col = Column( lists['star_vel'])
    sVel_rot_err_col = Column( lists['star_vel_err'])

    dmMass_interior_col = Column( lists['DM'])
    dmMass_interior_err_col = Column( lists['DM_err'])
    dmVel_rot_col = Column( lists['DM_vel'])
    dmVel_rot_err_col = Column( lists['DM_vel_err'])


    gal_ID_col = Column( [gal_ID])
    flux_center_col = Column( [center_flux])
    flux_center_err_col = Column( [center_flux_err])
    frac_masked_spaxels_col = Column( [frac_masked_spaxels])
    ############################################################################


    ############################################################################
    # Create data table
    #---------------------------------------------------------------------------
    data_table = QTable([ dist_col *  u.kpc,
                         #dist_err_col * u.kpc,
                         max_vel_col * ( u.km / u.s),
                         max_vel_err_col * ( u.km / u.s),
                         min_vel_col * ( u.km / u.s),
                         min_vel_err_col * ( u.km / u.s),
                         rot_curve_vel_avg_col * ( u.km / u.s),
                         rot_curve_vel_avg_err_col * ( u.km / u.s),
                         sMass_interior_col * (u.M_sun),
                         sVel_rot_col * ( u.km / u.s),
                         sVel_rot_err_col * ( u.km / u.s),
                         dmMass_interior_col * ( u.Msun),
                         dmMass_interior_err_col * ( u.M_sun),
                         dmVel_rot_col * ( u.km / u.s),
                         dmVel_rot_err_col * ( u.km / u.s),
                         totMass_interior_col * ( u.M_sun),
                         totMass_interior_err_col * ( u.M_sun),
                         rot_curve_vel_diff_col * ( u.km / u.s),
                         rot_curve_vel_diff_err_col * ( u.km / u.s)],
                names = ['deprojected_distance',
                         #'deprojected_distance_error',
                         'max_velocity',
                         'max_velocity_error',
                         'min_velocity',
                         'min_velocity_error',
                         'rot_vel_avg',
                         'rot_vel_avg_error',
                         'sMass_interior',
                         'sVel_rot',
                         'sVel_rot_error',
                         'dmMass_interior',
                         'dmMass_interior_error',
                         'dmVel_rot',
                         'dmVel_rot_error',
                         'mass_interior',
                         'mass_interior_error',
                         'rot_curve_diff',
                         'rot_curve_diff_error'])
    ############################################################################


    ############################################################################
    # Create table containing general galaxy information
    #---------------------------------------------------------------------------
    gal_stats = QTable([ gal_ID_col,
                        flux_center_col * ( u.erg / ( u.s * u.cgs.cm**2)),
                        flux_center_err_col * ( u.erg / ( u.s * u.cgs.cm**2)),
                        frac_masked_spaxels_col],
               names = ['gal_ID',
                        'center_flux',
                        'center_flux_error',
                        'frac_masked_spaxels'])
    ############################################################################

    return data_table, gal_stats