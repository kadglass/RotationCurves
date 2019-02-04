#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Created on Mon January 28 2019
@author: Jacob A. Smith
@version: 2.0

Analyzes mainly face-on galaxies to compute the rotation curve (rotational
velocity as a function of radius).
Also writes the rotation curve data to a .txt file in ecsv format and
statistical data about the galaxy (luminosity of the center spaxel, stellar
mass processed for each galaxy in this algorithm, and the errors associated
with these quantities as available) is also written to a .txt file in ecsv
format.
The function write_master_file creates a .txt file in ecsv format with
identifying information about each galaxy as well as specific parameters taken
from the NSA catalog in calculating the rotation curve for the galaxy.

To download the MaNGA .fits files used to calculate the rotation curves for
these galaxies, see the instructions for each data release via the following
links:

http://www.sdss.org/dr14/manga/manga-data/data-access/
http://www.sdss.org/dr15/manga/manga-data/data-access/
"""
import matplotlib.pyplot as plt
import gc

import math
import numpy as np, numpy.ma as ma
import warnings
warnings.simplefilter('ignore', np.RankWarning)

from astropy.io import fits, ascii
from astropy.coordinates import SkyCoord
from astropy.table import QTable, Column
import astropy.units as u
import astropy.constants as const

######################
# Declare constants. #
######################

H_0 = 100 * (u.km / ( u.s * u.Mpc))  # Hubble's Constant in units of km /s /Mpc

MANGA_FIBER_DIAMETER = 9.69627362219072e-06   # angular fiber diameter
                                              #   (2") in radians


def extract_data( file_name):
    """Open the MaNGA .fits file and extract data.

    @param:
        file_name:
            a string representation of the galaxy in question in the
            following format:
                [DATA RELEASE]-manga-[PLATE]-[IFUID].Pipe3D.cube.fits.gz

    @return:
        Ha_vel:
            an n-D numpy array containing the H-alpha velocity field data

        Ha_vel_err:
            an n-D numpy array containing the error in the H-alpha velocity
            field data

        v_band:
            an n-D numpy array containing the visual-band flux data

        v_band_err:
            an n-D numpy array containing the error in the visual band flux
            data

        sMass_density:
            an n-D numpy array containing the stellar mass density per square
            pixel

        manga_plate:
            int representation of the manga plate number of observation

        manga_fiberID:
            int representation of the manga fiber ID of observation

        gal_ra:
            the galaxy in question's righthand ascension

        gal_dec:
            the galaxy in question's declination
    """
    main_file = fits.open( file_name)

    ssp = main_file[1].data
    flux_elines = main_file[3].data
    org_hdr = main_file[0].header

    main_file.close()

    ###########################################################################
    # NOTE: The visual band fluxes are multiplied by 10E-16 as stated in the
    #       units of the MaNGA Data Model.
    #
    #       <https://data.sdss.org/datamodel/files/MANGA_PIPE3D/MANGADRP_VER
    #       /PIPE3D_VER/PLATE/manga.Pipe3D.cube.html#hdu1>
    ###########################################################################
    v_band = ssp[0] * 10E-16  # in units of erg / s / cm^2
    v_band_err = ssp[4] * 10E-16  # in units of erg / s / cm^2
    sMass_density = ssp[19] * u.dex( u.M_sun) # in units of
                                                # log10( Msun / spaxel**2)

    Ha_vel = flux_elines[102]  # in units of km/s
    Ha_vel_err = flux_elines[330]  # in units of km/s

    manga_plate = org_hdr['PLATEID']
    manga_fiberID = org_hdr['IFUDSGN']
    gal_ra = org_hdr['OBJRA']
    gal_dec = org_hdr['OBJDEC']

    return Ha_vel, Ha_vel_err, v_band, v_band_err, sMass_density, \
                manga_plate, manga_fiberID, gal_ra, gal_dec


def match_to_NSA( gal_ra, gal_dec, cat_coords):
    """Match the galaxy in question to the NSA Catalog and extract the NSA
    Catalog index to.

    NOTE: This function takes a substantially long time; it must match the
    RA and DEC of the galaxy in question to a singluar entry in the entire
    NASA-Sloan-Atlas catalog (510MB).

    @param:
        gal_ra:
            the galaxy in question's righthand ascension

        gal_dec:
            the galaxy in question's declination

        cat_coords:
            SkyCoord master list containing all of the righthand ascension
            and declination for the galaxies contained in the NSA
            catalog

    @return:
        idx:
            the NSA catalog integer index of the galaxy in question
    """
    gal_coord = SkyCoord( ra = gal_ra * u.degree,
                         dec = gal_dec * u.degree)

    idx = gal_coord.match_to_catalog_sky( cat_coords)[0]

    ###########################################################################
    # DIAGNOSTICS:
    #--------------------------------------------------------------------------
#    print("gal_coord:", gal_coord)
#    print("NSA index of galaxy:", idx)
    ###########################################################################

    return idx


def calc_rot_curve( Ha_vel, Ha_vel_err, v_band, v_band_err, sMass_density,
                   axes_ratio, phi_EofN_deg, zdist, zdist_err, gal_ID,
                   IMAGE_DIR):
    """Calculates the rotation curve (rotational velocity as a funciton of
    deprojected distance) of the galaxy in question. In addition a galaxy
    statistics file is created containing information about the galaxy's
    center luminosity, stellar mass processed in the algorithm, the errors
    associated with these quantities as available, and a string, gal_ID, that
    identifies the galaxy by SDSS data release and MaNGA plate and fiber ID.

    @param:
        Ha_vel:
            an n-D numpy array containing the H-alpha velocity field data

        Ha_vel_err:
            an n-D numpy array containing the error in the H-alpha velocity
            field data

        v_band:
            an n-D numpy array containing the visual band flux data

        v_band_err:
            an n-D numpy array containing the error in the visual band flux
            data

        sMass_density:
            an n-D numpy array containing the stellar mass density per square
            pixel

        axes_ratio:
            float representation of the ratio of the galaxy's minor axis to the
            major axis as obtained via a sersic fit of the galaxy

        phi_EofN_deg:
            float representation of the angle (east of north) of rotation in
            the 2-D, observational plane

            NOTE: east is 'left' per astronomy convention

        zdist:
            float representation of a measure of the distance to the galaxy in
            question as calculated by the shift in H-alpha flux

        zdist_err:
            float representation of the error in zdist measurement

        gal_ID:
            a string representation of the galaxy in question in the
            following format: [DATA RELEASE]-[PLATE]-[IFUID]

        IMAGE_DIR:
            string representation of the file path that pictures of the fitted
            rotation curves are saved to

    @return:
        data_table:
            an astropy QTable containing the deprojected distance; maximum,
            minimum, average, stellar, and dark matter velocities at that
            radius; difference between the maximum and minimum velocities;
            and the stellar, dark matter, and total mass interior to that
            radius as well as the errors associated with each quantity as
            available

        gal_stats:
            an astropy QTable containing single-valued columns of the center
            luminosity and its error and the stellar mass processed
    """
    ###########################################################################
    # Create a mask for the data arrays. Each of the boolean conditions are
    #    explained below. The final mask is applied to all data arrays
    #    extracted from the .fits file.
    #--------------------------------------------------------------------------
    # Ha_vel_err = 0 mask
    Ha_vel_err_boolean = Ha_vel_err == 0

    # v_band = 0 mask
    v_band_boolean = v_band == 0

    # v_band_err = 0 mask
    v_band_err_boolean = v_band_err == 0

    # sMass_density = np.nan mask
    sMass_density_boolean = np.isnan( sMass_density)


    # masking condition that combines all the above booleans
    mask_data_a = np.logical_or( Ha_vel_err_boolean, v_band_boolean)
    mask_data_b = np.logical_or( v_band_err_boolean, sMass_density_boolean)
    mask_data = np.logical_or( mask_data_a, mask_data_b)

    masked_Ha_vel = ma.masked_where( mask_data, Ha_vel)
    masked_Ha_vel_err = ma.masked_where( mask_data, Ha_vel_err)
    masked_sMass_density = ma.masked_where( mask_data, sMass_density)
    #--------------------------------------------------------------------------
    # Show the created mask where yellow points represent masked data points.
    #--------------------------------------------------------------------------
#    plt.figure(1)
#    plt.imshow( mask_data)
#    plt.show()
#    plt.close()
    ###########################################################################


    ###########################################################################
    # Create a meshgrid for all coordinate points based on the dimensions of
    # the H-alpha velocity numpy array.
    #--------------------------------------------------------------------------
    array_length = Ha_vel.shape[0]  # y-coordinate distance
    array_width = Ha_vel.shape[1]  # x-coordinate distance

    X_RANGE = np.arange(0, array_width, 1)
    Y_RANGE = np.arange(0, array_length, 1)
    X_COORD, Y_COORD = np.meshgrid( X_RANGE, Y_RANGE)
    ###########################################################################


    ###########################################################################
    # Initialize the rotation curve arrays that will store the rotation curve
    #    data.
    #--------------------------------------------------------------------------
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
    ###########################################################################


    ###########################################################################
    # DIAGNOSTICS:
    #--------------------------------------------------------------------------
    # Print the center of the galaxy as calculated from the most luminous
    #    point.
    #
#    print("Center: (%d, %d)" % (x_center, y_center))
    #--------------------------------------------------------------------------
    # Plot visual-band image.
    #
    vband_image = plt.figure(2)
    plt.title( gal_ID + ' Visual Band Image (RAW)')
    plt.imshow( v_band, origin='lower')

    cbar = plt.colorbar( ticks = np.linspace(  0, v_band.max(), 6))
    cbar.ax.tick_params( direction='in', color='white')
    cbar.set_label(r'Visual Band Flux [10E-17 erg s$^{-1}$ cm$^{-2}$]')

    ax = vband_image.add_subplot(111)
    plt.xticks( np.arange( 0, array_width, 10))
    plt.yticks( np.arange( 0, array_length, 10))
    plt.tick_params( axis='both', direction='in', color='white')
    ax.yaxis.set_ticks_position('both')
    ax.xaxis.set_ticks_position('both')
    plt.xlabel(r'$\Delta \alpha$ (arcsec)')
    plt.ylabel(r'$\Delta \delta$ (arcsec)')

    plt.savefig( IMAGE_DIR + "/unmasked_v_band/" + gal_ID + \
                "_v_band_raw.png", format='eps')
#    plt.show()
    plt.cla()
    plt.clf()
    plt.close( vband_image)
    gc.collect()
    #--------------------------------------------------------------------------
    # Plot H-alpha velocity field before systemic redshift
    #   subtraction. Galaxy velocities vary from file to file, so vmin and vmax
    #   will have to be manually adjusted for each galaxy before reshift
    #   subtraction.
    #
    vmin_bound = 0
    vmax_bound = np.max( Ha_vel)
    cbar_ticks = np.linspace( vmin_bound, vmax_bound, 11, dtype='int')

    Ha_vel_field_raw_fig = plt.figure(3)
    plt.title( gal_ID + r' H$\alpha$ Velocity Field (RAW)')
    plt.imshow( Ha_vel, cmap='bwr', origin='lower',
               vmin = vmin_bound, vmax = vmax_bound)

    cbar = plt.colorbar( ticks = cbar_ticks)
    cbar.ax.tick_params( direction='in')
    cbar.set_label(r'$V_{ROT}$ [$km s^{-1}$]')

    ax = Ha_vel_field_raw_fig.add_subplot(111)
    plt.xticks( np.arange( 0, array_width, 10))
    plt.yticks( np.arange( 0, array_length, 10))
    plt.tick_params( axis='both', direction='in')
    ax.yaxis.set_ticks_position('both')
    ax.xaxis.set_ticks_position('both')
    plt.xlabel(r'$\Delta \alpha$ (arcsec)')
    plt.ylabel(r'$\Delta \delta$ (arcsec)')

    plt.savefig( IMAGE_DIR + "/unmasked_Ha_vel/" + gal_ID + \
                "_Ha_vel_raw.png", format='eps')
#    plt.show()
    plt.cla()
    plt.clf()
    plt.close( Ha_vel_field_raw_fig)
    gc.collect()
    ###########################################################################


    ###########################################################################
    # Determine optical center via the max luminosity in the visual band.
    #--------------------------------------------------------------------------
    optical_center = np.argwhere( v_band.max() == v_band)

    x_center = optical_center[0][ 1]
    y_center = optical_center[0][ 0]
    ###########################################################################


    ###########################################################################
    # Subtract the systemic velocity from data points without the mask and then
    #    multiply the velocities by sin( inclination angle) to account for the
    #    galaxy's inclination affecting the rotational velocity.
    #
    # In addition, repeat the same calculations for the unmasked 'Ha_vel'
    #    array. This is for plotting purposes only within 'panel_fig.'
    #--------------------------------------------------------------------------
    sys_vel = masked_Ha_vel[ y_center, x_center]
    inclination_angle = np.arccos( axes_ratio)

    masked_Ha_vel -= sys_vel
    masked_Ha_vel /= np.sin( inclination_angle)

    Ha_vel[ ~masked_Ha_vel.mask] -= sys_vel
    Ha_vel[ ~masked_Ha_vel.mask] /= np.sin( inclination_angle)
    ###########################################################################


    ###########################################################################
    # Find the global max and global min of 'masked_Ha_vel' to use in graphical
    #    analysis.
    #
    # NOTE: If the entire data array is masked, 'global_max' and 'global_min'
    #       cannot be calculated. It has been found that if the
    #       'inclination_angle' is 0 degrees, the entire 'Ha_vel' array is
    #       masked. An if-statement tests this case, and sets 'unmasked_data'
    #       to False if there is no max/min in the array.
    #--------------------------------------------------------------------------
    global_max = np.max( masked_Ha_vel)
    global_min = np.min( masked_Ha_vel)

    unmasked_data = True

    if str( global_max) == '--':
        unmasked_data = False
        global_max = 0.1
        global_min = -0.1
    ###########################################################################


    ###########################################################################
    # Initialization code to draw the elliptical annuli and to normalize the
    #    2D-arrays for the max and min velocity so as to check for anomalous
    #    data.
    #--------------------------------------------------------------------------
    phi_elip = math.radians( 90 - ( phi_EofN_deg / u.deg)) * u.rad

    x_diff = X_COORD - x_center
    y_diff = Y_COORD - y_center

    ellipse = ( x_diff*np.cos( phi_elip) - y_diff*np.sin( phi_elip))**2 \
            + ( x_diff*np.sin( phi_elip) + y_diff*np.cos( phi_elip))**2 \
                  / ( axes_ratio)**2

    vel_contour_plot = np.zeros(( len(X_RANGE), len(Y_RANGE)))
    ###########################################################################


    ###########################################################################
    # Print the angle of rotation in the 2-D observational plane as taken from
    #    matching data from the MaNGA catalog to the NSA catalog.
    #--------------------------------------------------------------------------
#    phi_deg = ( 90 - phi_EofN_deg / u.deg) * u.deg
#    print("phi:", phi_deg)
    ###########################################################################


    ###########################################################################
    # If 'unmasked_data' was set to False by all of the 'Ha_vel' data being
    #    masked after correcting for the angle of inclination, append -1s to
    #    all of the data arrays and skip to creating the astropy Column objects
    #    for the astropy QTable.
    #--------------------------------------------------------------------------
    if not unmasked_data:
        rot_curve_dist.append( -1)

        rot_curve_max_vel.append( -1)
        rot_curve_max_vel_err.append( -1)
        rot_curve_min_vel.append( -1)
        rot_curve_min_vel_err.append( -1)
        rot_curve_vel_avg.append( -1)
        rot_curve_vel_avg_err.append( -1)
        rot_curve_vel_diff.append( -1)
        rot_curve_vel_diff_err.append( -1)

        totMass_interior_curve.append( -1)
        totMass_interior_curve_err.append( -1)

        sMass_interior_curve.append( -1)
        sVel_rot_curve.append( -1)
        sVel_rot_curve_err.append( -1)

        dmMass_interior_curve.append( -1)
        dmMass_interior_curve_err.append( -1)
        dmVel_rot_curve.append( -1)
        dmVel_rot_curve_err.append( -1)

        lum_center = -1
        lum_center_err = -1

        print("ALL DATA POINTS FOR THE GALAXY ARE MASKED!!!")
    ###########################################################################


    ###########################################################################
    # If there is unmasked data in the data array, execute the function as
    #    normal.
    #--------------------------------------------------------------------------
    else:
        #######################################################################
        # Convert pixel distance to physical distances in units of both
        #    kiloparsecs and centimeters.
        #----------------------------------------------------------------------
        dist_to_galaxy_kpc = ( zdist * const.c.to('km/s') / H_0).to('kpc')
        dist_to_galaxy_kpc_err = np.sqrt( (const.c.to('km/s') / H_0)**2 \
                                         * zdist_err**2 )
        dist_to_galaxy_cm = dist_to_galaxy_kpc.to( u.cgs.cm)
        dist_to_galaxy_cm_err = dist_to_galaxy_kpc_err.to( u.cgs.cm)

        pix_scale_factor = dist_to_galaxy_kpc * np.tan( MANGA_FIBER_DIAMETER)
        pix_scale_factor_err = np.sqrt( ( np.tan( MANGA_FIBER_DIAMETER))**2 \
                                       * dist_to_galaxy_kpc_err)

#        print("dist_to_galaxy_kpc:", dist_to_galaxy_kpc)
#        print("dist_to_galaxy_kpc_err:", dist_to_galaxy_kpc_err)
#        print("dist_to_galaxy_cm:", dist_to_galaxy_cm)
#        print("dist_to_galaxy_cm_err:", dist_to_galaxy_cm_err)
#        print("pix_scale_factor:", pix_scale_factor)
#        print("pix_scale_factor_err:", pix_scale_factor_err)
        #######################################################################


        #######################################################################
        # Extract the flux from the center of the galaxy and convert the
        #    measurement into luminosity in units of solar luminosity.
        #----------------------------------------------------------------------
        flux_center = v_band[ y_center, x_center] \
                                        * ( u.erg / ( u.s * u.cgs.cm**2))
        flux_center_error = v_band_err[ y_center, x_center] \
                                        * ( u.erg / ( u.s * u.cgs.cm**2))

        lum_center = (flux_center \
          * 4 * np.pi * dist_to_galaxy_cm**2).to('W') / const.L_sun
        lum_center_err = (flux_center_error \
          * 4 * np.pi * dist_to_galaxy_cm**2).to('W') / const.L_sun

#        print("flux_center:", flux_center)
#        print("flux_center_error:", flux_center_error)
#        print("lum_center:", lum_center)
#        print("lum_center_err:", lum_center_err)
#        #######################################################################


        #######################################################################
        # Initialize the stellar mass surface density interior to an annulus to
        #    be 0 solar masses.
        #----------------------------------------------------------------------
        sMass_interior = 0 * ( u.M_sun)
        #######################################################################


        #######################################################################
        # Find the first data point along the galaxy's semi-major axis where
        #    'v_band' equals zero therefore finding the point to signal to stop
        #    taking data points for the rotation curve. Set this point to -999
        #    as a flag value.
        #----------------------------------------------------------------------
        phi_edge = phi_EofN_deg.to('rad')
        slope = -1 * (np.cos( phi_edge) / np.sin( phi_edge))
        y_intercept = y_center - slope * x_center

        x_edge_pos = x_center
        y_edge_pos = y_center
        edge_pos = False
        while not edge_pos:
            x_temp_pos = x_edge_pos + 1
            y_temp_pos = int( slope * x_temp_pos + y_intercept)

            if ( x_temp_pos >= array_width) or \
              ( y_temp_pos < 0) or ( y_temp_pos >= array_length) \
              or ( v_band[ y_temp_pos, x_temp_pos] == 0):
                edge_pos = True

            else:
                x_edge_pos = x_temp_pos
                y_edge_pos = y_temp_pos

        x_edge_neg = x_center
        y_edge_neg = y_center
        edge_neg = False
        while not edge_neg:
            x_temp_neg = x_edge_neg - 1
            y_temp_neg = int( slope * x_temp_neg + y_intercept)

            if ( x_temp_neg < 0) or \
              ( y_temp_neg < 0) or ( y_temp_neg >= array_length) \
              or ( v_band[ y_temp_neg, x_temp_neg] == 0):
                edge_neg = True

            else:
                x_edge_neg = x_temp_neg
                y_edge_neg = y_temp_neg


        v_band[ y_edge_pos, x_edge_pos] = -999
        v_band[ y_edge_neg, x_edge_neg] = -999
        #######################################################################


        #######################################################################
        # While the data point signaling to stop data collection is not within
        #    'pix_between_annuli,' extract the maximum and minimum velocity at
        #    an annulus and the stellar mass interior to that annulus.
        #`~   ~   ~   ~   ~   ~   ~   ~   ~   ~   ~   ~   ~   ~   ~   ~   ~   ~
        dR = 2
        R = 2
        valid_data = True
        while valid_data:
            deproj_dist_kpc = R * pix_scale_factor
            deproj_dist_kpc_err = np.sqrt( R**2 * pix_scale_factor_err**2)
            deproj_dist_m = deproj_dist_kpc.to('m')
            deproj_dist_m_err = deproj_dist_kpc.to('m')
            ###################################################################
            # Define an eliptical annulus and check if either of the edge
            #    points are within that annulus.
            #
            # NOTE: Although if the edge point is within 'pix_between_annuli'
            #       and thus 'valid_data' is set to False, the current
            #       iteration of the loop still completes as intended. The test
            #       for 'valid_data' is included in this block to keep the code
            #       together.
            #------------------------------------------------------------------
            pix_between_annuli = np.logical_and(
                    (R-dR)**2 <= ellipse,
                    ellipse < R**2)

            if np.any( v_band[ pix_between_annuli] == -999):
                valid_data = False
            ###################################################################


            ###################################################################
            # Find the coordinates of the max/min velocity for a given annulus
            #    and normalize theta such that the location for the max/min
            #    corresponds with an angle of 0.
            #
            # If there is no max/min velocity point (i.e. all eligible data
            #    points are masked), then set the velocities to 'np.nan.'
            #
            # If there is more than one max or more than one min velocity at
            #    the given annulus, then use the first one found. The first
            #     velocity is used because the point at which it comes from is
            #    not relavent. Rather, it is only the velocity itself that is
            #    relavent to this project.
            #------------------------------------------------------------------
            max_vel_point = np.argwhere(
                masked_Ha_vel[ pix_between_annuli].max() == masked_Ha_vel)
            min_vel_point = np.argwhere(
                masked_Ha_vel[ pix_between_annuli].min() == masked_Ha_vel)

            # ! # ! # ! # ! # ! # ! # ! # ! # ! # ! # ! # ! # ! # ! # ! # ! # !
            # NOTE: Only the length of 'max_vel_point' is tested because if
            #       the maximum of 'masked_Ha_vel' returns nothing, the
            #       minimum will also return nothing because there are only
            #       masked values.
            # ! # ! # ! # ! # ! # ! # ! # ! # ! # ! # ! # ! # ! # ! # ! # ! # !
            if len( max_vel_point) == 0:
                max_vel_at_annulus = np.nan * ( u.km / u.s)
                max_vel_at_annulus_err = np.nan * ( u.km / u.s)
                min_vel_at_annulus = np.nan * ( u.km / u.s)
                min_vel_at_annulus_err = np.nan * ( u.km / u.s)
                print("ALL DATA POINTS AT R=" + str( R) + " ANNULUS ARE MASKED!!!")

            else:
                # ! # ! # ! # ! # ! # ! # ! # ! # ! # ! # ! # ! # ! # ! # ! # !
                # NOTE: The max/min velocity coordinates are extracted in the
                #       [0][i] fashion because sometimes there can be more than
                #       one point that contains the max/min velocity at that
                #       annulus. Thus, the first point with the maximum
                #       velocity is extracted and separated into its x- and y-
                #       coordinates.
                # ! # ! # ! # ! # ! # ! # ! # ! # ! # ! # ! # ! # ! # ! # ! # !
                max_vel_point_x = max_vel_point[ 0][ 1]
                max_vel_point_y = max_vel_point[ 0][ 0]

                min_vel_point_x = min_vel_point[ 0][ 1]
                min_vel_point_y = min_vel_point[ 0][ 0]


                max_vel_at_annulus = masked_Ha_vel[
                        max_vel_point_y, max_vel_point_x] * ( u.km / u.s)
                max_vel_at_annulus_err = masked_Ha_vel_err[
                        max_vel_point_y, max_vel_point_x] * ( u.km / u.s)

                min_vel_at_annulus = masked_Ha_vel[
                        min_vel_point_y, min_vel_point_x] * ( u.km / u.s)
                min_vel_at_annulus_err = masked_Ha_vel_err[
                        min_vel_point_y, min_vel_point_x] * ( u.km / u.s)
            ###################################################################


            ###################################################################
            # This block calculates the average rotational velocity at an
            #    annulus, its error, differnce in the rotational velocities,
            #    its error, and the total mass interior to that annulus along
            #     with its error.
            #
            # NOTE: If the maximum and minimum velocities at an annulus were
            #       not found, 'max_vel_at_annulus' and 'min_vel_at_annulus'
            #       were set to 'np.nan.' If this is the case, all calculations
            #       below will also evaluate to 'np.nan.'
            #------------------------------------------------------------------
            avg_vel_at_annulus = ( max_vel_at_annulus +
                                  abs( min_vel_at_annulus)) / 2
            avg_vel_at_annulus_err = np.sqrt(
                    max_vel_at_annulus_err**2 + min_vel_at_annulus_err**2 )

            rot_vel_diff = abs(
                    max_vel_at_annulus - abs( min_vel_at_annulus))
            rot_vel_diff_err = np.sqrt(
                    max_vel_at_annulus_err**2 + min_vel_at_annulus**2)

            mass_interior = avg_vel_at_annulus.to('m/s')**2 \
                          * deproj_dist_m \
                          / const.G
            mass_interior = mass_interior.to('M_sun')
            mass_interior_err = np.sqrt(
                ((2 * avg_vel_at_annulus.to('m/s') * deproj_dist_m) \
                 / ( const.G * const.M_sun) )**2 \
                 * avg_vel_at_annulus_err.to('m/s')**2 \
              + ((-1 * avg_vel_at_annulus.to('m/s')**2 * deproj_dist_m) \
                 / ( const.G**2 * const.M_sun) )**2 \
                 * ( const.G.uncertainty * const.G.unit)**2 \
              + ((-1 * avg_vel_at_annulus.to('m/s')**2 * deproj_dist_m) \
                 / ( const.G * const.M_sun**2) )**2 \
                 * (const.M_sun.uncertainty * const.M_sun.unit)**2) \
              * ( u.M_sun)
            ###################################################################


            ###################################################################
            # This block calculates the stellar rotational velocity at a
            #    radius, its error, and the stellar mass interior to that
            #    radius.
            #
            # NOTE: 'masked_sMass_density,' in units of
            #       log10( M_sun / spaxe**2), is data pulled from the MaNGA
            #       datacube. Because it is in units proportional to
            #       log( spaxel**(-2)) and in the system of spaxels a spaxel
            #       can be given units of one, 'masked_sMass_density' is
            #       essentially in units of log10( M_sun). To find the stellar
            #       mass interior to a radius that satisfies the
            #       'pix_between_annuli' condition, 10 is raised to the power
            #       of 'masked_sMass_density' for a given spaxel.
            #------------------------------------------------------------------
            for spaxel in masked_sMass_density[ pix_between_annuli]:
                try:
                    sMass_interior += spaxel.physical

                except AttributeError:
                    pass

            sVel_rot = np.sqrt(
                    const.G \
                    * ( sMass_interior.to('kg')) \
                    / deproj_dist_m)
            sVel_rot = sVel_rot.to('km/s')

            sVel_rot_err = 1 / 2000 * \
            np.sqrt(
              ( (sMass_interior / u.M_sun) * const.M_sun) \
                / ( deproj_dist_m * const.G ) \
                * ( const.G.uncertainty * const.G.unit)**2 \
              + ( const.G * ( sMass_interior / u.M_sun)) \
                / ( deproj_dist_m * const.M_sun) \
                * ( const.M_sun.uncertainty * const.M_sun.unit)**2 \
              + ( const.G * ( sMass_interior / u.M_sun) * const.M_sun) \
                / ( deproj_dist_m**3) \
                * ( deproj_dist_m_err)**2 )
            sVel_rot_err = sVel_rot_err.to('km/s')
            ###################################################################


            ###################################################################
            # This block calculates the rotational velocities at a radius and
            #    the dark matter mass interior to that radius along with the
            #    errors associated with them.
            #
            # NOTE: If the total mass interior to a radius cannot be determined
            #       because the max/min velocities are set to 'np.nan,'
            #       'dmMass-' variables will follow suit and also return as
            #       'np.nan.'
            #
            # NOTE: There is no error in sMass_density_interior therefore the
            #       error in the dark matter mass is the same as the error in
            #       the total mass.
            #------------------------------------------------------------------
            dmMass_interior = mass_interior - sMass_interior
            dmMass_interior_err = mass_interior_err

            dmVel_rot = np.sqrt(
                    const.G
                    * ( dmMass_interior.to('kg')) \
                    / deproj_dist_m)
            dmVel_rot = dmVel_rot.to('km/s')

            dmVel_rot_err = 1 / 2000 * \
                np.sqrt(
                  ( (dmMass_interior / u.M_sun) * const.M_sun) \
                / ( deproj_dist_m * const.G ) \
                * ( const.G.uncertainty * const.G.unit)**2 \
              + ( const.G * const.M_sun) \
                / ( deproj_dist_m * ( dmMass_interior / u.M_sun)) \
                * ( dmMass_interior_err / u.M_sun)**2 \
              + ( const.G * ( dmMass_interior / u.M_sun)) \
                / ( deproj_dist_m * const.M_sun) \
                * ( const.M_sun.uncertainty * const.M_sun.unit)**2)
            ###################################################################


            ###################################################################
            # Append the corresponding values to their respective arrays to
            #    write to the roatation curve file. The quantities are stirpped
            #     of their units at this stage in the algorithm because astropy
            #    Column objects cannot be created with quantities that have
            #    dimensions. The respective dimensions are added back when the
            #    Column objects are added to the astropy QTable.
            #------------------------------------------------------------------
            rot_curve_dist.append( deproj_dist_kpc.value)

            rot_curve_max_vel.append( max_vel_at_annulus.value)
            rot_curve_max_vel_err.append( max_vel_at_annulus_err.value)
            rot_curve_min_vel.append( min_vel_at_annulus.value)
            rot_curve_min_vel_err.append( min_vel_at_annulus_err.value)
            rot_curve_vel_avg.append( avg_vel_at_annulus.value)
            rot_curve_vel_avg_err.append( avg_vel_at_annulus_err.value)
            rot_curve_vel_diff.append( rot_vel_diff.value)
            rot_curve_vel_diff_err.append( rot_vel_diff_err.value)

            totMass_interior_curve.append( mass_interior.value)
            totMass_interior_curve_err.append( mass_interior_err.value)

            sMass_interior_curve.append( sMass_interior.value)
            sVel_rot_curve.append( sVel_rot.value)
            sVel_rot_curve_err.append( sVel_rot_err.value)

            dmMass_interior_curve.append( dmMass_interior.value)
            dmMass_interior_curve_err.append( dmMass_interior_err.value)
            dmVel_rot_curve.append( dmVel_rot.value)
            dmVel_rot_curve_err.append( dmVel_rot_err.value)
            ###################################################################


            ###################################################################
            # The line below adds the pixels of the H-alpha velocity field
            #    analyzed in the current iteration of the algorithm to an image
            #    that plots all the pixels analyzed for a given galaxy.
            #------------------------------------------------------------------
            vel_contour_plot[ pix_between_annuli] = masked_Ha_vel[
                                                           pix_between_annuli]
            ###################################################################


            ###################################################################
            # DIAGNOSTICS:
            #------------------------------------------------------------------
            # Below are print statements that give information about the
            #    max/min and average velocities at an annulus, stellar mass,
            #    dark matter mass, and total mass along with the rotational
            #    velocities due to them. Errors are given for all quantites
            #    except 'sMass_interior' for which there exists no error.
            #------------------------------------------------------------------
#            print("---------------------------------------------------------")
#            print("R = ", R)
#            print("deproj_dist_m:", deproj_dist_m)
#            print("max_vel_at_annulus:", max_vel_at_annulus)
#            print("max_vel_at_annulus_err:", max_vel_at_annulus_err)
#            print("min_vel_at_annulus:", min_vel_at_annulus)
#            print("min_vel_at_annulus_err:", min_vel_at_annulus_err)
#            print("avg_vel_at_annulus:", avg_vel_at_annulus)
#            print("avg_vel_at_annulus_err:", avg_vel_at_annulus_err)
#            print("rot_vel_diff:", rot_vel_diff)
#            print("rot_vel_diff_err:", rot_vel_diff_err)
#            print("mass_interior:", mass_interior)
#            print("mass_interior_err:", mass_interior_err)
#            print("sMass_interior:", sMass_interior)
#            print("sVel_rot:", sVel_rot)
#            print("sVel_rot_err:", sVel_rot_err)
#            print("dmMass_interior:", dmMass_interior)
#            print("dmMass_interior_err:", dmMass_interior_err)
#            print("dmVel_rot:", dmVel_rot)
#            print("dmVel_rot_err:", dmVel_rot_err)
#            print("---------------------------------------------------------")
            #------------------------------------------------------------------
            # Plot the pixels at the current annulus.
            #
#            current_pix_fig = plt.figure(4)
#            plt.title('Pixels at ' + str(R - dR) + ' < R < ' + str(R))
#            plt.imshow( pix_between_annuli, origin='lower')
#
#            ax = current_pix_fig.add_subplot(111)
#            plt.xticks( np.arange( 0, array_width, 10))
#            plt.yticks( np.arange( 0, array_length, 10))
#            plt.tick_params( axis='both', direction='in')
#            ax.yaxis.set_ticks_position('both')
#            ax.xaxis.set_ticks_position('both')
#            plt.xlabel(r'$\Delta \alpha$ (arcsec)')
#            plt.ylabel(r'$\Delta \delta$ (arcsec)')
#
#            plt.show()
#            plt.close()
            ###################################################################


            ###################################################################
            # Increment the radius of the annuluus R by dR.
            #------------------------------------------------------------------
            R += dR
            ###################################################################
        # ~   ~   ~   ~   ~   ~   ~   ~   ~   ~   ~   ~   ~   ~   ~   ~   ~   ~
    ###########################################################################


    ###########################################################################
    # Convert the data arrays into astropy Column objects and then add those
    #    Column objects to an astropy QTable.
    #
    # NOTE: 'gal_stats' contains general statistics about luminosity and
    #       stellar mass for the entire galaxy
    #--------------------------------------------------------------------------
    dist_col = Column( rot_curve_dist)
    max_vel_col = Column( rot_curve_max_vel)
    max_vel_err_col = Column( rot_curve_max_vel_err)
    min_vel_col = Column( rot_curve_min_vel)
    min_vel_err_col = Column( rot_curve_min_vel_err)
    rot_curve_vel_avg_col = Column( rot_curve_vel_avg)
    rot_curve_vel_avg_err_col = Column( rot_curve_vel_avg_err)
    rot_curve_vel_diff_col = Column( rot_curve_vel_diff)
    rot_curve_vel_diff_err_col = Column( rot_curve_vel_diff_err)

    totMass_interior_col = Column( totMass_interior_curve)
    totMass_interior_err_col = Column( totMass_interior_curve_err)

    sMass_interior_col = Column( sMass_interior_curve)
    sVel_rot_col = Column( sVel_rot_curve)
    sVel_rot_err_col = Column( sVel_rot_curve_err)

    dmMass_interior_col = Column( dmMass_interior_curve)
    dmMass_interior_err_col = Column( dmMass_interior_curve_err)
    dmVel_rot_col = Column( dmVel_rot_curve)
    dmVel_rot_err_col = Column( dmVel_rot_curve_err)


    gal_ID_col = Column( [gal_ID])
    lum_center_col = Column( [lum_center])
    lum_center_err_col = Column( [lum_center_err])


    data_table = QTable([ dist_col *  u.kpc,
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

    gal_stats = QTable([ gal_ID_col,
                        lum_center_col * u.L_sun,
                        lum_center_err_col * u.L_sun],
               names = ['gal_ID',
                        'center_luminosity',
                        'center_luminosity_error'])
    ###########################################################################


    ###########################################################################
    # NOTE: All further statements with the exception of the return statement
    #       are used to give information on the terminating loop for data
    #       collection. Figures are generated that show the phi from the NSA
    #       Catalog, as well as the pixels used from the H-alpha velocity field
    #       to generate the min and max rotation curves. The caught, anomalous
    #       max and min for the while loop are also printed to verify the
    #       algorithm is working correctly.
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Print the systemic velocity (taken from the most luminous point in
    #    the galaxy), and absolute maximum and minimum velocities in the
    #   entire numpy n-D array after the velocity subtraction.
    #
#    print("Systemic Velocity:", sys_vel)
#    print("Global MAX:", global_max)
#    print("Global min:", global_min)
#    print("inclination_angle:", np.degrees( inclination_angle))
    #--------------------------------------------------------------------------
    # Plot H-alpha velocity field with redshift subtracted.
    # -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -
    vmax_bound = -1 * min( global_max, global_min)
    vmin_bound = min( global_max, global_min)
    cbar_ticks = np.linspace( vmin_bound, vmax_bound, 11, dtype='int')

    Ha_vel_field_fig = plt.figure(5)
    plt.title( gal_ID + r' H$\alpha$ Velocity Field')
    plt.imshow( masked_Ha_vel, cmap='bwr', origin='lower',
               vmin = vmin_bound, vmax = vmax_bound)

    cbar = plt.colorbar( ticks = cbar_ticks)
    cbar.ax.tick_params( direction='in')
    cbar.set_label(r'$V_{ROT}$ [$km s^{-1}$]')

    ax = Ha_vel_field_fig.add_subplot(111)
    plt.xticks( np.arange( 0, array_width, 10))
    plt.yticks( np.arange( 0, array_length, 10))
    plt.tick_params( axis='both', direction='in')
    ax.yaxis.set_ticks_position('both')
    ax.xaxis.set_ticks_position('both')
    plt.xlabel(r'$\Delta \alpha$ (arcsec)')
    plt.ylabel(r'$\Delta \delta$ (arcsec)')

    plt.savefig( IMAGE_DIR + "/masked_Ha_vel/" + gal_ID + \
                "_Ha_vel_field.png", format='eps')
#    plt.show()
    plt.cla()
    plt.clf()
    plt.close( Ha_vel_field_fig)
    gc.collect()
    #--------------------------------------------------------------------------
    # Ha velocity field collected though all iterations of the loop.
    # -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -
    vmax_bound = -1 * min( global_max, global_min)
    vmin_bound = min( global_max, global_min)
    cbar_ticks = np.linspace( vmin_bound, vmax_bound, 12)

    vel_field_collected_fig = plt.figure(6,
                                         figsize=(6, 6))
    plt.title( gal_ID + " " + r'H$\alpha$ Velocity Field Collected',
              fontsize=12)
    plt.imshow( vel_contour_plot, origin='lower',
       vmin = vmin_bound, vmax = vmax_bound, cmap='bwr')

    cbar = plt.colorbar( ticks = cbar_ticks)
    cbar.ax.tick_params( direction='in')
    cbar.set_label(r'$V_{ROT}$ [$kms^{-1}$]')

    ax = vel_field_collected_fig.add_subplot(111)
    plt.xticks( np.arange( 0, array_width, 10))
    plt.yticks( np.arange( 0, array_length, 10))
    plt.tick_params( axis='both', direction='in')
    ax.yaxis.set_ticks_position('both')
    ax.xaxis.set_ticks_position('both')
    plt.xlabel(r'$\Delta \alpha$ (arcsec)')
    plt.ylabel(r'$\Delta \delta$ (arcsec)')

    plt.savefig( IMAGE_DIR + "/collected_velocity_fields/" + gal_ID + \
                "_collected_vel_field.png", format='eps')
#    plt.show()
    plt.cla()
    plt.clf()
    plt.close( vel_field_collected_fig)
    gc.collect()
    #--------------------------------------------------------------------------
    # Rotational velocity as a function of deprojected radius.
    # -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -
    rot_curve_fig = plt.figure(7, figsize=(5, 5))
    plt.title( gal_ID + ' Rotation Curves')
    plt.plot( rot_curve_dist, rot_curve_max_vel, 'rs', markersize=5)
    plt.plot( rot_curve_dist, np.abs( rot_curve_min_vel), 'b^', markersize=5)
    plt.plot( rot_curve_dist, rot_curve_vel_avg, 'gp', markersize=7)
    plt.plot( rot_curve_dist, sVel_rot_curve, 'cD', markersize=4)
    plt.plot( rot_curve_dist, dmVel_rot_curve, 'kX', markersize=7)

    ax = rot_curve_fig.add_subplot(111)
    plt.tick_params( axis='both', direction='in')
    ax.yaxis.set_ticks_position('both')
    ax.xaxis.set_ticks_position('both')
    plt.xlabel('Deprojected Radius [kpc]')
    plt.ylabel(r'Rotational Velocity [$km s^{-1}$]')

    plt.savefig( IMAGE_DIR + "/rot_curves/" + gal_ID + "_rot_curve.png",
                format='eps')
#    plt.show()
    plt.cla()
    plt.clf()
    plt.close( rot_curve_fig)
    gc.collect()
    #--------------------------------------------------------------------------
    # Mass interior to a radius.
    # -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -
    mass_curve_fig = plt.figure(8, figsize=(5, 5))
    plt.title( gal_ID + ' Mass Curves')
    plt.plot( rot_curve_dist, totMass_interior_curve, 'gp', markersize=7)
    plt.plot( rot_curve_dist, sMass_interior_curve, 'cD', markersize=4)
    plt.plot( rot_curve_dist, dmMass_interior_curve, 'kX', markersize=7)

    ax = mass_curve_fig.add_subplot(111)
    plt.tick_params( axis='both', direction='in')
    ax.yaxis.set_ticks_position('both')
    ax.xaxis.set_ticks_position('both')
    plt.xlabel('Deprojected Radius [kpc]')
    plt.ylabel(r'Mass Interior [$M_{\odot}$]')

    plt.savefig( IMAGE_DIR + "/mass_curves/" + gal_ID + "_mass_curve.png",
                format='eps')
#    plt.show()
    plt.cla()
    plt.clf()
    plt.close( mass_curve_fig)
    gc.collect()
    ###########################################################################


    ###########################################################################
    # Plot a two by two paneled image containging the entire 'Ha_vel' array,
    #    the masked version of this array, 'masked_Ha_vel,' the
    #    'vel_contour_plot' array containing ovals of the data points processed
    #    in the algorithm, and the averaged max and min rotation curves
    #    alongside the stellar mass rotation curve,
    #--------------------------------------------------------------------------
    vmax_bound = -1 * min( global_max, global_min)
    vmin_bound = min( global_max, global_min)
    cbar_ticks = np.linspace( vmin_bound, vmax_bound, 13)

    panel_fig, (( Ha_vel_panel, mHa_vel_panel),
                ( contour_panel, rot_curve_panel)) = plt.subplots( 2, 2)
    panel_fig.set_figheight( 10)
    panel_fig.set_figwidth( 10)
    plt.suptitle( gal_ID + " Diagnostic Panel", y=1.05, fontsize=16)

    Ha_vel_im = Ha_vel_panel.imshow( Ha_vel, origin='lower',
                      vmin=vmin_bound, vmax=vmax_bound, cmap='bwr')
    Ha_vel_panel.set_title(r'Unmasked H$\alpha$ Velocity Field')
    Ha_cbar = plt.colorbar( Ha_vel_im, ax=Ha_vel_panel, ticks = cbar_ticks)
    Ha_cbar.ax.tick_params( direction='in')
    Ha_cbar.set_label(r'$V_{ROT}$ [$kms^{-1}$]')
    Ha_vel_panel.set_xlabel(r'$\Delta \alpha$ (arcsec)')
    Ha_vel_panel.set_ylabel(r'$\Delta \delta$ (arcsec)')
    Ha_vel_panel.set_xticks( np.arange( 0, array_width, 10))
    Ha_vel_panel.set_yticks( np.arange( 0, array_length, 10))
    Ha_vel_panel.xaxis.set_ticks_position('both')
    Ha_vel_panel.yaxis.set_ticks_position('both')
    Ha_vel_panel.tick_params( axis='both', direction='in')

    mHa_vel_im = mHa_vel_panel.imshow( masked_Ha_vel, origin='lower',
                      vmin=vmin_bound, vmax=vmax_bound, cmap='bwr')
    mHa_vel_panel.set_title(r'Masked H$\alpha$ Velocity Field')
    mHa_cbar = plt.colorbar( mHa_vel_im, ax=mHa_vel_panel, ticks = cbar_ticks)
    mHa_cbar.ax.tick_params( direction='in')
    mHa_cbar.set_label(r'$V_{ROT}$ [$kms^{-1}$]')
    mHa_vel_panel.set_xlabel(r'$\Delta \alpha$ (arcsec)')
    mHa_vel_panel.set_ylabel(r'$\Delta \delta$ (arcsec)')
    mHa_vel_panel.set_xticks( np.arange( 0, array_width, 10))
    mHa_vel_panel.set_yticks( np.arange( 0, array_length, 10))
    mHa_vel_panel.xaxis.set_ticks_position('both')
    mHa_vel_panel.yaxis.set_ticks_position('both')
    mHa_vel_panel.tick_params( axis='both', direction='in')

    contour_im = contour_panel.imshow( vel_contour_plot, origin='lower',
                      vmin=vmin_bound, vmax=vmax_bound, cmap='bwr')
    contour_panel.set_title(r'H$\alpha$ Velocity Field Collected')
    contour_cbar = plt.colorbar( contour_im, ax=contour_panel, ticks = cbar_ticks)
    contour_cbar.ax.tick_params( direction='in')
    contour_cbar.set_label(r'$V_{ROT}$ [$kms^{-1}$]')
    contour_panel.set_xlabel(r'$\Delta \alpha$ (arcsec)')
    contour_panel.set_ylabel(r'$\Delta \delta$ (arcsec)')
    contour_panel.set_xticks( np.arange( 0, array_width, 10))
    contour_panel.set_yticks( np.arange( 0, array_length, 10))
    contour_panel.xaxis.set_ticks_position('both')
    contour_panel.yaxis.set_ticks_position('both')
    contour_panel.tick_params( axis='both', direction='in')

    rot_curve_panel.plot( rot_curve_dist, rot_curve_vel_avg,
                         'gp', markersize=7)
    rot_curve_panel.set_title('Rotation Curves')
    rot_curve_panel.plot( rot_curve_dist, sVel_rot_curve, 'cD', markersize=4)
    rot_curve_panel.set_xlabel('Deprojected Radius [kpc]')
    rot_curve_panel.set_ylabel(r'Rotational Velocity [$km s^{-1}$]')
    rot_curve_panel.xaxis.set_ticks_position('both')
    rot_curve_panel.yaxis.set_ticks_position('both')
    rot_curve_panel.tick_params( axis='both', direction='in')

    panel_fig.tight_layout()

    plt.savefig( IMAGE_DIR + "/diagnostic_panels/" + gal_ID + \
                "_diagnostic_panel.png",
                format='eps')
#    plt.show()
    plt.cla()
    plt.clf()
    plt.close( panel_fig)
    gc.collect()
    ###########################################################################

    return data_table, gal_stats


def write_rot_curve( data_table, gal_stats,
                    gal_ID,
                    ROT_CURVE_MASTER_FOLDER,
                    ROT_CURVE_DATA_INDICATOR, GAL_STAT_DATA_INDICATOR):
    """Write data_table with an ascii-commented header to a .txt file
    specified by the LOCAL_PATH, ROT_CURVE_MASTER_FOLDER, output_data_folder,
    output_data_name, and ROT_CURVE_DATA_INDICATOR variables.

    @param:
        data_table:
            an astropy QTable containing the deprojected distance, maximum and
            minimum velocities at that radius, average luminosities for each
            half of the galaxy at that radius, luminosity interior to the
            radius, and the stellar mass interior to the radius

        gal_stats:
            an astropy QTable containing single valued columns of the processed
            and unprocessed luminosities and corresponding masses as well as
            the luminosity at the center of the galaxy in question

        gal_ID:
            a string representation of the galaxy in question in the
            following format: [DATA RELEASE]-[PLATE]-[IFUID]

        LOCAL_PATH: string specifying the path the main script is executed in

        ROT_CURVE_MASTER_FOLDER:
            name in which to store the data subfolders into

        ROT_CURVE_DATA_INDICATOR:
            the extension of the file before '.txt' that identifies the
            respective file as a rotation curve data file

        GAL_STAT_DATA_INDICATOR:
            the extension of the file before '.txt' that identifies the
            respective file as a galaxy statistics data file
    """
    ###########################################################################
    # Write the astropy QTables to text files in ecsv format.
    #--------------------------------------------------------------------------
    ascii.write( data_table, ROT_CURVE_MASTER_FOLDER + '/' + gal_ID + \
                ROT_CURVE_DATA_INDICATOR + '.txt',
                format = 'ecsv',
                overwrite = True)

    ascii.write( gal_stats, ROT_CURVE_MASTER_FOLDER + '/' + gal_ID + \
                GAL_STAT_DATA_INDICATOR + '.txt',
                format = 'ecsv',
                overwrite = True)
    ###########################################################################


def write_master_file( manga_plate_master, manga_fiberID_master,
                      manga_data_release_master,
                      nsa_plate_master, nsa_fiberID_master, nsa_mjd_master,
                      nsa_gal_idx_master, nsa_ra_master, nsa_dec_master,
                      nsa_axes_ratio_master, nsa_phi_master, nsa_zdist_master,
                      nsa_mStar_master,
                      LOCAL_PATH):
    """Create the master file containing identifying information about each
    galaxy. The output file of this function determines the structure of the
    master file that will contain the best fit parameters for the fitted
    rotation curve equations.

    @param:
        manga_plate_master:
            master list containing the MaNGA plate information for each galaxy

        manga_fiberID_master:
            master list containing the MaNGA fiber ID information for each
            galaxy

        manga_data_release_master:
            master list containing the MaNGA data release number for each
            galaxy

        nsa_plate_master:
            master list containing the NSA plate information for each galaxy

        nsa_fiberID_master:
            master list containing the NSA fiber ID information for each galaxy

        nsa_mjd_master:
            master list containing the NSA MJD information for each galaxy

        nsa_gal_idx_master:
            master list containing the matched index for each galaxy as
            calculated through the 'match' function above

        nsa_ra_master:
            master list containing all of the righthand ascension values for
            galaxies in the MaNGA survey

        nsa_dec_master:
            master list containing all of the declination values for galaxies
            in the MaNGA survey

        nsa_axes_ratio_master:
            master list containing all the axes ratios used in extracting the
            rotation curve

        nsa_phi_master:
            master list containing all the rotation angles of the galaxies used
            in extracting the rotation curve

        nsa_zdist_master:
            master list containing all the redshift distances of the galaxies
            used in extracting the rotation curve

        nsa_mStar_master:
            master list containing all the stellar mass estimates of the
            galaxies matched to the NSA catalog

        LOCAL_PATH:
            the directory path of the main script file
    """
    ###########################################################################
    # Convert the master data arrays into Column objects to add to the master
    #    data table.
    #--------------------------------------------------------------------------
    manga_plate_col = Column( manga_plate_master)
    manga_fiberID_col = Column( manga_fiberID_master)
    manga_data_release_col = Column( manga_data_release_master)
    nsa_plate_col = Column( nsa_plate_master)
    nsa_fiberID_col = Column( nsa_fiberID_master)
    nsa_mjd_col = Column( nsa_mjd_master)
    nsa_gal_idx_col = Column( nsa_gal_idx_master)
    nsa_ra_col = Column( nsa_ra_master)
    nsa_dec_col = Column( nsa_dec_master)
    nsa_axes_ratio_col = Column( nsa_axes_ratio_master)
    nsa_phi_col = Column( nsa_phi_master)
    nsa_zdist_col = Column( nsa_zdist_master)
    nsa_mStar_col = Column( nsa_mStar_master)
    ###########################################################################


    ###########################################################################
    # Add the column objects to an astropy QTable.
    #--------------------------------------------------------------------------
    master_table = QTable([ manga_plate_col,
                            manga_fiberID_col,
                            manga_data_release_col,
                            nsa_plate_col,
                            nsa_fiberID_col,
                            nsa_mjd_col,
                            nsa_gal_idx_col,
                            nsa_ra_col * u.degree,
                            nsa_dec_col * u.degree,
                            nsa_axes_ratio_col,
                            nsa_phi_col * u.degree,
                            nsa_zdist_col,
                            nsa_mStar_col],
                   names = ['MaNGA_plate',
                            'MaNGA_fiberID',
                            'MaNGA_data_release',
                            'NSA_plate',
                            'NSA_fiberID',
                            'NSA_MJD',
                            'NSA_index',
                            'NSA_RA',
                            'NSA_DEC',
                            'NSA_b/a',
                            'NSA_phi',
                            'NSA_zdist',
                            'NSA_mStar'])
    ###########################################################################


    ###########################################################################
    # Write the master data file in ecsv format.
    #--------------------------------------------------------------------------
    ascii.write( master_table, LOCAL_PATH + '/master_file.txt',
                format = 'ecsv',
                overwrite = True)
    ###########################################################################