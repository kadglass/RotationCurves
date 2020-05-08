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

'''
import matplotlib
matplotlib.use('TKAgg')
'''
import matplotlib.pyplot as plt


import numpy as np, numpy.ma as ma

import warnings
warnings.simplefilter('ignore', np.RankWarning)

import os.path

from astropy.io import fits
from astropy.coordinates import SkyCoord
from astropy.table import QTable, Column
import astropy.units as u

from Pipe3D_rotation_curve_functions import build_mask, find_rot_curve, put_data_in_QTable
from Pipe3D_rotation_curve_plottingFunctions import plot_vband_image, plot_Ha_vel, plot_rot_curve, plot_mass_curve, plot_diagnostic_panel



###############################################################################
###############################################################################
###############################################################################

def extract_data( file_name):
    """Open the MaNGA .fits file and extract data.


    PARAMETERS
    ==========
    
    file_name : string
        [DATA RELEASE]-manga-[PLATE]-[IFUID].Pipe3D.cube.fits.gz

    
    RETURNS
    =======

    target_galaxy : boolean
        Boolean indicating whether or not this object was a galaxy target in 
        MaNGA.  True = galaxy target; False = other target

    data_quality : boolean
        Boolean representing the quality bitmask of the MaNGA data.  True = 
        data is good; False = do not use.
    
    Ha_vel : numpy array of shape (N,N)
        H-alpha velocity field in units of km/s

    Ha_vel_err : numpy array of shape (N,N)
        error in the H-alpha velocity field in units of km/s

    v_band : numpy array of shape (N,N)
         visual-band flux in units of 10^-17 ers/s/cm^2

    v_band_err : numpy array of shape (N,N)
        error in the visual band flux in units of 10^-17 erg/s/cm^2

    sMass_density : numpy array of shape (N,N)
        stellar mass density per square pixel (units of log(Msun/spaxel^2))

    gal_ra : float
        righthand ascension of galaxy in units of degrees

    gal_dec : float
        declination of galaxy in units of degrees
    """


    main_file = fits.open( file_name)

    ssp = main_file[1].data
    flux_elines = main_file[3].data
    org_hdr = main_file[0].header

    main_file.close()

    ###########################################################################
    # NOTE: The visual band fluxes are multiplied by 10^-16 as stated in the
    #       units of the MaNGA Data Model.
    #
    #       <https://data.sdss.org/datamodel/files/MANGA_PIPE3D/MANGADRP_VER
    #       /PIPE3D_VER/PLATE/manga.Pipe3D.cube.html#hdu1>
    ###########################################################################
    v_band = ssp[0]  # in units of erg / s / cm^2
    v_band_err = ssp[4]  # in units of erg / s / cm^2
    sMass_density = ssp[19] * u.dex( u.M_sun) # in units of log10( Msun / spaxel**2)

    Ha_vel = flux_elines[102]  # in units of km/s
    Ha_vel_err = flux_elines[330]  # in units of km/s

    gal_ra = org_hdr['OBJRA']
    gal_dec = org_hdr['OBJDEC']

    target_galaxy = True
    MaNGA_galaxy_target = org_hdr['MNGTARG1']
    if MaNGA_galaxy_target == 0:
        target_galaxy = False

    data_quality = True
    DRP_3D_quality = org_hdr['DRP3QUAL']
    if DRP_3D_quality > 10000:
        data_quality = False

    return target_galaxy, data_quality, Ha_vel, Ha_vel_err, v_band, v_band_err, \
           sMass_density, gal_ra, gal_dec



###############################################################################
###############################################################################
###############################################################################


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



################################################################################
################################################################################
################################################################################


def calc_rot_curve( Ha_vel, Ha_vel_err, v_band, v_band_err, sMass_density,
                   axis_ratio, phi_EofN_deg, z, gal_ID, plot_diagnostics=True, 
                   IMAGE_DIR=None, IMAGE_FORMAT='eps', num_masked_gal=0):
    '''
    Calculate the rotation curve (rotational velocity as a funciton of
    deprojected distance) of the galaxy.  In addition, a galaxy statistics file 
    is created that contains information about the galaxy's center luminosity, 
    stellar mass processed in the algorithm, the errors associated with these 
    quantities as available, and gal_ID, which identifies the galaxy by SDSS 
    data release and MaNGA plate and fiber ID.


    Parameters:
    ===========

    Ha_vel : numpy array of shape (n,n)
        H-alpha velocity field data

    Ha_vel_err : numpy array of shape (n,n)
        Error in the H-alpha velocity field data

    v_band : numpy array of shape (n,n)
        Visual band flux data

    v_band_err : numpy array of shape (n,n)
        Error in the visual band flux data

    sMass_density : numpy array of shape (n,n)
        Stellar mass density per spaxel

    axis_ratio : float
        Ratio of the galaxy's minor axis to major axis as obtained via a sersic 
        fit of the galaxy

    phi_EofN_deg : float
        Angle (east of north) of rotation in the 2-D, observational plane

        NOTE: east is 'left' per astronomy convention

    z : float
        Galaxy redshift as calculated by the shift in H-alpha flux

    gal_ID : string
        [DATA RELEASE]-[PLATE]-[IFUID]

    plot_diagnostics : boolean
        Flag to determine whether or not to plot the various plot diagnostics.  
        Default is True (plot all figures).

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

    data_table : astropy QTable
        Contains the deprojected distance; maximum, minimum, average, stellar, 
        and dark matter velocities at that radius; difference between the 
        maximum and minimum velocities; and the stellar, dark matter, and total 
        mass interior to that radius as well as the errors associated with each 
        quantity as available

    gal_stats : astropy QTable
        Contains single-valued columns of the center luminosity and its error, 
        the stellar mass processed, and the fraction of spaxels masked

    num_masked_gal : float
        Cumulative number of completely masked galaxies
    '''

    ###########################################################################
    # Create a mask for the data arrays. The final mask is applied to all data 
    # arrays extracted from the .fits file.
    #--------------------------------------------------------------------------
    mask_data = build_mask( Ha_vel_err, v_band, v_band_err, sMass_density)

    num_masked_spaxels = np.sum(mask_data) - np.sum(v_band == 0)
    frac_masked_spaxels = num_masked_spaxels/np.sum(np.logical_not(v_band == 0))

    masked_Ha_vel = ma.masked_where( mask_data, Ha_vel)
    masked_Ha_vel_err = ma.masked_where( mask_data, Ha_vel_err)
    masked_sMass_density = ma.masked_where( mask_data, sMass_density)
    '''
    #--------------------------------------------------------------------------
    # Show the created mask where yellow points represent masked data points.
    #--------------------------------------------------------------------------
    plt.figure(1)
    plt.imshow( mask_data)
    plt.show()
    plt.close()
    '''
    ###########################################################################
    

    ###########################################################################
    # DIAGNOSTICS:
    #--------------------------------------------------------------------------
    if plot_diagnostics:
        #----------------------------------------------------------------------
        # Plot visual-band image
        #----------------------------------------------------------------------
        plot_vband_image( v_band, gal_ID, IMAGE_DIR=IMAGE_DIR, 
                          IMAGE_FORMAT=IMAGE_FORMAT)

        if IMAGE_DIR is None:
            plt.show()
        #----------------------------------------------------------------------
        # Plot H-alpha velocity field before systemic redshift subtraction. 
        # Galaxy velocities vary from file to file, so vmin and vmax will have 
        # to be manually adjusted for each galaxy before reshift subtraction.
        #----------------------------------------------------------------------
        plot_Ha_vel( Ha_vel, gal_ID, 
                     IMAGE_DIR=IMAGE_DIR, FOLDER_NAME='/unmasked_Ha_vel/', 
                     IMAGE_FORMAT=IMAGE_FORMAT, FILENAME_SUFFIX='_Ha_vel_raw.')

        if IMAGE_DIR is None:
            plt.show()
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
    inclination_angle = np.arccos( axis_ratio)

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

    '''
    ###########################################################################
    # Print the angle of rotation in the 2-D observational plane as taken from
    #    the NSA catalog.
    #--------------------------------------------------------------------------
    phi_deg = ( 90 - phi_EofN_deg / u.deg) * u.deg
    print('phi:', phi_deg)
    ###########################################################################
    '''

    ###########################################################################
    # Preserve original v_band image for plotting in the 'diagnostic_panel'
    #    image.
    #--------------------------------------------------------------------------
    v_band_raw = v_band.copy()
    ###########################################################################


    ###########################################################################
    # If 'unmasked_data' was set to False by all of the 'Ha_vel' data being
    #    masked after correcting for the angle of inclination, set all of the 
    #    data arrays to be -1.
    #--------------------------------------------------------------------------
    if not unmasked_data:
        lists = {'radius':[-1], #'radius_err':[-1],
                 'max_vel':[-1], 'max_vel_err':[-1],
                 'min_vel':[-1], 'min_vel_err':[-1],
                 'avg_vel':[-1], 'avg_vel_err':[-1],
                 'vel_diff':[-1], 'vel_diff_err':[-1],
                 'M_tot':[-1], 'M_tot_err':[-1],
                 'M_star':[-1], 
                 'star_vel':[-1], 'star_vel_err':[-1],
                 'DM':[-1], 'DM_err':[-1],
                 'DM_vel':[-1], 'DM_vel_err':[-1]}

        center_flux = -1
        center_flux_err = -1

        num_masked_gal += 1

        print("ALL DATA POINTS FOR THE GALAXY ARE MASKED!!!")
    ###########################################################################


    ###########################################################################
    # If there is unmasked data in the data array, execute the function as
    #    normal.
    #--------------------------------------------------------------------------
    else:
        lists, center_flux, center_flux_err, masked_vel_contour_plot = find_rot_curve( z, 
                                                                                       mask_data, 
                                                                                       v_band, 
                                                                                       v_band_err, 
                                                                                       Ha_vel, 
                                                                                       masked_Ha_vel, 
                                                                                       masked_Ha_vel_err, 
                                                                                       masked_sMass_density, 
                                                                                       optical_center, 
                                                                                       phi_EofN_deg, 
                                                                                       axis_ratio)

        if plot_diagnostics:
            ####################################################################
            # Plot the H-alapha velocity field within the annuli
            #-------------------------------------------------------------------
            plot_Ha_vel( masked_vel_contour_plot, gal_ID, 
                         IMAGE_DIR=IMAGE_DIR, 
                         FOLDER_NAME='/collected_velocity_fields/', 
                         IMAGE_FORMAT=IMAGE_FORMAT, 
                         FILENAME_SUFFIX='_collected_vel_field.')

            if IMAGE_DIR is None:
                plt.show()
            ####################################################################


            ####################################################################
            # Plot H-alpha velocity field with redshift subtracted.
            #-------------------------------------------------------------------
            plot_Ha_vel( masked_Ha_vel, gal_ID, 
                         IMAGE_DIR=IMAGE_DIR, FOLDER_NAME='/masked_Ha_vel/', 
                         IMAGE_FORMAT=IMAGE_FORMAT, 
                         FILENAME_SUFFIX='_Ha_vel_field.')

            if IMAGE_DIR is None:
                plt.show()
            ####################################################################
    ############################################################################


    ###########################################################################
    # Convert the data arrays into astropy Column objects and then add those
    #    Column objects to an astropy QTable.
    #
    # NOTE: 'gal_stats' contains general statistics about luminosity and
    #       stellar mass for the entire galaxy
    #--------------------------------------------------------------------------
    data_table, gal_stats = put_data_in_QTable(lists, gal_ID, center_flux, 
                                               center_flux_err, frac_masked_spaxels)
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
    '''
    # Print the systemic velocity (taken from the most luminous point in the 
    # galaxy), and absolute maximum and minimum velocities in the entire numpy 
    # n-D array after the velocity subtraction.
    #
    print("Systemic Velocity:", sys_vel)
    print("Global MAX:", global_max)
    print("Global min:", global_min)
    print("inclination_angle:", np.degrees( inclination_angle))
    '''


    if unmasked_data and plot_diagnostics:

        #######################################################################
        # Rotational velocity as a function of deprojected radius.
        #----------------------------------------------------------------------
        plot_rot_curve( gal_ID, data_table, 
                        IMAGE_DIR=IMAGE_DIR, IMAGE_FORMAT=IMAGE_FORMAT)

        if IMAGE_DIR is None:
            plt.show()
        #######################################################################


        #######################################################################
        # Plot cumulative mass as a function of deprojected radius.
        #----------------------------------------------------------------------
        plot_mass_curve( IMAGE_DIR, IMAGE_FORMAT, gal_ID, data_table)

        if IMAGE_DIR is None:
            plt.show()
        #######################################################################


        #######################################################################
        # Plot a two by two paneled image containging the entire 'Ha_vel' 
        # array, the masked version of this array, 'masked_Ha_vel,' the masked
        # 'vel_contour_plot' array containing ovals of the data points 
        # processed in the algorithm, and the averaged max and min rotation 
        # curves alongside the stellar mass rotation curve.
        #----------------------------------------------------------------------
        plot_diagnostic_panel( IMAGE_DIR, IMAGE_FORMAT, gal_ID, v_band_raw, 
                               masked_Ha_vel, masked_vel_contour_plot, 
                               data_table)

        if IMAGE_DIR is None:
            plt.show()
        #######################################################################


    return data_table, gal_stats, num_masked_gal



###############################################################################
###############################################################################
###############################################################################


def write_rot_curve( data_table, gal_stats, gal_ID, ROT_CURVE_MASTER_FOLDER):
    '''
    Write tables with an ascii-commented header to a .txt file in ecsv format,
    specified by the LOCAL_PATH, ROT_CURVE_MASTER_FOLDER, output_data_folder,
    and the output_data_name variables.


    Parameters:
    ===========
    
    data_table : astropy QTable of shape (m,p)
        Contains the deprojected distance, maximum and minimum velocities at 
        that radius, average luminosities for each half of the galaxy at that 
        radius, luminosity interior to the radius, and the stellar mass interior 
        to the radius

    gal_stats : astropy QTable of shape (1,n)
        Contains single valued columns of the processed and unprocessed 
        luminosities and corresponding masses, the luminosity at the center of 
        the galaxy, and the fraction of masked spaxels

    gal_ID : string
        [DATA RELEASE]-[PLATE]-[IFUID]

    LOCAL_PATH : string 
        Path of the main script

    ROT_CURVE_MASTER_FOLDER : string
        Name in which to store the data subfolders into
    '''

    data_table.write( ROT_CURVE_MASTER_FOLDER + gal_ID + '_rot_curve_data.txt',
                      format='ascii.ecsv', overwrite=True)

    gal_stats.write( ROT_CURVE_MASTER_FOLDER + gal_ID + '_gal_stat_data.txt',
                     format='ascii.ecsv', overwrite=True)



###############################################################################
###############################################################################
###############################################################################


def write_master_file( manga_plate_master, manga_fiberID_master,
                       nsa_plate_master, nsa_fiberID_master, nsa_mjd_master,
                       nsa_gal_idx_master, nsa_ra_master, nsa_dec_master,
                       nsa_axes_ratio_master, nsa_phi_master, nsa_z_master,
                       nsa_mStar_master, nsa_rabsmag_master, LOCAL_PATH, 
                       MASTER_FILENAME='master_file.txt'):
    '''
    Create the master file containing identifying information about each
    galaxy.  The output file of this function determines the structure of the
    master file that will contain the best fit parameters for the fitted
    rotation curve equations.

    
    Parameters:
    ===========

    manga_plate_master : numpy array of shape (n,1)
        master list containing the MaNGA plate information for each galaxy

    manga_fiberID_master : numpy array of shape (n,1)
        master list containing the MaNGA fiber ID information for each galaxy

    nsa_plate_master : numpy array of shape (n,1)
        master list containing the NSA plate information for each galaxy

    nsa_fiberID_master : numpy array of shape (n,1)
        master list containing the NSA fiber ID information for each galaxy

    nsa_mjd_master : numpy array of shape (n,1)
        master list containing the NSA MJD information for each galaxy

    nsa_gal_idx_master : numpy array of shape (n,1)
        master list containing the NSA ID information for each galaxy

        NOTE: This is NOT the index number of the galaxy in the NSA catalog.

    nsa_ra_master : numpy array of shape (n,1)
        master list containing the NSA righthand ascension values for each galaxy

    nsa_dec_master : numpy array of shape (n,1)
        master list containing the NSA declination values for each galaxy

    nsa_axes_ratio_master : numpy array of shape (n,1)
        master list containing the NSA axis ratio for each galaxy

    nsa_phi_master : numpy array of shape (n,1)
        master list containing the NSA rotation angle for each galaxy

    nsa_z_master : numpy array of shape (n,1)
        master list containing the NSA redshift for each galaxy

    nsa_mStar_master : numpy array of shape (n,1)
        master list containing the NSA stellar mass estimate for each galaxy

    nsa_rabsmag_master : numpy array of shape (n,1)
        master list containing the NSA SDSS r-band absolute magnitude for each 
        galaxy

    LOCAL_PATH : string
        the directory path of the main script file

    MASTER_FILENAME : string
        File name of master file.  Default is 'master_file.txt'
    '''

    ###########################################################################
    # Convert the master data arrays into Column objects to add to the master
    #    data table.
    #--------------------------------------------------------------------------
    manga_plate_col = Column( manga_plate_master)
    manga_fiberID_col = Column( manga_fiberID_master)
    nsa_plate_col = Column( nsa_plate_master)
    nsa_fiberID_col = Column( nsa_fiberID_master)
    nsa_mjd_col = Column( nsa_mjd_master)
    nsa_gal_idx_col = Column( nsa_gal_idx_master)
    nsa_ra_col = Column( nsa_ra_master)
    nsa_dec_col = Column( nsa_dec_master)
    nsa_axes_ratio_col = Column( nsa_axes_ratio_master)
    nsa_phi_col = Column( nsa_phi_master)
    nsa_z_col = Column( nsa_z_master)
    nsa_mStar_col = Column( nsa_mStar_master)
    nsa_rabsmag_col = Column( nsa_rabsmag_master)
    ###########################################################################


    master_filename = LOCAL_PATH + MASTER_FILENAME

    if not os.path.isfile( master_filename):
        ########################################################################
        # Add the column objects to an astropy QTable.
        #-----------------------------------------------------------------------
        master_table = QTable([ manga_plate_col,
                                manga_fiberID_col,
                                nsa_plate_col,
                                nsa_fiberID_col,
                                nsa_mjd_col,
                                nsa_gal_idx_col,
                                nsa_ra_col * u.degree,
                                nsa_dec_col * u.degree,
                                nsa_axes_ratio_col,
                                nsa_phi_col * u.degree,
                                nsa_z_col,
                                nsa_mStar_col,
                                nsa_rabsmag_col],
                       names = ['MaNGA_plate',
                                'MaNGA_fiberID',
                                'NSA_plate',
                                'NSA_fiberID',
                                'NSA_MJD',
                                'NSA_index',
                                'NSA_RA',
                                'NSA_DEC',
                                'NSA_ba',
                                'NSA_phi',
                                'NSA_redshift',
                                'NSA_Mstar',
                                'NSA_rabsmag'])
        ########################################################################
    else:
        ########################################################################
        # Read in current master_file.txt file
        #-----------------------------------------------------------------------
        master_table = QTable.read( master_filename, format='ascii.ecsv')
        ########################################################################


        ########################################################################
        # Build reference dictionary of plate, fiberID combinations
        #-----------------------------------------------------------------------
        index_dict = {}

        for i in range( len( manga_plate_master)):
            index_dict[ (manga_plate_master[i], manga_fiberID_master[i])] = i
        ########################################################################


        ########################################################################
        # Update column values in master_table
        #-----------------------------------------------------------------------
        for i in range( len( master_table)):
            col_idx = index_dict[ (master_table['MaNGA_plate'][i], master_table['MaNGA_fiberID'][i])]

            master_table['NSA_plate'][i] = nsa_plate_col[col_idx]
            master_table['NSA_fiberID'][i] = nsa_fiberID_col[col_idx]
            master_table['NSA_MJD'][i] = nsa_mjd_col[col_idx]
            master_table['NSA_index'][i] = nsa_gal_idx_col[col_idx]
            master_table['NSA_RA'][i] = nsa_ra_col[col_idx] * u.degree
            master_table['NSA_DEC'][i] = nsa_dec_col[col_idx] * u.degree
            master_table['NSA_ba'][i] = nsa_axes_ratio_col[col_idx]
            master_table['NSA_phi'][i] = nsa_phi_col[col_idx] * u.degree
            master_table['NSA_redshift'][i] = nsa_z_col[col_idx]
            master_table['NSA_Mstar'][i] = nsa_mStar_col[col_idx]
            master_table['NSA_rabsmag'][i] = nsa_rabsmag_col[col_idx]
        ########################################################################


    ###########################################################################
    # Write the master data file in ecsv format.
    #--------------------------------------------------------------------------
    master_table.write( master_filename, format='ascii.ecsv', overwrite=True)
    ###########################################################################