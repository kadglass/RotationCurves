'''
Analyzes mainly face-on galaxies to compute the rotation curve (rotational
velocity as a function of radius).

Also writes the rotation curve data to a .txt file in ecsv format and
statistical data about the galaxy (luminosity of the center spaxel, stellar
mass processed for each galaxy in this algorithm, and the errors associated
with these quantities as available) is also written to a .txt file in ecsv
format.

The function write_master_file creates a .txt file in ecsv format with
identifying information about each galaxy as well as specific parameters taken
from the DRPall catalog in calculating the rotation curve for the galaxy.

To download the MaNGA .fits files used to calculate the rotation curves for
these galaxies, see the instructions for each data release via the following
links:

http://www.sdss.org/dr14/manga/manga-data/data-access/
http://www.sdss.org/dr15/manga/manga-data/data-access/
http://www.sdss.org/dr16/manga/manga-data/data-access/
'''


################################################################################
# IMPORT MODULES
################################################################################
import numpy as np
import numpy.ma as ma

import matplotlib.pyplot as plt

import warnings
warnings.simplefilter('ignore', np.RankWarning)

import os.path

from astropy.io import fits
from astropy.table import QTable, Column
import astropy.units as u

from DRP_rotation_curve_functions import build_mask, \
                                         find_rot_curve, \
                                         put_data_in_QTable

from DRP_rotation_curve_plottingFunctions import plot_rband_image, \
                                                 plot_Ha_vel, \
                                                 plot_rot_curve, \
                                                 plot_mass_curve, \
                                                 plot_diagnostic_panel
################################################################################



################################################################################
################################################################################
################################################################################

def extract_data( DRP_FOLDER, gal_ID):
    """
    Open the MaNGA .fits file and extract data.

    
    PARAMETERS
    ==========
    
    DRP_FOLDER : string
        Address to location of DRP data on computer system

    gal_ID : string
        '[PLATE]-[IFUID]' of the galaxy

    
    RETURNS
    =======
    
    Ha_vel : n-D numpy array
        H-alpha velocity field in units of km/s

    Ha_vel_ivar : n-D numpy array
        Inverse variance in the H-alpha velocity field in units of 1/(km/s)^2

    Ha_vel_mask : n-D numpy array
        Bitmask for the H-alpha velocity field

    r_band : n-D numpy array
        r-band flux in units of 1E-17 erg/s/cm^2/ang/spaxel

    r_band_ivar : n-D numpy array
        Inverse variance in the r-band flux in units of 
        1/(1E-17 erg/s/cm^2/ang/spaxel)^2
    """

    [plate, IFU] = gal_ID.split('-')
    file_name = DRP_FOLDER + plate + '/' + IFU + '/manga-' + gal_ID + '-MAPS-HYB10-GAU-MILESHC.fits.gz'
    
    if not os.path.isfile(file_name):
        print(gal_ID, 'data file does not exist.')
        return None, None, None, None, None, None, None, None, None

    cube = fits.open( file_name)

    r_band = cube['SPX_MFLUX'].data
    r_band_ivar = cube['SPX_MFLUX_IVAR'].data

    Ha_flux = cube['EMLINE_GFLUX'].data[18]
    
    Ha_vel = cube['EMLINE_GVEL'].data[18]
    Ha_vel_ivar = cube['EMLINE_GVEL_IVAR'].data[18]
    Ha_vel_mask = cube['EMLINE_GVEL_MASK'].data[18]

    Ha_sigma = cube['EMLINE_GSIGMA'].data[18]
    Ha_sigma_ivar = cube['EMLINE_GSIGMA_IVAR'].data[18]
    Ha_sigma_mask = cube['EMLINE_GSIGMA_MASK'].data[18]

    cube.close()

    return Ha_vel, Ha_vel_ivar, Ha_vel_mask, r_band, r_band_ivar, Ha_flux, Ha_sigma, Ha_sigma_ivar, Ha_sigma_mask


###############################################################################
###############################################################################
###############################################################################


def extract_Pipe3d_data( PIPE3D_FOLDER, gal_ID):
    '''
    Open the MaNGA Pipe3d .fits file and extract data.

    
    PARAMETERS
    ==========

    PIPE3D_FOLDER : string
        Address to location of Pipe3d data on computer system
    
    gal_ID : string
        '[PLATE]-[IFUID]' of the galaxy

    
    RETURNS
    =======

    sMass_density : n-D numpy array
        Stellar mass density map, in units of log(Msun/spaxel^2)
    '''

    [plate, IFU] = gal_ID.split('-')
    pipe3d_filename = PIPE3D_FOLDER + plate + '/manga-' + gal_ID + '.Pipe3D.cube.fits.gz'

    if not os.path.isfile(pipe3d_filename):
        print(gal_ID, 'Pipe3d data file does not exist.')
        return None

    main_file = fits.open( pipe3d_filename)
    ssp = main_file[1].data
    main_file.close()

    sMass_density = ssp[19] * u.dex( u.M_sun)

    return sMass_density


###############################################################################
###############################################################################
###############################################################################


def match_to_DRPall( gal_ID, DRPall_plateIFU):
    """
    Match the galaxy in question to the DRPall catalog and extract the galaxy's 
    index into DRPall.

    
    PARAMETERS
    ==========
    
    gal_ID : string
        [PLATE]-[IFU]

    DRPall_plateIFU : numpy array of shape (N,)
        DRPall list containing all of the [PLATE]-[IFU] for the galaxies 
        contained in the DRPall catalog

    
    RETURNS
    =======
    
    idx : integer
        The DRPall catalog integer index of the galaxy in question
    """

    idx = np.where(DRPall_plateIFU == gal_ID)

    return idx[0][0]



################################################################################
################################################################################
################################################################################


def calc_rot_curve( Ha_vel, Ha_vel_ivar, Ha_vel_mask, r_band, r_band_ivar,
                    sMass_density, axis_ratio, phi_EofN_deg, z, gal_ID,
                    IMAGE_DIR=None, IMAGE_FORMAT='eps', num_masked_gal=0):
    '''
    Calculate the rotation curve (rotational velocity as a funciton of
    deprojected distance) of the galaxy.  In addition, a galaxy statistics file 
    is created that contains information about the galaxy's center luminosity, 
    the errors associated with these quantities as available, and gal_ID, which 
    identifies the galaxy by SDSS data release and MaNGA plate and IFU.


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

    sMass_density : numpy array of shape (n,n)
        Stellar mass density map in units of log(Msun/spaxel^2)

    axis_ratio : float
        Ratio of the galaxy's minor axis to major axis as obtained via an 
        elliptical sersic fit of the galaxy

    phi_EofN_deg : float
        Angle (east of north) of rotation in the 2-D, observational plane

        NOTE: east is 'left' per astronomy convention

    z : float
        Galaxy redshift as calculated by the shift in H-alpha flux

    gal_ID : string
        [PLATE]-[IFU]

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
        Contains the deprojected distance; maximum, minimum, average, and dark 
        matter velocities at that radius; difference between the maximum and 
        minimum velocities; and the dark matter and total mass interior to that 
        radius as well as the errors associated with each quantity as available

    gal_stats : astropy QTable
        Contains single-valued columns of the center luminosity and its error 
        and the fraction of spaxels masked

    num_masked_gal : float
        Cumulative number of completely masked galaxies
    '''

    ###########################################################################
    # Create a mask for the data arrays. The final mask is applied to all data 
    # arrays extracted from the .fits file.
    #--------------------------------------------------------------------------
    data_mask = build_mask( Ha_vel_mask, sMass_density)

    num_masked_spaxels = np.sum(data_mask) - np.sum(r_band == 0)
    frac_masked_spaxels = num_masked_spaxels/np.sum(r_band != 0)

    mr_band = ma.array( r_band, mask=data_mask)
    mr_band_ivar = ma.array( r_band_ivar, mask=data_mask)

    mHa_vel = ma.array( Ha_vel, mask=data_mask)
    mHa_vel_ivar = ma.array( Ha_vel_ivar, mask=data_mask)

    msMass_density = ma.array( sMass_density, mask=data_mask)
    '''
    #--------------------------------------------------------------------------
    # Show the created mask where yellow points represent masked data points.
    #--------------------------------------------------------------------------
    plt.figure(1)
    plt.imshow( data_mask)
    plt.show()
    plt.close()
    '''
    ###########################################################################
    

    ############################################################################
    # DIAGNOSTICS:
    #---------------------------------------------------------------------------
    # Plot r-band image
    #---------------------------------------------------------------------------
    plot_rband_image( r_band, gal_ID, IMAGE_DIR=IMAGE_DIR, IMAGE_FORMAT=IMAGE_FORMAT)

    if IMAGE_DIR is None:
        plt.show()
    #---------------------------------------------------------------------------
    # Plot H-alpha velocity field before systemic redshift subtraction.  Galaxy 
    # velocities vary from file to file, so vmin and vmax will have to be 
    # manually adjusted for each galaxy before reshift subtraction.
    #-----------------------------------------------------------------------
    plot_Ha_vel( Ha_vel, gal_ID, 
                 IMAGE_DIR=IMAGE_DIR, FOLDER_NAME='/unmasked_Ha_vel/', 
                 IMAGE_FORMAT=IMAGE_FORMAT, FILENAME_SUFFIX='_Ha_vel_raw.')

    if IMAGE_DIR is None:
        plt.show()
    ############################################################################


    ############################################################################
    # Determine optical center via the max luminosity in the r-band.
    #---------------------------------------------------------------------------
    optical_center = np.unravel_index(ma.argmax(mr_band, axis=None), mr_band.shape)

    '''
    x_center = optical_center[0][ 1]
    y_center = optical_center[0][ 0]
    '''
    ############################################################################


    ############################################################################
    # Subtract the systemic velocity from data points without the mask and then
    # multiply the velocities by sin( inclination angle) to account for the
    # galaxy's inclination affecting the rotational velocity.
    #
    # In addition, repeat the same calculations for the unmasked 'Ha_vel' array.
    # This is for plotting purposes only within 'panel_fig.'
    #---------------------------------------------------------------------------
    sys_vel = mHa_vel[ optical_center]
    inclination_angle = np.arccos( axis_ratio)

    mHa_vel -= sys_vel
    mHa_vel /= np.sin( inclination_angle)

    Ha_vel -= sys_vel
    Ha_vel /= np.sin( inclination_angle)
    '''
    Ha_vel[ ~mHa_vel.mask] -= sys_vel
    Ha_vel[ ~mHa_vel.mask] /= np.sin( inclination_angle)
    '''
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
    global_max = np.max( mHa_vel)
    global_min = np.min( mHa_vel)

    unmasked_data = True

    if np.isnan( global_max):
        unmasked_data = False
        global_max = 0.1
        global_min = -0.1
    ############################################################################


    ############################################################################
    # Preserve original r-band image for plotting in the 'diagnostic_panel'
    # image.
    #---------------------------------------------------------------------------
    r_band_raw = r_band.copy()
    ############################################################################


    ############################################################################
    # If 'unmasked_data' was set to False by all of the 'Ha_vel' data being
    # masked after correcting for the angle of inclination, set all of the data 
    # arrays to be -1.
    #---------------------------------------------------------------------------
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
    ############################################################################


    ############################################################################
    # If there is unmasked data in the data array, execute the function as
    # normal.
    #---------------------------------------------------------------------------
    else:
        lists, center_flux, center_flux_err, mVel_contour_plot = find_rot_curve( z, 
                                                                                 data_mask, 
                                                                                 r_band, 
                                                                                 r_band_ivar,
                                                                                 Ha_vel, 
                                                                                 mHa_vel, 
                                                                                 mHa_vel_ivar, 
                                                                                 msMass_density, 
                                                                                 optical_center, 
                                                                                 phi_EofN_deg, 
                                                                                 axis_ratio)

        ########################################################################
        # Plot the H-alapha velocity field within the annuli
        #-----------------------------------------------------------------------
        plot_Ha_vel( mVel_contour_plot, gal_ID, 
                     IMAGE_DIR=IMAGE_DIR, FOLDER_NAME='/collected_velocity_fields/', 
                     IMAGE_FORMAT=IMAGE_FORMAT, FILENAME_SUFFIX='_collected_vel_field.')

        if IMAGE_DIR is None:
            plt.show()
        ########################################################################


        ########################################################################
        # Plot H-alpha velocity field with redshift subtracted.
        #-----------------------------------------------------------------------
        plot_Ha_vel( mHa_vel, gal_ID, 
                     IMAGE_DIR=IMAGE_DIR, FOLDER_NAME='/masked_Ha_vel/', 
                     IMAGE_FORMAT=IMAGE_FORMAT, FILENAME_SUFFIX='_Ha_vel_field.')

        if IMAGE_DIR is None:
            plt.show()
        ########################################################################
    ############################################################################


    ############################################################################
    # Convert the data arrays into astropy Column objects and then add those
    # Column objects to an astropy QTable.
    #
    # NOTE: 'gal_stats' contains general statistics about luminosity and stellar 
    #       mass for the entire galaxy
    #---------------------------------------------------------------------------
    data_table, gal_stats = put_data_in_QTable( lists, gal_ID, 
                                                center_flux, center_flux_err, 
                                                frac_masked_spaxels)
    ############################################################################


    ############################################################################
    # NOTE: All further statements with the exception of the return statement
    #       are used to give information on the terminating loop for data
    #       collection.  Figures are generated that show the phi from the NSA
    #       Catalog, as well as the pixels used from the H-alpha velocity field
    #       to generate the min and max rotation curves. The caught, anomalous
    #       max and min for the while loop are also printed to verify that the
    #       algorithm is working correctly.
    #---------------------------------------------------------------------------
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


    if unmasked_data:

        ########################################################################
        # Rotational velocity as a function of deprojected radius.
        #-----------------------------------------------------------------------
        plot_rot_curve( gal_ID, data_table, 
                        IMAGE_DIR=IMAGE_DIR, IMAGE_FORMAT=IMAGE_FORMAT)

        if IMAGE_DIR is None:
            plt.show()
        ########################################################################


        ########################################################################
        # Plot cumulative mass as a function of deprojected radius.
        #-----------------------------------------------------------------------
        plot_mass_curve( gal_ID, data_table, 
                         IMAGE_DIR=IMAGE_DIR, IMAGE_FORMAT=IMAGE_FORMAT)

        if IMAGE_DIR is None:
            plt.show()
        ########################################################################


        ########################################################################
        # Plot a two by two paneled image containging the entire 'Ha_vel' array, 
        # the masked version of this array, 'masked_Ha_vel,' the masked
        # 'vel_contour_plot' array containing ovals of the data points processed 
        # in the algorithm, and the averaged max and min rotation curves along 
        # with the stellar mass rotation curve.
        #-----------------------------------------------------------------------
        plot_diagnostic_panel( gal_ID, 
                               r_band_raw, 
                               mHa_vel, 
                               mVel_contour_plot, 
                               data_table, 
                               IMAGE_DIR=IMAGE_DIR, IMAGE_FORMAT=IMAGE_FORMAT)

        if IMAGE_DIR is None:
            plt.show()
        ########################################################################


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
        [PLATE]-[IFU]

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


def write_master_file( manga_plate_master, manga_IFU_master, NSAid_master, 
                       ra_master, dec_master, z_master,
                       axis_ratio_master, phi_master, 
                       mStar_master, rabsmag_master, 
                       LOCAL_PATH, MASTER_FILENAME):
    '''
    Create the master file containing identifying information about each
    galaxy.  The output file of this function determines the structure of the
    master file that will contain the best fit parameters for the fitted
    rotation curve equations.

    
    Parameters:
    ===========

    manga_plate_master : numpy array of shape (n,)
        master list containing the MaNGA plate information for each galaxy

    manga_IFU_master : numpy array of shape (n,)
        master list containing the MaNGA IFU information for each galaxy

    NSAid_master : numpy array of shape (n,)
        master list containing the unique NSA ID number for each galaxy

    ra_master : numpy array of shape (n,)
        master list containing the RA values for each galaxy

    dec_master : numpy array of shape (n,)
        master list containing the declination values for each galaxy

    z_master : numpy array of shape (n,)
        master list containing the redshift for each galaxy

    axis_ratio_master : numpy array of shape (n,)
        master list containing the axis ratio for each galaxy

    phi_master : numpy array of shape (n,)
        master list containing the rotation angle for each galaxy

    mStar_master : numpy array of shape (n,)
        master list containing the stellar mass estimate for each galaxy

    rabsmag_master : numpy array of shape (n,)
        master list containing the SDSS r-band absolute magnitude for each 
        galaxy

    LOCAL_PATH : string
        the directory path of the main script file

    MASTER_FILENAME : string
        file name for the master file
    '''

    ###########################################################################
    # Convert the master data arrays into Column objects to add to the master
    #    data table.
    #--------------------------------------------------------------------------
    manga_plate_col = Column( manga_plate_master)
    manga_IFU_col = Column( manga_IFU_master)

    NSAid_col = Column( NSAid_master)

    ra_col = Column( ra_master)
    dec_col = Column( dec_master)
    z_col = Column( z_master)

    axis_ratio_col = Column( axis_ratio_master)
    phi_col = Column( phi_master)

    mStar_col = Column( mStar_master)
    rabsmag_col = Column( rabsmag_master)
    ###########################################################################


    if not os.path.isfile( LOCAL_PATH + MASTER_FILENAME):
        ########################################################################
        # Add the column objects to an astropy QTable.
        #-----------------------------------------------------------------------
        master_table = QTable([ manga_plate_col,
                                manga_IFU_col,
                                NSAid_col, 
                                ra_col * u.degree,
                                dec_col * u.degree,
                                z_col,
                                axis_ratio_col,
                                phi_col * u.degree,
                                mStar_col * u.M_sun,
                                rabsmag_col],
                       names = ['MaNGA_plate',
                                'MaNGA_IFU',
                                'NSAID', 
                                'ra',
                                'dec',
                                'redshift',
                                'ba',
                                'phi',
                                'Mstar',
                                'rabsmag'])
        ########################################################################
    else:
        ########################################################################
        # Read in current master_file.txt file
        #-----------------------------------------------------------------------
        master_table = QTable.read( LOCAL_PATH + MASTER_FILENAME, 
                                    format='ascii.ecsv')
        ########################################################################


        ########################################################################
        # Build reference dictionary of plate, IFU combinations
        #-----------------------------------------------------------------------
        index_dict = {}

        for i in range( len( manga_plate_master)):
            index_dict[ (manga_plate_master[i], manga_IFU_master[i])] = i
        ########################################################################


        ########################################################################
        # Update column values in master_table
        #-----------------------------------------------------------------------
        for i in range( len( master_table)):
            gal_key = (master_table['MaNGA_plate'][i], master_table['MaNGA_IFU'][i])

            if gal_key in index_dict:
                col_idx = index_dict[ gal_key]

                master_table['NSAID'][i] = NSAid_col[col_idx]

                master_table['ra'][i] = ra_col[col_idx] * u.degree
                master_table['dec'][i] = dec_col[col_idx] * u.degree
                master_table['redshift'][i] = z_col[col_idx]

                master_table['ba'][i] = axis_ratio_col[col_idx]
                master_table['phi'][i] = phi_col[col_idx] * u.degree

                master_table['Mstar'][i] = mStar_col[col_idx] * u.M_sun
                master_table['rabsmag'][i] = rabsmag_col[col_idx]
        ########################################################################


    ###########################################################################
    # Write the master data file in ecsv format.
    #--------------------------------------------------------------------------
    master_table.write( LOCAL_PATH + MASTER_FILENAME,
                        format='ascii.ecsv', overwrite=True)
    ###########################################################################
