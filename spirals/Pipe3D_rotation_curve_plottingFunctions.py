import os
import gc

import numpy as np
import numpy.ma as ma

import matplotlib

#matplotlib.use('TKAgg')

# for sciserver
matplotlib.use('Agg')
import matplotlib.pyplot as plt



###############################################################################
###############################################################################
###############################################################################



def plot_vband_image(v_band, gal_ID, IMAGE_DIR=None, IMAGE_FORMAT='eps', ax=None):
    '''
    Creates a plot of the v-band flux map


    Parameters:
    ===========

    v_band : numpy array of shape (n,n)
        v_band flux map

    gal_ID : string
        MaNGA plate number - MaNGA fiberID number

    IMAGE_DIR : string
        Path of directory to store images

    IMAGE_FORMAT : string
        Format of saved image.  Default is eps

    ax : matplotlib.pyplot axis object
        Axes handle on which to create plot
    '''


    ###########################################################################
    # Dimensions of array
    #--------------------------------------------------------------------------
    array_length = v_band.shape[0]  # y-coordinate distance
    array_width = v_band.shape[1]  # x-coordinate distance
    ###########################################################################


    ###########################################################################
    v_band_cbar_ticks = np.linspace(  0, v_band.max(), 7)

    for val, i in zip( v_band_cbar_ticks, range( len( v_band_cbar_ticks))):
        val = '%.3f' % val
        v_band_cbar_ticks[i] = val

    if ax is None:
        fig, ax = plt.subplots()

    ax.set_title( gal_ID + ' Visual Band')
    v_band_im = ax.imshow( v_band, origin='lower')

    cbar = plt.colorbar( v_band_im, ax=ax, ticks = np.linspace(  0, v_band.max(), 6))
    cbar.ax.tick_params( direction='in', color='white')
    cbar.set_label('Visual Band Flux [$10^{-17}$ erg s$^{-1}$ cm$^{-2}$]')

    ax.set_xticks( np.arange( 0, array_width, 10))
    ax.set_yticks( np.arange( 0, array_length, 10))
    ax.tick_params( axis='both', direction='in', color='white')
    ax.yaxis.set_ticks_position('both')
    ax.xaxis.set_ticks_position('both')
    ax.set_xlabel('spaxel')
    ax.set_ylabel('spaxel')
    ###########################################################################


    ###########################################################################
    # Save figure
    #--------------------------------------------------------------------------
    if IMAGE_DIR is not None:
        #######################################################################
        # Create output directory if it does not already exist
        #----------------------------------------------------------------------
        if not os.path.isdir( IMAGE_DIR + '/unmasked_v_band'):
            os.makedirs( IMAGE_DIR + '/unmasked_v_band')
        #######################################################################


        plt.savefig( IMAGE_DIR + '/unmasked_v_band/' + gal_ID + '_v_band_raw.' + IMAGE_FORMAT, format=IMAGE_FORMAT)


        #######################################################################
        # Figure cleanup
        #----------------------------------------------------------------------
        #plt.show()
        plt.cla()
        plt.clf()
        plt.close()
        del cbar, v_band_im
        gc.collect()
        #######################################################################



###############################################################################
###############################################################################
###############################################################################



def plot_sMass_image(sMass, 
                     gal_ID, 
                     IMAGE_DIR=None, 
                     IMAGE_FORMAT='eps', 
                     ax=None):
    '''
    Creates a plot of the stellar mass density map


    Parameters:
    ===========

    sMass : numpy array of shape (n,n)
        stellar mass density map

    gal_ID : string
        MaNGA plate number - MaNGA fiberID number

    IMAGE_DIR : string
        Path of directory to store images

    IMAGE_FORMAT : string
        Format of saved image.  Default is eps

    ax : matplotlib.pyplot axis object
        Axes handle on which to create plot
    '''


    ###########################################################################
    # Dimensions of array
    #--------------------------------------------------------------------------
    array_length = sMass.shape[0]  # y-coordinate distance
    array_width = sMass.shape[1]  # x-coordinate distance
    ###########################################################################


    ###########################################################################
    sMass_max = ma.max(sMass)
    sMass_cbar_ticks = np.linspace(  0, sMass_max, 7)
    print('ticks:', sMass_cbar_ticks)
    '''
    for val, i in zip( sMass_cbar_ticks, range( len( sMass_cbar_ticks))):
        val = '%.3f' % val
        sMass_cbar_ticks[i] = val
    '''

    print('ticks:', sMass_cbar_ticks)

    if ax is None:
        fig, ax = plt.subplots()

    ax.set_title( gal_ID + ' stellar mass density', fontsize=14)
    sMass_im = ax.imshow( sMass, origin='lower')

    cbar = plt.colorbar( sMass_im, ax=ax, ticks = sMass_cbar_ticks)
    cbar.ax.tick_params( direction='in', color='white', labelsize=12)
    cbar.set_label(r'stellar mass density [log($M/M_\odot$)]', fontsize=14)

    ax.set_xticks( np.arange( 0, array_width, 10))
    ax.set_yticks( np.arange( 0, array_length, 10))
    ax.tick_params( axis='both', direction='in', color='white', labelsize=12)
    ax.yaxis.set_ticks_position('both')
    ax.xaxis.set_ticks_position('both')
    ax.set_xlabel('spaxel', fontsize=14)
    ax.set_ylabel('spaxel', fontsize=14)
    ###########################################################################


    ###########################################################################
    # Save figure
    #--------------------------------------------------------------------------
    if IMAGE_DIR is not None:
        #######################################################################
        # Create output directory if it does not already exist
        #----------------------------------------------------------------------
        if not os.path.isdir( IMAGE_DIR + '/sMass'):
            os.makedirs( IMAGE_DIR + '/sMass')
        #######################################################################


        plt.savefig(IMAGE_DIR + '/sMass/' + gal_ID + '_sMass.' + IMAGE_FORMAT, 
                    format=IMAGE_FORMAT, bbox_inches = 'tight', pad_inches = 0)


        #######################################################################
        # Figure cleanup
        #----------------------------------------------------------------------
        #plt.show()
        plt.cla()
        plt.clf()
        plt.close()
        del cbar, sMass_im
        gc.collect()
        #######################################################################



###############################################################################
###############################################################################
###############################################################################



def plot_Ha_vel(Ha_vel, 
                gal_ID, 
                IMAGE_DIR=None, 
                FOLDER_NAME=None, 
                IMAGE_FORMAT='eps', 
                FILENAME_SUFFIX=None, 
                ax=None):
    '''
    Creates a plot of the H-alpha velocity map.


    Parameters:
    ===========

    Ha_vel : numpy array of shape (n,n)
        H-alpha velocity map

    gal_ID : string
        MaNGA plate number - MaNGA fiberID number

    IMAGE_DIR : string
        Path of directory to store images

    FOLDER_NAME : string
        Name of folder in which to save image

    IMAGE_FORMAT : string
        Format of saved image.  Default is eps

    FILENAME_SUFFIX : string
        Suffix to append to gal_ID to create image filename

    ax : matplotlib.pyplot figure axis object
        Axes handle on which to create plot
    '''


    if ax is None:
        fig, ax = plt.subplots()


    ###########################################################################
    # Dimensions of array
    #--------------------------------------------------------------------------
    array_length = Ha_vel.shape[0]  # y-coordinate distance
    array_width = Ha_vel.shape[1]  # x-coordinate distance
    ###########################################################################


    ###########################################################################
    minimum = np.min( Ha_vel)
    maximum = np.max( Ha_vel)
    if minimum > 0:
        vmax_bound = maximum
        vmin_bound = 0
    else:
        vmax_bound = np.max( [np.abs(minimum), np.abs(maximum)])
        vmin_bound = -vmax_bound
    cbar_ticks = np.linspace( vmin_bound, vmax_bound, 11, dtype='int')

    ax.set_title( gal_ID + r' H$\alpha$ Velocity')
    Ha_vel_im = ax.imshow( Ha_vel, cmap='bwr', origin='lower', vmin = vmin_bound, vmax = vmax_bound)

    cbar = plt.colorbar( Ha_vel_im, ax=ax, ticks = cbar_ticks)
    cbar.ax.tick_params( direction='in')
    cbar.set_label('$v_{rot}$ [km/s]')

    ax.set_xticks( np.arange( 0, array_width, 10))
    ax.set_yticks( np.arange( 0, array_length, 10))
    ax.tick_params( axis='both', direction='in')
    ax.yaxis.set_ticks_position('both')
    ax.xaxis.set_ticks_position('both')
    ax.set_xlabel('spaxel')
    ax.set_ylabel('spaxel')
    ###########################################################################


    ###########################################################################
    # Save figure
    #--------------------------------------------------------------------------
    if IMAGE_DIR is not None:
        ###########################################################################
        # Create output directory if it does not already exist
        #--------------------------------------------------------------------------
        if not os.path.isdir( IMAGE_DIR + FOLDER_NAME):
            os.makedirs( IMAGE_DIR + FOLDER_NAME)
        ###########################################################################


        plt.savefig( IMAGE_DIR + FOLDER_NAME + gal_ID + FILENAME_SUFFIX + IMAGE_FORMAT, format=IMAGE_FORMAT)


        #######################################################################
        # Figure cleanup
        #----------------------------------------------------------------------
        #plt.show()
        plt.cla()
        plt.clf()
        plt.close()
        del cbar, Ha_vel_im
        gc.collect()
        #######################################################################



###############################################################################
###############################################################################
###############################################################################



def plot_rot_curve(gal_ID, data_table, IMAGE_DIR=None, IMAGE_FORMAT=None, ax=None):
    '''
    Plot galaxy rotation curves


    Parameters:
    ===========

    gal_ID : string
        MaNGA plate number - MaNGA fiberID number

    data_table : Astropy QTable
        Table containing measured rotational velocities at given deprojected 
        radii

    IMAGE_DIR : string
        Path of directory to store images

    IMAGE_FORMAT : string
        Format of saved image

    ax : matplotlib.pyplot figure axis object
        Axes handle on which to create plot

    '''


    if ax is None:
        fig, ax = plt.subplots( figsize=(5, 5))


    ###########################################################################
    ax.set_title( gal_ID + ' Rotation Curves')
    ax.plot( data_table['deprojected_distance'], data_table['max_velocity'], 
              'rs', markersize=5, label='Total (pos)')
    ax.plot( data_table['deprojected_distance'], np.abs( data_table['min_velocity']), 
              'b^', markersize=5, label='Total (neg)')
    ax.plot( data_table['deprojected_distance'], data_table['rot_vel_avg'], 
              'gp', markersize=7, label='Total (avg)')
    ax.plot( data_table['deprojected_distance'], data_table['sVel_rot'], 
              'cD', markersize=4, label='Stellar mass')
    ax.plot( data_table['deprojected_distance'], data_table['dmVel_rot'], 
              'kX', markersize=7, label='Dark matter')

    ax.tick_params( axis='both', direction='in')
    ax.yaxis.set_ticks_position('both')
    ax.xaxis.set_ticks_position('both')
    ax.set_xlabel('Deprojected Radius [kpc/h]')
    ax.set_ylabel('Rotational Velocity [km/s]')
    ax.legend(loc='upper left')
    ###########################################################################


    ###########################################################################
    # Save figure
    #--------------------------------------------------------------------------
    if IMAGE_DIR is not None:
        #######################################################################
        # Create output directory if it does not already exist
        #----------------------------------------------------------------------
        if not os.path.isdir( IMAGE_DIR + '/rot_curves'):
            os.makedirs( IMAGE_DIR + '/rot_curves')
        #######################################################################


        plt.savefig( IMAGE_DIR + "/rot_curves/" + gal_ID + "_rot_curve." + IMAGE_FORMAT,
                    format=IMAGE_FORMAT)


        #######################################################################
        # Figure cleanup
        #----------------------------------------------------------------------
        #plt.show()
        plt.cla()
        plt.clf()
        plt.close()
        gc.collect()
        #######################################################################



###############################################################################
###############################################################################
###############################################################################



def plot_mass_curve(IMAGE_DIR, IMAGE_FORMAT, gal_ID, data_table):
    '''
    Plot the cumulative mass as a function of deprojected radius


    Parameters:
    ===========

    IMAGE_DIR : string
        Path of directory to store images

    IMAGE_FORMAT : string
        Format of saved image

    gal_ID : string
        MaNGA plate number - MaNGA fiberID number

    data_table : Astropy QTable
        Table containing measured rotational velocities at given deprojected 
        radii

    '''

    plt.figure( figsize=(5, 5))

    plt.title( gal_ID + ' Mass Curves')
    plt.plot( data_table['deprojected_distance'], data_table['mass_interior'], 
              'gp', markersize=7, label='Total mass (avg)')
    plt.plot( data_table['deprojected_distance'], data_table['sMass_interior'], 
              'cD', markersize=4, label='Stellar mass')
    plt.plot( data_table['deprojected_distance'], data_table['dmMass_interior'], 
              'kX', markersize=7, label='Dark matter mass')

    plt.tick_params( axis='both', direction='in')
    #ax.yaxis.set_ticks_position('both')
    #ax.xaxis.set_ticks_position('both')
    plt.xlabel('Deprojected Radius [kpc/h]')
    plt.ylabel('Mass Interior [$M_{\odot}$]')
    plt.legend(loc='upper left')

    if IMAGE_DIR is not None:
        ########################################################################
        # Create output directory if it does not already exist
        #-----------------------------------------------------------------------
        if not os.path.isdir( IMAGE_DIR + '/mass_curves'):
            os.makedirs( IMAGE_DIR + '/mass_curves')
        ########################################################################

        ########################################################################
        # Save figure
        #-----------------------------------------------------------------------
        plt.savefig( IMAGE_DIR + "/mass_curves/" + gal_ID + "_mass_curve." + IMAGE_FORMAT,
                    format=IMAGE_FORMAT)
        ########################################################################

        ########################################################################
        # Clean up figure objects
        #-----------------------------------------------------------------------
        plt.cla()
        plt.clf()
        plt.close()
        gc.collect()
        ########################################################################



###############################################################################
###############################################################################
###############################################################################




def plot_diagnostic_panel( IMAGE_DIR, IMAGE_FORMAT, gal_ID, v_band, masked_Ha_vel, masked_vel_contour_plot, data_table):
    '''
    Plot a two by two paneled image containging the entire v-band array, the 
    masked H-alpha array, the masked H-alpha array containing ovals of the 
    spaxels processed in the algorithm, and the averaged max and min rotation 
    curves alongside the stellar mass rotation curve,


    Parameters:
    ===========

    IMAGE_DIR : string
        Path of directory to store images

    IMAGE_FORMAT : string
        Format of saved image

    gal_ID : string
        MaNGA plate number - MaNGA fiberID number

    v_band : numpy array of shape (n,n)
        v_band flux map

    masked_Ha_vel : numpy array of shape (n,n)
        Masked H-alpha velocity map

    masked_vel_contour_plot : numpy array of shape (n,n)
        Masked H-alpha velocity map showing only those spaxels within annuli

    data_table : Astropy QTable
        Table containing measured rotational velocities at given deprojected 
        radii

    '''


#    panel_fig, (( Ha_vel_panel, mHa_vel_panel),
#                ( contour_panel, rot_curve_panel)) = plt.subplots( 2, 2)
    panel_fig, (( v_band_panel, mHa_vel_panel),
                ( contour_panel, rot_curve_panel)) = plt.subplots( 2, 2)
    panel_fig.set_figheight( 10)
    panel_fig.set_figwidth( 10)
    plt.suptitle( gal_ID + " Diagnostic Panel", y=1.05, fontsize=16)


    plot_vband_image( v_band, gal_ID, ax=v_band_panel)

    plot_Ha_vel( masked_Ha_vel, gal_ID, ax=mHa_vel_panel)

    plot_Ha_vel( masked_vel_contour_plot, gal_ID, ax=contour_panel)

    plot_rot_curve( gal_ID, data_table, ax=rot_curve_panel)

    panel_fig.tight_layout()


    if IMAGE_DIR is not None:
        ########################################################################
        # Create output directory if it does not already exist
        #-----------------------------------------------------------------------
        if not os.path.isdir( IMAGE_DIR + '/diagnostic_panels'):
            os.makedirs( IMAGE_DIR + '/diagnostic_panels')
        ########################################################################

        ########################################################################
        # Save figure
        #-----------------------------------------------------------------------
        plt.savefig( IMAGE_DIR + "/diagnostic_panels/" + gal_ID + "_diagnostic_panel." + IMAGE_FORMAT,
                    format=IMAGE_FORMAT)
        ########################################################################

        ########################################################################
        # Figure cleanup
        #-----------------------------------------------------------------------
        plt.cla()
        plt.clf()
        plt.close( panel_fig)
        del panel_fig, v_band_panel, mHa_vel_panel, contour_panel, rot_curve_panel
        gc.collect()
        ########################################################################

