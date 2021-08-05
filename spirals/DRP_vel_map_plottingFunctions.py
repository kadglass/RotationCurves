import numpy as np
import numpy.ma as ma

import os
import gc

import matplotlib.pyplot as plt

from DRP_vel_map_functions import deproject_spaxel

from dark_matter_mass_v1 import rot_fit_BB, rot_fit_tanh

from DRP_rotation_curve_plottingFunctions import plot_rband_image, plot_Ha_vel



################################################################################
################################################################################
################################################################################s

def plot_Ha_sigma(Ha_sigma, 
                  gal_ID, 
                  IMAGE_DIR=None, 
                  FOLDER_NAME=None, 
                  IMAGE_FORMAT='eps', 
                  FILENAME_SUFFIX=None, 
                  ax=None):
    '''
    Creates a plot of the H-alpha line widths.


    Parameters:
    ===========

    Ha_sigma : numpy array of shape (n,n)
        H-alpha line width map

    gal_ID : string
        [MaNGA plate] - [MaNGA IFU]

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
    minimum = ma.min( Ha_sigma)
    maximum = ma.max( Ha_sigma)

    cbar_ticks = np.linspace( minimum, maximum, 11, dtype='int')

    ax.set_title( gal_ID + r' H$\alpha$ $\sigma$')
    Ha_sigma_im = ax.imshow( Ha_sigma, 
                             #cmap='RdBu_r', 
                             origin='lower', 
                             vmin = minimum, 
                             vmax = maximum)

    cbar = plt.colorbar( Ha_sigma_im, ax=ax, ticks=cbar_ticks)
    cbar.ax.tick_params( direction='in')
    cbar.set_label(r'$\sigma$ [km/s]')

    ax.tick_params( axis='both', direction='in')
    ax.yaxis.set_ticks_position('both')
    ax.xaxis.set_ticks_position('both')
    ax.set_xlabel('spaxel')
    ax.set_ylabel('spaxel')
    '''
    ax.set_xlabel('$\Delta \alpha$ [arcsec]')
    ax.set_ylabel('$\Delta \delta$ [arcsec]')
    '''
    ############################################################################


    
    if IMAGE_DIR is not None:
        ########################################################################
        # Create output directory if it does not already exist
        #-----------------------------------------------------------------------
        if not os.path.isdir( IMAGE_DIR + FOLDER_NAME):
            os.makedirs( IMAGE_DIR + FOLDER_NAME)
        ########################################################################

        ########################################################################
        # Save figure
        #-----------------------------------------------------------------------
        plt.savefig( IMAGE_DIR + FOLDER_NAME + gal_ID + FILENAME_SUFFIX + IMAGE_FORMAT, 
                     format=IMAGE_FORMAT)
        ########################################################################

        ########################################################################
        # Figure cleanup
        #-----------------------------------------------------------------------
        plt.cla()
        plt.clf()
        plt.close()
        del cbar, Ha_sigma_im
        gc.collect()
        ########################################################################




################################################################################
################################################################################
################################################################################


def plot_rot_curve(mHa_vel, 
                   mHa_vel_ivar,
                   best_fit_values, 
                   i_angle, 
                   phi, 
                   scale,
                   gal_ID, 
                   fit_function,
                   IMAGE_DIR=None, 
                   IMAGE_FORMAT='eps',
                   ax=None):
    '''
    Plot the galaxy rotation curve.


    PARAMETERS
    ==========

    mHa_vel : numpy ndarray of shape (n,n)
        Masked H-alpha velocity array

    mHa_vel_ivar : numpy ndarray of shape (n,n)
        Masked array of the inverse variance of the H-alpha velocity 
        measurements

    best_fit_values : dictionary
        Best-fit values for the velocity map

    i_angle : float
        Inclination angle of the galaxy

    phi : float
        Orientation angle of the galaxy, measured east of north

    scale : float
        Pixel scale (to convert from pixels to kpc)

    gal_ID : string
        MaNGA <plate>-<IFU> for the current galaxy

    fit_function : string
        Determines which function to use for the velocity.  Options are 'BB' and 
        'tanh'.

    IMAGE_DIR : str
        Path of directory in which to store plot.

        Default is None (image will not be saved)

    IMAGE_FORMAT : str
        Format of saved plot

        Default is 'eps'

    ax : matplotlib.pyplot figure axis object
        Axis handle on which to create plot
    '''


    if ax is None:
        fig, ax = plt.subplots(figsize=(5,5))


    ############################################################################
    # Deproject all data values in the given velocity map
    #---------------------------------------------------------------------------
    vel_array_shape = mHa_vel.shape

    r_deproj = np.zeros(vel_array_shape)
    v_deproj = np.zeros(vel_array_shape)

    theta = np.zeros(vel_array_shape)

    for i in range(vel_array_shape[0]):
        for j in range(vel_array_shape[1]):

            r_deproj[i,j], theta[i,j] = deproject_spaxel((i,j), 
                                                         (best_fit_values['x0'], best_fit_values['y0']), 
                                                         phi, 
                                                         i_angle)

            ####################################################################
            # Find the sign of r_deproj
            #-------------------------------------------------------------------
            if np.cos(theta[i,j]) < 0:
                r_deproj[i,j] *= -1
            ####################################################################

    # Scale radii to convert from spaxels to kpc
    r_deproj *= scale

    # Deproject velocity values
    v_deproj = (mHa_vel - best_fit_values['v_sys'])/np.abs(np.cos(theta))
    v_deproj /= np.sin(i_angle)

    # Apply mask to arrays
    rm_deproj = ma.array(r_deproj, mask=mHa_vel.mask)
    vm_deproj = ma.array(v_deproj, mask=mHa_vel.mask)
    ############################################################################


    ############################################################################
    # Calculate functional form of rotation curve
    #---------------------------------------------------------------------------
    r = np.linspace(ma.min(rm_deproj), ma.max(rm_deproj), 100)

    if fit_function == 'BB':
        v = rot_fit_BB(r, [best_fit_values['v_max'], 
                           best_fit_values['r_turn'], 
                           best_fit_values['alpha']])
    elif fit_function == 'tanh':
        v = rot_fit_tanh(r, [best_fit_values['v_max'], 
                             best_fit_values['r_turn']])
    else:
        print('Fit function not known.  Please update plot_rot_curve function.')
    ############################################################################


    ############################################################################
    # Plot rotation curve
    #---------------------------------------------------------------------------
    ax.set_title(gal_ID + ' rotation curve')

    ax.plot(rm_deproj, vm_deproj, 'k.', markersize=1)
    ax.plot(r, v, 'c')

    ax.set_ylim([-1.25*best_fit_values['v_max'], 1.25*best_fit_values['v_max']])
    ax.tick_params(axis='both', direction='in')
    ax.yaxis.set_ticks_position('both')
    ax.xaxis.set_ticks_position('both')
    ax.set_xlabel('Deprojected radius [kpc/h]')
    ax.set_ylabel('Rotational velocity [km/s]')
    ############################################################################


    if IMAGE_DIR is not None:
        ########################################################################
        # Create output directory if it does not already exist
        #-----------------------------------------------------------------------
        if not os.path.isdir(IMAGE_DIR + '/vel_map_rot_curve_' + fit_function):
            os.makedirs(IMAGE_DIR + '/vel_map_rot_curve_' + fit_function)
        ########################################################################

        ########################################################################
        # Save figure
        #-----------------------------------------------------------------------
        plt.savefig(IMAGE_DIR + '/vel_map_rot_curve_' + fit_function + '/' + gal_ID + '_rot_curve_' + fit_function + '.' + IMAGE_FORMAT,
                    format=IMAGE_FORMAT)
        ########################################################################

        ########################################################################
        # Figure cleanup
        #-----------------------------------------------------------------------
        plt.cla()
        plt.clf()
        plt.close()
        gc.collect()
        ########################################################################




################################################################################
################################################################################
################################################################################



def plot_residual(model_map, 
                  data_map, 
                  gal_ID, 
                  IMAGE_DIR=None,
                  FOLDER_NAME=None,
                  IMAGE_FORMAT='eps', 
                  FILENAME_SUFFIX=None,
                  ax=None):
    '''
    Creates a plot of the residual between the model and the data.


    PARAMETERS
    ==========

    model_map : numpy array of shape (n,n)
        Model H-alpha velocity map [km/s]

    data_map : numpy array of shape (n,n)
        Measured H-alpha velocity map [km/s]

    gal_ID : string
        [MaNGA plate] - [MaNGA IFU]

    IMAGE_DIR : string
        Path of directory to store images.  Default is None (image will not be 
        saved).

    FOLDER_NAME : string
        Name of folder in which to save images.  Default is None (iamge will not 
        be saved).

    IMAGE_FORMAT : string
        Format of saved images.  Default is eps.

    FILENAME_SUFFIX : string
        Suffix to append to gal_ID to create image file name.  Default is None 
        (image will not be saved).

    ax : matplotlib.pyplot figure axis object
        Axis handle on which to create plot.
    '''

    if ax is None:
        fig, ax = plt.subplots()

    ############################################################################
    # Create residual array
    #---------------------------------------------------------------------------
    residual_map = model_map - data_map
    ############################################################################


    ############################################################################
    rmax_bound = ma.max(np.abs(residual_map))
    rmin_bound = -rmax_bound

    cbar_ticks = np.linspace(rmin_bound, rmax_bound, 11, dtype='int')

    ax.set_title(gal_ID + ' residual')

    residual_im = ax.imshow(residual_map, 
                            cmap='PiYG_r', 
                            origin='lower', 
                            vmin=rmin_bound,
                            vmax=rmax_bound)

    cbar = plt.colorbar(residual_im, ax=ax, ticks=cbar_ticks)
    cbar.ax.tick_params(direction='in')
    cbar.set_label('residual (model - data)')

    ax.tick_params(axis='both', direction='in')
    ax.yaxis.set_ticks_position('both')
    ax.xaxis.set_ticks_position('both')
    ax.set_xlabel('spaxel')
    ax.set_ylabel('spaxel')
    ############################################################################


    if IMAGE_DIR is not None:
        ########################################################################
        # Create output directory if it does not already exist
        #-----------------------------------------------------------------------
        if not os.path.isdir(IMAGE_DIR + FOLDER_NAME):
            os.makedirs(IMAGE_DIR + FOLDER_NAME)
        ########################################################################

        ########################################################################
        # Save figure
        #-----------------------------------------------------------------------
        plt.savefig(IMAGE_DIR + FOLDER_NAME + gal_ID + FILENAME_SUFFIX + IMAGE_FORMAT, 
                    format=IMAGE_FORMAT)
        ########################################################################

        ########################################################################
        # Figure cleanup
        #-----------------------------------------------------------------------
        plt.cla()
        plt.clf()
        plt.close()
        del cbar, residual_im
        gc.collect()
        ########################################################################



################################################################################
################################################################################
################################################################################



def plot_residual_norm(model_map, 
                       data_map, 
                       gal_ID, 
                       IMAGE_DIR=None,
                       FOLDER_NAME=None,
                       IMAGE_FORMAT='eps', 
                       FILENAME_SUFFIX=None,
                       ax=None):
    '''
    Creates a plot of the residual between the model and the data, normalized by 
    the model.


    PARAMETERS
    ==========

    model_map : numpy array of shape (n,n)
        Model H-alpha velocity map [km/s]

    data_map : numpy array of shape (n,n)
        Measured H-alpha velocity map [km/s]

    gal_ID : string
        [MaNGA plate] - [MaNGA IFU]

    IMAGE_DIR : string
        Path of directory to store images.  Default is None (image will not be 
        saved).

    FOLDER_NAME : string
        Name of folder in which to save images.  Default is None (iamge will not 
        be saved).

    IMAGE_FORMAT : string
        Format of saved images.  Default is eps.

    FILENAME_SUFFIX : string
        Suffix to append to gal_ID to create image file name.  Default is None 
        (image will not be saved).

    ax : matplotlib.pyplot figure axis object
        Axis handle on which to create plot.
    '''

    if ax is None:
        fig, ax = plt.subplots()

    ############################################################################
    # Create residual array
    #---------------------------------------------------------------------------
    residual_map = (data_map - model_map)/model_map
    ############################################################################


    ############################################################################
    rmax_bound = ma.max(np.abs(residual_map))
    rmin_bound = -rmax_bound

    cbar_ticks = np.linspace(rmin_bound, rmax_bound, 11, dtype='int')

    ax.set_title(gal_ID + ' normalized residual')

    residual_im = ax.imshow(residual_map, 
                            cmap='PiYG_r', 
                            origin='lower', 
                            vmin=rmin_bound,
                            vmax=rmax_bound)

    cbar = plt.colorbar(residual_im, ax=ax, ticks=cbar_ticks)
    cbar.ax.tick_params(direction='in')
    cbar.set_label('normalized residual (data - model)/model')

    ax.tick_params(axis='both', direction='in')
    ax.yaxis.set_ticks_position('both')
    ax.xaxis.set_ticks_position('both')
    ax.set_xlabel('spaxel')
    ax.set_ylabel('spaxel')
    ############################################################################


    if IMAGE_DIR is not None:
        ########################################################################
        # Create output directory if it does not already exist
        #-----------------------------------------------------------------------
        if not os.path.isdir(IMAGE_DIR + FOLDER_NAME):
            os.makedirs(IMAGE_DIR + FOLDER_NAME)
        ########################################################################

        ########################################################################
        # Save figure
        #-----------------------------------------------------------------------
        plt.savefig(IMAGE_DIR + FOLDER_NAME + gal_ID + FILENAME_SUFFIX + IMAGE_FORMAT, 
                    format=IMAGE_FORMAT)
        ########################################################################

        ########################################################################
        # Figure cleanup
        #-----------------------------------------------------------------------
        plt.cla()
        plt.clf()
        plt.close()
        del cbar, residual_im
        gc.collect()
        ########################################################################




################################################################################
################################################################################
################################################################################



def plot_chi2(model_map, 
              data_map, 
              ivar_map, 
              gal_ID, 
              IMAGE_DIR=None,
              FOLDER_NAME=None,
              IMAGE_FORMAT='eps', 
              FILENAME_SUFFIX=None,
              ax=None):
    '''
    Creates a plot of the chi2 values of the model.


    PARAMETERS
    ==========

    model_map : numpy array of shape (n,n)
        Model H-alpha velocity map [km/s]

    data_map : numpy array of shape (n,n)
        Measured H-alpha velocity map [km/s]

    ivar_map : numpy array of shape (n,n)
        Measured inverse variances of the H-alpha velocity map [s/km]^2

    gal_ID : string
        [MaNGA plate] - [MaNGA IFU]

    IMAGE_DIR : string
        Path of directory to store images.  Default is None (image will not be 
        saved).

    FOLDER_NAME : string
        Name of folder in which to save images.  Default is None (iamge will not 
        be saved).

    IMAGE_FORMAT : string
        Format of saved images.  Default is eps.

    FILENAME_SUFFIX : string
        Suffix to append to gal_ID to create image file name.  Default is None 
        (image will not be saved).

    ax : matplotlib.pyplot figure axis object
        Axis handle on which to create plot.
    '''

    if ax is None:
        fig, ax = plt.subplots()

    ############################################################################
    # Create chi2 array
    #---------------------------------------------------------------------------
    chi2_map = ivar_map*(model_map - data_map)**2
    ############################################################################


    ############################################################################
    rmax_bound = ma.max(chi2_map)
    rmin_bound = 0

    cbar_ticks = np.linspace(rmin_bound, rmax_bound, 11, dtype='int')

    ax.set_title(gal_ID + r' $\chi^2$')

    chi2_im = ax.imshow(chi2_map, 
                        cmap='PiYG_r', 
                        origin='lower', 
                        vmin=rmin_bound,
                        vmax=rmax_bound)

    cbar = plt.colorbar(chi2_im, ax=ax, ticks=cbar_ticks)
    cbar.ax.tick_params(direction='in')
    cbar.set_label(r'$\chi^2$')

    ax.tick_params(axis='both', direction='in')
    ax.yaxis.set_ticks_position('both')
    ax.xaxis.set_ticks_position('both')
    ax.set_xlabel('spaxel')
    ax.set_ylabel('spaxel')
    ############################################################################


    if IMAGE_DIR is not None:
        ########################################################################
        # Create output directory if it does not already exist
        #-----------------------------------------------------------------------
        if not os.path.isdir(IMAGE_DIR + FOLDER_NAME):
            os.makedirs(IMAGE_DIR + FOLDER_NAME)
        ########################################################################

        ########################################################################
        # Save figure
        #-----------------------------------------------------------------------
        plt.savefig(IMAGE_DIR + FOLDER_NAME + gal_ID + FILENAME_SUFFIX + IMAGE_FORMAT, 
                    format=IMAGE_FORMAT)
        ########################################################################

        ########################################################################
        # Figure cleanup
        #-----------------------------------------------------------------------
        plt.cla()
        plt.clf()
        plt.close()
        del cbar, chi2_im
        gc.collect()
        ########################################################################




################################################################################
################################################################################
################################################################################



def plot_diagnostic_panel(r_band, 
                          mHa_vel, 
                          mHa_vel_ivar, 
                          best_fit_map,
                          best_fit_values,
                          i_angle, 
                          phi, 
                          scale,
                          gal_ID, 
                          fit_function,
                          IMAGE_DIR=None, 
                          IMAGE_FORMAT='eps'):
    '''
    Plot a two-by-two paneled image containing the entire rgb image, the masked 
    H-alpha velocity array, the masked H-alpha model velocity array, and the 
    rotation curve.


    PARAMETERS
    ==========

    r_band : numpy array of shape (n,n)
        r-band flux data

    mHa_vel : numpy ndarray of shape (n,n)
        Masked H-alpha velocity array

    mHa_vel_ivar : numpy ndarray of shape (n,n)
        Masked array of the inverse variance of the H-alpha velocity measurements

    best_fit_map : numpy ndarray of shape (n,n)
        Masked best-fit model H-alpha velocity array

    best_fit_values : dictionary
        Best-fit values for the velocity map

    i_angle : float
        Inclination angle of the galaxy

    phi : float
        Orientation angle of the galaxy, measured east of north

    scale : float
        Pixel scale (to convert from pixels to kpc)

    gal_ID : string
        MaNGA <plate>-<IFU> for the current galaxy

    fit_function : string
        Determines which function to use for the velocity.  Options are 'BB' and 
        'tanh'.

    IMAGE_DIR : string
        Path of directory in which to store plot.

        Default is None (image will not be saved)

    IMAGE_FORMAT : string
        Format of saved plot

        Default is 'eps'
    '''

    panel_fig, ((r_band_panel, rot_curve_panel),
                (mHa_vel_panel, Ha_vel_model_panel)) = plt.subplots(2,2)
    panel_fig.set_figheight(10)
    panel_fig.set_figwidth(10)
    plt.suptitle(gal_ID + ' diagnostic panel', y=1.05, fontsize=16)

    plot_rband_image(r_band, gal_ID, ax=r_band_panel)

    plot_rot_curve(mHa_vel, 
                   mHa_vel_ivar, 
                   best_fit_values, 
                   i_angle, 
                   phi, 
                   scale, 
                   gal_ID, 
                   fit_function, 
                   ax=rot_curve_panel)

    plot_Ha_vel(mHa_vel, gal_ID, ax=mHa_vel_panel)

    plot_Ha_vel(best_fit_map, gal_ID, ax=Ha_vel_model_panel)

    panel_fig.tight_layout()


    if IMAGE_DIR is not None:
        ########################################################################
        # Create output directory if it does not already exist
        #-----------------------------------------------------------------------
        if not os.path.isdir(IMAGE_DIR + '/diagnostic_panels'):
            os.makedirs(IMAGE_DIR + '/diagnostic_panels')
        ########################################################################

        ########################################################################
        # Save figure
        #-----------------------------------------------------------------------
        plt.savefig(IMAGE_DIR + '/diagnostic_panels/' + gal_ID + '_vel_map_diagnostic_panel.' + IMAGE_FORMAT,
                    format=IMAGE_FORMAT)
        ########################################################################

        ########################################################################
        # Figure cleanup
        #-----------------------------------------------------------------------
        plt.cla()
        plt.clf()
        plt.close(panel_fig)
        del panel_fig, r_band_panel, rot_curve_panel, mHa_vel_panel, Ha_vel_model_panel
        gc.collect()
        ########################################################################







