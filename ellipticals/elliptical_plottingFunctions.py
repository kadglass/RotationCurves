import gc
import numpy as np
import numpy.ma as ma
import matplotlib.pyplot as plt
import matplotlib

matplotlib.rcParams['figure.dpi'] = 100
matplotlib.rcParams['savefig.dpi'] = 300

def plot_sigma_map(gal_ID, sigma_map, corr=False, ax=None, IMAGE_DIR=None, 
                   IMAGE_FORMAT='png'):

    '''
    
    Plot stellar velocity dispersion map

    PARAMETERS
    ==========
    gal_ID : string
        galaxy plate-ifu
    
    sigma_map : array
        map to be plot
    
    corr : boolean
        indicates if the map is corrected. default is False

    ax : matplotlib.pyplot figure axis object
        Axes handle on which to create plot
    
    IMAGE_DIR : string
        directory to save plot

    IMAGE_FORMAT : string
        format to save plot. default is 'png'
    
    
    '''

    if ax is None:
        fig, ax = plt.subplots()


    ############################################################################
    # Determine limits of color scale
    #---------------------------------------------------------------------------
    vmax = ma.max(sigma_map)
    cbar_ticks = np.linspace( 0, vmax, 11, dtype='int')
    ############################################################################


    ############################################################################
    # Create plot title
    #---------------------------------------------------------------------------
    if corr:
        map_type = ' corrected'
    else:
        map_type = ''

    ax.set_title(gal_ID + map_type + ' stellar velocity dispersion', fontsize=12)
    ############################################################################
    

    ############################################################################
    # Create plot
    #---------------------------------------------------------------------------
    vel_im = ax.imshow( sigma_map, 
                        cmap='winter', # 4h 
                        origin='lower', 
                        vmin=0, 
                        vmax=vmax)

    cbar = plt.colorbar( vel_im, ax=ax, ticks=cbar_ticks)
    cbar.ax.tick_params( direction='in', labelsize=10)
    cbar.set_label(r'$\sigma_{*}$ [km/s]', fontsize=12) 

    ax.tick_params( axis='both', direction='in', labelsize=10)
    ax.yaxis.set_ticks_position('both')
    ax.xaxis.set_ticks_position('both')
    ax.set_xlabel('spaxel', fontsize=12)
    ax.set_ylabel('spaxel', fontsize=12)

    ############################################################################
    
    if IMAGE_DIR is not None:
        if corr:
            FOLDER_NAME = 'star_sigma_corrected'
        else:
            FOLDER_NAME = 'star_sigma'

        ########################################################################
        # Save figure
        #-----------------------------------------------------------------------
        plt.savefig( IMAGE_DIR + FOLDER_NAME + '/' + gal_ID + '_' + FOLDER_NAME + '.' + IMAGE_FORMAT, 
                     format=IMAGE_FORMAT, 
                     #bbox_inches = 'tight', 
                     #pad_inches = 0
                     )
        ########################################################################

        ########################################################################
        # Figure cleanup
        #-----------------------------------------------------------------------
        plt.cla()
        plt.clf()
        plt.close()
        del cbar, vel_im
        gc.collect()

def plot_vel(vel, 
             gal_ID, 
             ax=None,
             IMAGE_DIR=None,  
             IMAGE_FORMAT='png'
             ):
    '''
    Creates a plot of the velocity map.


    PARAMETERS
    ==========

    vel : numpy array of shape (n,n)
        velocity map

    gal_ID : string
        [MaNGA plate] - [MaNGA IFU]
        
    ax : matplotlib.pyplot figure axis object
        Axes handle on which to create plot  

    IMAGE_DIR : string
        Path of directory to store images

    IMAGE_FORMAT : string
        Format of saved image.  Default is eps


    
    '''


    if ax is None:
        fig, ax = plt.subplots()


    ############################################################################
    # Determine limits of color scale
    #---------------------------------------------------------------------------
    minimum = ma.min(vel)
    maximum = ma.max(vel)

    if minimum > 0:
        vmax_bound = maximum
        vmin_bound = 0
    else:
        vmax_bound = np.max( [np.abs(minimum), np.abs(maximum)])
        vmin_bound = -vmax_bound

    cbar_ticks = np.linspace( vmin_bound, vmax_bound, 11, dtype='int')
    ############################################################################


    ############################################################################
    # Create plot title
    #---------------------------------------------------------------------------

    ax.set_title(gal_ID + r' H$\alpha$ velocity map', fontsize=12)
    ############################################################################
    

    ############################################################################
    # Create plot
    #---------------------------------------------------------------------------
    vel_im = ax.imshow( vel, 
                        cmap='RdBu_r', 
                        origin='lower', 
                        vmin=vmin_bound, 
                        vmax=vmax_bound)

    cbar = plt.colorbar( vel_im, ax=ax, ticks=cbar_ticks)
    cbar.ax.tick_params( direction='in', labelsize=10)
    #cbar.set_label('$v$ [km/s]')
    cbar.set_label(r'$v_{rot}$ [km/s]', fontsize=12) # formatting for paper

    ax.tick_params( axis='both', direction='in', labelsize=10)
    ax.yaxis.set_ticks_position('both')
    ax.xaxis.set_ticks_position('both')
    ax.set_xlabel('spaxel', fontsize=12)
    ax.set_ylabel('spaxel', fontsize=12)

    
    ############################################################################


    
    if IMAGE_DIR is not None:

        ########################################################################
        # Save figure
        #-----------------------------------------------------------------------
        plt.savefig( IMAGE_DIR + '/Ha_velocity/' + gal_ID + '_Ha_velocity.' + IMAGE_FORMAT, 
                     format=IMAGE_FORMAT, 
                     #bbox_inches = 'tight', 
                     #pad_inches = 0
                     )
        ########################################################################

        ########################################################################
        # Figure cleanup
        #-----------------------------------------------------------------------
        plt.cla()
        plt.clf()
        plt.close()
        del cbar, vel_im
        gc.collect()
        ########################################################################



def plot_flux_map(gal_ID, flux_map, ax=None, IMAGE_DIR=None, 
                  IMAGE_FORMAT='png'):

    '''
    
    Plot flux map

    PARAMETERS
    ==========
    gal_ID : string
        galaxy plate-ifu
    
    flux_map : array
        flux map for plot

    ax : matplotlib.pyplot figure axis object
        Axes handle on which to create plot  

    IMAGE_DIR : string
        directory to save plot

    IMAGE_FORMAT : string
        format to save plot. default is 'png'
    
    '''

    if ax is None:  
        fig, ax = plt.subplots()

    flux_im = ax.imshow( flux_map, origin='lower')


    cbar = plt.colorbar( flux_im, ax=ax)
    cbar.ax.tick_params( direction='in', labelsize=10)
    cbar.ax.set_ylabel(r'flux [10$^{-17}$ erg/s/cm$^2$]', fontsize=12)

    ax.set_title( gal_ID + r' $g$-band weighted mean flux')
    ax.set_xlabel('spaxel')
    ax.set_ylabel('spaxel')

    ###########################################################################


    
    if IMAGE_DIR is not None:
        

        #######################################################################
        # Save figure
        #----------------------------------------------------------------------
        plt.savefig( IMAGE_DIR + '/flux/' + gal_ID + '_mflux.' + IMAGE_FORMAT, 
                     format=IMAGE_FORMAT)
        #######################################################################

        #######################################################################
        # Figure cleanup
        #----------------------------------------------------------------------
        plt.cla()
        plt.clf()
        plt.close()
        del cbar, flux_im
        gc.collect()



def plot_diagnostic_panel(flux_map,
                          sigma_map,
                          sigma_corr_map,
                          mvel,
                          gal_ID,
                          IMAGE_DIR=None,
                          IMAGE_FORMAT='png'):
    '''
    Plot a two-by-two paneled image containing the entire rgb image, the masked
    velocity array, the masked model velocity array, and the rotation curve.


    PARAMETERS
    ==========

    flux_map : numpy array of shape (n,n)
        flux data

    sigma_map : array
        stellar velocity dispersion map
    
    sigma_corr_map : array
        corrected and masked stellar velocity dispersion map

    mvel : array
        H-alpha velocity map
    
    gal_ID : string
        galaxy plate-ifu

    IMAGE_DIR : string
        directory to save diagnostic panel. default is None

    IMAGE FORMAT : string
        format to save diagnostic panel. default is 'png'
    '''

    panel_fig, ((flux_panel, mvel_panel),
                (sigma_panel, sigma_corr_panel)) = plt.subplots(2,2)

    panel_fig.set_figheight(8)
    panel_fig.set_figwidth(10)

    #plt.suptitle(gal_ID + ' diagnostic panel', y=1.05, fontsize=16)

    plot_flux_map(gal_ID, flux_map, ax=flux_panel)

    plot_vel(mvel, gal_ID, ax=mvel_panel)
    
    plot_sigma_map(gal_ID, sigma_map, corr=False, ax=sigma_panel)

    plot_sigma_map(gal_ID, sigma_corr_map, corr=True, ax=sigma_corr_panel)

    panel_fig.tight_layout()


    if IMAGE_DIR is not None:
        ########################################################################
        # Create output directory if it does not already exist
        #-----------------------------------------------------------------------
        #if not os.path.isdir(IMAGE_DIR + '/diagnostic_panels'):
        #    os.makedirs(IMAGE_DIR + '/diagnostic_panels')
        ########################################################################

        ########################################################################
        # Save figure
        #-----------------------------------------------------------------------
        plt.savefig(IMAGE_DIR + '/diagnostic_panels/' + gal_ID + '_diagnostic_panel.' + IMAGE_FORMAT,
                    format=IMAGE_FORMAT,
                    #bbox_inches = 'tight', 
                    #pad_inches = 0
                    )
        ########################################################################

        ########################################################################
        # Figure cleanup
        #-----------------------------------------------------------------------
        plt.cla()
        plt.clf()
        plt.close(panel_fig)
        del panel_fig, flux_panel, mvel_panel, sigma_panel, sigma_corr_panel
        gc.collect()
        ########################################################################
