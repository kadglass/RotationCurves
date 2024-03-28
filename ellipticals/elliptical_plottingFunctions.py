import gc
import numpy as np
import numpy.ma as ma
import matplotlib.pyplot as plt

def plot_sigma_map(IMAGE_DIR, IMAGE_FORMAT, gal_ID, sigma_map, corr=False):


    fig, ax = plt.subplots()


    ############################################################################
    # Determine limits of color scale
    #---------------------------------------------------------------------------
    minimum = ma.min(sigma_map)
    maximum = ma.max(sigma_map)

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
    if corr:
        map_type = ' corrected'
    else:
        map_type = ''

    ax.set_title(gal_ID + map_type + ' stellar velocity dispersion', fontsize=18)
    ############################################################################
    

    ############################################################################
    # Create plot
    #---------------------------------------------------------------------------
    vel_im = ax.imshow( sigma_map, 
                        cmap='RdBu_r', 
                        origin='lower', 
                        vmin=vmin_bound, 
                        vmax=vmax_bound)

    cbar = plt.colorbar( vel_im, ax=ax, ticks=cbar_ticks)
    cbar.ax.tick_params( direction='in', labelsize=16)
    #cbar.set_label('$v$ [km/s]')
    cbar.set_label(r'$v_{*}$ dispersion [km/s]', fontsize=18) # formatting for paper

    ax.tick_params( axis='both', direction='in', labelsize=16)
    ax.yaxis.set_ticks_position('both')
    ax.xaxis.set_ticks_position('both')
    ax.set_xlabel('spaxel', fontsize=18)
    ax.set_ylabel('spaxel', fontsize=18)

    '''
    ax.set_xlabel('$\Delta \alpha$ [arcsec]')
    ax.set_ylabel('$\Delta \delta$ [arcsec]')
    '''
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
                     format=IMAGE_FORMAT, bbox_inches = 'tight', pad_inches = 0)
        ########################################################################

        ########################################################################
        # Figure cleanup
        #-----------------------------------------------------------------------
        plt.cla()
        plt.clf()
        plt.close()
        del cbar, vel_im
        gc.collect()


def plot_photo_map(IMAGE_DIR, IMAGE_FORMAT, gal_ID, photo):

    fig, ax = plt.subplots()

    photo_im = ax.imshow( photo, origin='lower')

    cbar = plt.colorbar( photo_im, ax=ax)
    cbar.ax.set_ylabel(r'g-band weighted mean flux [10$^{-17}$ erg/s/cm$^2$]')

    ax.set_title( gal_ID + ' g-band weighted mean')
    ax.set_xlabel('spaxel')
    ax.set_ylabel('spaxel')
    '''
    ax.set_xlabel('$\Delta \alpha$ [arcsec]')
    ax.set_ylabel('$\Delta \delta$ [arcsec]')
    '''
    ###########################################################################


    
    if IMAGE_DIR is not None:
        #######################################################################
        # Create output directory if it does not already exist
        #----------------------------------------------------------------------
        #if not os.path.isdir( IMAGE_DIR + '/unmasked_r_band'):
        #    os.makedirs( IMAGE_DIR + '/unmasked_r_band')
        #######################################################################

        #######################################################################
        # Save figure
        #----------------------------------------------------------------------
        plt.savefig( IMAGE_DIR + '/photo/' + gal_ID + '_mflux.' + IMAGE_FORMAT, 
                     format=IMAGE_FORMAT)
        #######################################################################

        #######################################################################
        # Figure cleanup
        #----------------------------------------------------------------------
        plt.cla()
        plt.clf()
        plt.close()
        del cbar, photo_im
        gc.collect()