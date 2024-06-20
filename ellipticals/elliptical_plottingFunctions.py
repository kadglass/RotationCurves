import gc
import numpy as np
import numpy.ma as ma
import matplotlib.pyplot as plt
import matplotlib

from elliptical_stellar_mass_functions import exponential_sphere

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



def plot_stellar_mass(gal_ID,
                              data_table,
                              best_fit_values,
                              COV_DIR,
                              IMAGE_DIR,
                              IMAGE_FORMAT):
    '''
    
    
    
    
    
    '''
    
    plt.errorbar(data_table['radius'], 10**data_table['M_star'], yerr=10**data_table['M_star_err'], 
             color='k', fmt='.')
    r = np.linspace(data_table['radius'][0] , data_table['radius'][-1], 100)

    cov = np.load(COV_DIR + gal_ID + '_cov.npy')
    random_sample = np.random.multivariate_normal(mean=[best_fit_values['rho_c'],
                                                best_fit_values['R_scale']],
                                                cov=cov,
                                                size =1000)

    is_good_random = (random_sample[:,0] > 0) & (random_sample[:,1] > 0) 
    good_randoms = random_sample[is_good_random, :]

    for i in range(len(r)):
        y_sample = exponential_sphere(r[i], good_randoms[:,0], good_randoms[:,1])

            
    stdevs = np.nanstd(y_sample, axis=0)


    y = exponential_sphere(r, best_fit_values['rho_c'],best_fit_values['R_scale'])

    plt.plot(r, y, color='orange')
    plt.fill_between(r, y-stdevs, y+stdevs, facecolor='orange',alpha=0.2)
    plt.xlabel('r [kpc]')
    plt.ylabel(r'Stellar Mass [log(M$_\odot$)]')
    plt.yscale('log')

    # params_str ='\n'.join((r'$\chi^{2}_{\nu}$: $%.3E$' % (best_fit_values['chi2_M_star'], ), 
    #                         r'$\rho_{c}$: $%.3E$ $M_{\odot}$/kpc$^3$' % (best_fit_values['rho_c'], ), 
    #                         r'a: $%.3f$ kpc' % (best_fit_values['R_scale'], )
    #                         ))




    # plt.text(data_table['radius'][-5], 
    #          10**((data_table['M_star'][0] + data_table['M_star'][1])/2), 
    #          params_str  , fontsize = 8, 
    #         bbox = dict(facecolor = 'orange', alpha = 0.5))
    
    plt.savefig(IMAGE_DIR + '/stellar_mass/' + gal_ID + '_stellar_mass.' + IMAGE_FORMAT)

    plt.cla()
    plt.clf()
    plt.close()

def med_err(vals, logscale=True, return_logscale=True):
    '''
    Takes a a list of values in bins, calculates the log median and uncertainty (RMS/sqrt(N)) and returns log median and log err bars

    vals : array
        2D array of dimension (N,M) corresponding to N bins in histogram
    
    '''

    p = np.ones((len(vals),3))*np.nan
    
    # for each bin calculate the median and rms
    for i in range(len(vals)):

        if len(vals[i]) >= 10:
    
            sqrtN = np.sqrt(len(vals[i]))
            
            if logscale:
                median = ma.median(10**vals[i])
                rms = ma.sqrt(ma.mean((10**vals[i])**2))
                
            else:
                
                median = ma.median(vals[i])
                rms = ma.sqrt(ma.mean(vals[i]**2))
    

    
            if return_logscale:
                
                p[i][0] = ma.log10(median)
                p[i][2] = ma.log10(rms/sqrtN + median) - ma.log10(median)
                
                if rms/sqrtN >= median:
                    p[i][1] = np.inf
    
                else:
                    p[i][1] = np.log10(median) - np.log10(median-rms/sqrtN)
                
    
            else:
    
                p[i][0] = median
                p[i][1] = rms/sqrtN
                p[i][2] = rms/sqrtN

    return p
        

def med_percentile(vals, logscale=True, return_logscale=True):
    '''
    take array of vals in bins and calculate median, 16th, 84th percentiles in logscale

    PARAMETERS
    ==========
    val : array
        2d array (N,M) - N is number of bins

    logscale : boolean
        True if vals is in log units

    return_logscale : boolean
        True if percentiles are in log units

    RETURNS
    =======

    p : array
        array of medians, lower and upper bound of error bar (16th/84th percentile)
    
    '''

    p = np.ones((len(vals),3))*np.nan
    
    for i in range(len(vals)):
        if len(vals[i]) == 0:
            continue
        
        if len(vals[i]) >= 3:
            if logscale:
                a, b, c = np.percentile(10**vals[i],[50,16,84])
            else:
                a, b, c = np.percentile(vals[i],[50,16,84])
        else:
            if logscale:
                a = np.median(10**vals[i])
                b = a
                c = a
            else:
                a = np.median(vals[i])
                b = a
                c = a

        low = (a - b)
        high = (c - a)
        
        if return_logscale:
            p[i][0] = np.log10(a)
            p[i][1] = np.log10(a) - np.log10(b)
            p[i][2] = np.log10(c) - np.log10(a)
        
        else:
            p[i][0] = a
            p[i][1] = a - b
            p[i][2] = c - a
    
    return p

    
def med_percentile_scaled(vals, logscale=True, return_logscale=True):
    '''
    take array of vals in bins and calculate median, 16th, 84th percentiles in logscale
    if bin has < 10 data points, return nans

    PARAMETERS
    ==========
    val : array
        2d array (N,M) - N is number of bins

    logscale : boolean
        True if vals is in log units

    return_logscale : boolean
        True if percentiles are in log units

    RETURNS
    =======

    p : array
        median, lower error bar length, upper error bar length, in lin or log units
    '''

    p = np.ones((len(vals),3))*np.nan
    
    for i in range(len(vals)):
        if len(vals[i]) < 10:
            continue
        
        else:
            if logscale:
                a, b, c = np.percentile(10**vals[i],[50,16,84])
            else:
                a, b, c = np.percentile(vals[i],[50,16,84])
        # else:
        #     if logscale:
        #         a = np.median(10**vals[i])
        #         b = a
        #         c = a
        #     else:
        #         a = np.median(vals[i])
        #         b = a
        #         c = a

        low = (a - b) / np.sqrt(len(vals[i])) #lower error bar length (linear)
        high = (c - a) / np.sqrt(len(vals[i])) # upper error bar length (linear)
        
        if return_logscale:
            p[i][0] = np.log10(a)
            p[i][1] = np.log10(a) - np.log10(a-low)
            p[i][2] = np.log10(high+a) - np.log10(a)
        
        else:
            p[i][0] = a
            p[i][1] = low
            p[i][2] = high
    
    return p
