import numpy as np

from scipy.stats import norm, ks_2samp

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


###############################################################################
# Plot formatting
#------------------------------------------------------------------------------
#plt.rc('text', usetex=True)
plt.rc('font', size=16)#family='serif')
lwidth = 2 # Line width used in plots
###############################################################################





def DM_SM_hist( void_ratios, wall_ratios, bins=None, hist_range=(0,60), 
                y_max=0.05, y_err=False, 
                plot_title='$M_{DM}$ / $M_*$ distribution', 
                save_fig=False, FILE_SUFFIX='', IMAGE_DIR='', 
                IMAGE_FORMAT='eps'):
    '''
    Histogram the dark matter to stellar mass ratios as separated by 
    environment

    Parameters:
    ===========

    void_ratios : numpy array of shape (n, )
        Ratio of dark matter halo mass to stellar mass for void galaxies

    wall_ratios : numpy array of shape (m, )
        Ratio of dark matter halo mass to stellar mass for wall galaxies

    bins : numpy array of shape (p, )
        Histogram bin edges

    hist_range : tuple
        Minimum and maximum of histogram

    y_max : float
        Upper limit of y-axis; default value is 0.05.

    y_err : boolean
        Determines whether or not to plot sqrt(N) error bars on histogram.  
        Default value is False (no error bars).

    plot_title : string
        Title of plot; default is '$M_{DM}$ / $M_*$ distribution'

    save_fig : boolean
        Flag to determine whether or not the figure should be saved.  Default 
        is False (do not save).

    FILE_SUFFIX : string
        Additional information to include at the end of the figure file name.  
        Default is '' (nothing).

    IMAGE_DIR : string
        Address to directory to save file.  Default is current directory.

    IMAGE_FORMAT : string
        Format for saved image.  Default is 'eps'
    '''

    if bins is None:
        bins = np.linspace(hist_range[0], hist_range[1], 13)

    ###########################################################################
    # Initialize figure and axes
    #--------------------------------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), tight_layout=True)
    ###########################################################################


    ###########################################################################
    # Plot histograms in first figure
    #--------------------------------------------------------------------------
    Ntot_void = sum(void_ratios > 0)
    Ntot_wall = sum(wall_ratios > 0)

    Nv,_ = np.histogram(void_ratios, bins=bins)
    Nw,_ = np.histogram(wall_ratios, bins=bins)

    nv = Nv/Ntot_void
    nw = Nw/Ntot_wall

    ax1.step(bins[:-1], nv, 'r', where='post', linewidth=lwidth, 
             label='Void: ' + str( Ntot_void))
    ax1.step(bins[:-1], nw, 'k', where='post', linewidth=lwidth, linestyle=':', 
             label='Wall: ' + str( Ntot_wall))

    if y_err:
        Nv = Nv.astype('float')
        Nw = Nw.astype('float')

        # Set all 0-counts to infinity
        Nv[Nv==0] = np.infty
        Nw[Nw==0] = np.infty

        ax1.errorbar(0.5*(bins[1:] + bins[:-1]), nv, yerr=Nv**-0.5/Ntot_void, ecolor='r', fmt='none')
        ax1.errorbar(0.5*(bins[1:] + bins[:-1]), nw, yerr=Nw**-0.5/Ntot_wall, ecolor='k', fmt='none')
    #--------------------------------------------------------------------------
    # Histogram plot formatting
    #--------------------------------------------------------------------------
    ax1.set_xlabel(r'$M_{DM}$ / $M_{*}$')
    ax1.set_ylabel('Galaxy fraction')
    ax1.set_title(plot_title)

    ax1.tick_params( axis='both', direction='in')
    ax1.yaxis.set_ticks_position('both')
    ax1.xaxis.set_ticks_position('both')
    ax1.set_xlim( hist_range)
    ax1.set_ylim( (0, y_max))
    #ax1.set_yscale('log')

    ax1.legend()
    ###########################################################################


    ###########################################################################
    # CDF of void and wall mass ratios
    #--------------------------------------------------------------------------
    ks_stat, p_val = ks_2samp( wall_ratios, void_ratios)

    ax2.hist( void_ratios, bins=1000, range=hist_range, density=True, 
             cumulative=True, histtype='step', color='r', linewidth=lwidth, label='Void')
    ax2.hist( wall_ratios, bins=1000, range=hist_range, density=True, 
             cumulative=True, histtype='step', color='k', linewidth=lwidth, linestyle=':', 
             label='Wall')
    #--------------------------------------------------------------------------
    # CDF plot formatting
    #--------------------------------------------------------------------------
    ax2.set_xlabel(r'$M_{DM}$ / $M_*$')
    ax2.set_ylabel('Galaxy fraction')
    ax2.set_title(plot_title)

    ax2.tick_params( axis='both', direction='in')
    ax2.yaxis.set_ticks_position('both')
    ax2.xaxis.set_ticks_position('both')
    ax2.set_xlim( hist_range)
    #ax2.set_xticks( BINS)

    ax2.legend(loc='upper left')

    ax2.text( hist_range[1] * 0.65, 0.15, "p-val: " + "{:.{}f}".format( p_val, 3))
    ###########################################################################
    

    ###########################################################################
    # Save figure?
    #--------------------------------------------------------------------------
    if save_fig:
        plt.savefig( IMAGE_DIR + '/histograms/dm_to_stellar_mass_ratio_hist' + FILE_SUFFIX + '.' + IMAGE_FORMAT,
                    format=IMAGE_FORMAT)
    ###########################################################################
    




###############################################################################
###############################################################################
###############################################################################





def DM_SM_hist_std(ratios, bins, plot_title='$M_{DM}$ / $M_*$ distribution', 
                   save_fig=False, FILE_SUFFIX='', IMAGE_DIR='', IMAGE_FORMAT='eps'):
    '''
    Histogram of mass ratio with 1-, 2-sigma deviations marked

    Parameters:
    ===========

    ratios : numpy array of shape (n, )
        Ratio of dark matter halo mass to stellar mass for galaxies

    bins : numpy array of shape (m, )
        Histogram bin edges

    plot_title : string
        Title of plot; default is '$M_{DM}$ / $M_*$ distribution'

    save_fig : boolean
        Flag to determine whether or not the figure should be saved.  Default 
        is False (do not save).

    FILE_SUFFIX : string
        Additional information to include at the end of the figure file name.  
        Default is '' (nothing).

    IMAGE_DIR : string
        Address to directory to save file.  Default is current directory.

    IMAGE_FORMAT : string
        Format for saved image.  Default is 'eps'
    '''


    ###########################################################################
    # Initialize figure
    #--------------------------------------------------------------------------
    plt.figure()
    ###########################################################################


    ###########################################################################
    # Retrieve plot parameters
    #--------------------------------------------------------------------------
    xmin, xmax = plt.xlim()
    x = np.linspace(xmin, xmax, 100)
    ###########################################################################


    ###########################################################################
    # Compute sample statistics
    #--------------------------------------------------------------------------
    mean = np.mean(ratios)
    std = np.std(ratios)

    p = norm.pdf(x, mean, stdev)
    ###########################################################################


    ###########################################################################
    # 
    #--------------------------------------------------------------------------
    plt.hist( ratios, bins, color='green', density=True, alpha=0.9)
    plt.plot(x, p, 'g--', linewidth=2)
    plt.axvline( mean, color='green', linestyle='-', linewidth=1.5)
    plt.axvline( mean + std, color='green', linestyle=':', linewidth=1)
    plt.axvline( mean - std, color='green', linestyle=':', linewidth=1)
    plt.axvline( mean + 2*std, color='green', linestyle=':', linewidth=1)
    plt.axvline( mean - 2*std, color='green', linestyle=':', linewidth=1)
    _, mean_ratio_ = plt.ylim()
    plt.text(mean + mean/10, mean_ratio_ - mean_ratio_/10, 'Mean: {:.2f}'.format( mean))
    ###########################################################################


    #void_patch = mpatches.Patch( color='red', label='Void: ' + str( len( dm_to_stellar_mass_ratio_void)))
    #wall_patch = mpatches.Patch( color='black', label='Wall: ' + str( len( dm_to_stellar_mass_ratio_wall)), alpha=0.5)
    #plt.legend( handles = [ void_patch, wall_patch])

    # textstr = '\n'.join((
    # #          r'STDEV: $%.2f$' % ( ratio_stdev, ),
    #       r'$STDEV_{wall}$: $%.2f$' % ( ratio_wall_stdev, ),
    #       r'$STDEV_{void}$: $%.2f$' % ( ratio_void_stdev, ),
    # #          r'RMS: $%.2f$' % ( ratio_rms, ),
    #       r'$RMS_{wall}$: $%.2f$' % ( ratio_wall_rms, ),
    #       r'$RMS_{void}$: $%.2f$' % ( ratio_void_rms, )))

    # props = dict( boxstyle='round', facecolor='cornsilk', alpha=0.6)

    # ax.text(0.72, 0.95, textstr,
    #         verticalalignment='top', horizontalalignment='left',
    #         transform=ax.transAxes,
    #         color='black', fontsize=8, bbox=props)


    if save_fig:
        plt.savefig( IMAGE_DIR + '/histograms/dm_to_stellar_mass_ratio_hist_stdev' + file_suffix + '.' + IMAGE_FORMAT,
                     format=IMAGE_FORMAT)