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

label_dict = {'Mdark_Mstar_ratio':'$M_{DM}/M_*$',
              'Mtot':'$M_{tot}$ [$M_\odot$]',
              'Mstar':'$M_*$ [$M_\odot$]',
              'rabsmag': '$M_r$',
              'Z12logOH': '12 + log(O/H)',
              'Rmax': '$R_{max}$ [kpc]',
              'avg_r_turn': '$R_{turn}$ [kpc]'}
###############################################################################



def median_hist( void, wall, x_param, y_param, bins, error_linestyle=['', ':'],
                 save_fig=False, IMAGE_DIR='', IMAGE_FORMAT='eps'):
    '''
    Bin the data in x and calculate the median of the y property in each bin.  
    Plot with associated standard deviation as y error bars, and bin width as 
    x error bars.


    PARAMETERS
    ==========

    void : astropy table of length n
        Table of parameters for void galaxies

    wall : astropy table of length m
        Table of parameters for wall galaxies

    x_param : string
        Galaxy parameter to plot on the x-axis

    y_param : string
        Galaxy parameter to plot on the y_axis

    bins : numpy array of shape (p,)
        Bin edges in x parameter

    error_linestyle : length-2 list of strings
        Line styling for the error bars.  Order is [void, wall].  Default 
        styling is solid for void and dotted for wall.

    save_fig : boolean
        Flag to determine whether or not the figure should be saved.  Default 
        is False (do not save).

    IMAGE_DIR : string
        Address to directory to save file.  Default is current directory.

    IMAGE_FORMAT : string
        Format for saved image.  Default is 'eps'
    '''

    ###########################################################################
    # Initialize figure
    #--------------------------------------------------------------------------
    #plt.figure(tight_layout=True)
    ###########################################################################


    ###########################################################################
    # Bin galaxies
    #--------------------------------------------------------------------------
    i_bin_void = np.digitize(void[x_param], bins)
    i_bin_wall = np.digitize(wall[x_param], bins)
    ###########################################################################


    ###########################################################################
    # Calculate bin statistics
    #--------------------------------------------------------------------------
    median_void = np.zeros(len(bins))
    median_wall = np.zeros(len(bins))

    std_void = np.zeros(len(bins))
    std_wall = np.zeros(len(bins))

    for i in range(len(bins)):
        # Bin median
        median_void[i] = np.median(void[y_param][i_bin_void == i])
        median_wall[i] = np.median(wall[y_param][i_bin_wall == i])

        # Bin standard deviation
        std_void[i] = np.std(void[y_param][i_bin_void == i])/len(void[i_bin_void == i])
        std_wall[i] = np.std(wall[y_param][i_bin_wall == i])/len(wall[i_bin_wall == i])
    ###########################################################################


    ###########################################################################
    # Plot scatter plot
    #
    # x error bars are the width of the bins
    # y error bars are the standard deviation of the y-values in each bin
    #--------------------------------------------------------------------------
    bin_width = 0.5*(bins[1] - bins[0])

    # Void galaxies
    v = plt.errorbar(bins + bin_width, median_void, 
                     xerr=bin_width, yerr=std_void, 
                     marker='o', mfc='r', ms=200, 
                     ecolor='r', fmt='none')
    plt.plot(bins + bin_width, median_void, 'ro', label='Void')
    if error_linestyle[0] is not '':
        v[-1][0].set_linestyle(error_linestyle[0])
        v[-1][1].set_linestyle(error_linestyle[0])

    # Wall galaxies
    w = plt.errorbar(bins + bin_width, median_wall, 
                     xerr=bin_width, yerr=std_wall,
                     marker='^', mfc='k', ms=200,
                     ecolor='k', fmt='none')
    w[-1][0].set_linestyle(error_linestyle[1])
    w[-1][1].set_linestyle(error_linestyle[1])
    plt.plot(bins + bin_width, median_wall, 'k^', label='Wall')

    
    plt.xlabel(label_dict[x_param])
    plt.ylabel('median ' + label_dict[y_param])

    plt.legend()
    ###########################################################################


    ###########################################################################
    # Save figure?
    #--------------------------------------------------------------------------
    if save_fig:
        plt.savefig( IMAGE_DIR + '/histograms/' + x_param + '-' + y_param + '.' + IMAGE_FORMAT,
                    format=IMAGE_FORMAT)
    ###########################################################################



###############################################################################
###############################################################################
###############################################################################



def param_hist_scatter( void, wall, field, bins, save_fig=False, IMAGE_DIR='', 
                        IMAGE_FORMAT='eps'):
    '''
    Histogram the specified field parameter as separated by environment.


    PARAMETERS
    ==========

    void : astropy table of length n
        Table of parameters for void galaxies

    wall : astropy table of length m
        Table of parameters for wall galaxies

    param : string
        Galaxy parameter to histogram

    bins : numpy array of shape (p,)
        Histogram bin edges

    save_fig : boolean
        Flag to determine whether or not the figure should be saved.  Default 
        is False (do not save).

    IMAGE_DIR : string
        Address to directory to save file.  Default is current directory.

    IMAGE_FORMAT : string
        Format for saved image.  Default is 'eps'
    '''


    bin_step = bins[1] - bins[0]


    ###########################################################################
    # Histrogram of samples
    #--------------------------------------------------------------------------
    Ntot_void = len(void)
    Ntot_wall = len(wall)

    Nv,_ = np.histogram(void[field], bins=bins)
    Nw,_ = np.histogram(wall[field], bins=bins)

    nv = Nv/Ntot_void
    nw = Nw/Ntot_wall

    # Change integer counts to floats
    Nv = Nv.astype('float')
    Nw = Nw.astype('float')

    # Set 0-counts to equal infinity
    Nv[Nv==0] = np.infty
    Nw[Nw==0] = np.infty
    ###########################################################################


    ###########################################################################
    # Plot scatter plot of histogram
    #--------------------------------------------------------------------------
    # Void
    plt.plot(bins[:-1] + 0.5*bin_step, nv, 'ro', label='Void')
    plt.errorbar(bins[:-1] + 0.5*bin_step, nv, xerr=0.5*bin_step, 
                 yerr=1/(Ntot_void*np.sqrt(Nv)), 
                 ecolor='r', fmt='none')

    # Wall
    plt.plot(bins[:-1] + 0.5*bin_step, nw, 'k^', label='Wall')
    w = plt.errorbar(bins[:-1] + 0.5*bin_step, nw, xerr=0.5*bin_step, 
                     yerr=1/(Ntot_wall*np.sqrt(Nw)),
                     ecolor='k', fmt='none')
    w[-1][0].set_linestyle(':')
    w[-1][1].set_linestyle(':')

    plt.xlabel(label_dict[field])
    plt.ylabel('Fraction')

    plt.legend()
    ###########################################################################


    ###########################################################################
    # Save figure?
    #--------------------------------------------------------------------------
    if save_fig:
        plt.savefig( IMAGE_DIR + '/histograms/' + field + '_scatter_hist.' + IMAGE_FORMAT,
                     format=IMAGE_FORMAT)
    ###########################################################################



###############################################################################
###############################################################################
###############################################################################



def param_hist( void, wall, field, bins, save_fig=False, IMAGE_DIR='', 
                IMAGE_FORMAT='eps'):
    '''
    Histogram the specified field parameter as separated by environment.


    PARAMETERS
    ==========

    void : astropy table of length n
        Table of parameters for void galaxies

    wall : astropy table of length m
        Table of parameters for wall galaxies

    param : string
        Galaxy parameter to histogram

    bins : numpy array of shape (p,)
        Histogram bin edges

    save_fig : boolean
        Flag to determine whether or not the figure should be saved.  Default 
        is False (do not save).

    IMAGE_DIR : string
        Address to directory to save file.  Default is current directory.

    IMAGE_FORMAT : string
        Format for saved image.  Default is 'eps'
    '''


    ###########################################################################
    # Histrogram of samples
    #--------------------------------------------------------------------------
    Ntot_void = len(void)
    Ntot_wall = len(wall)

    Nv,_ = np.histogram(void[field], bins=bins)
    Nw,_ = np.histogram(wall[field], bins=bins)

    nv = Nv/Ntot_void
    nw = Nw/Ntot_wall
    ###########################################################################


    ###########################################################################
    # Plot step histogram
    #--------------------------------------------------------------------------
    plt.step(bins[:-1], nv, 'r', where='post', linewidth=lwidth, 
             label='Void: ' + str( Ntot_void))
    plt.step(bins[:-1], nw, 'k', where='post', linewidth=lwidth, linestyle=':', 
             label='Wall: ' + str( Ntot_wall))
    #--------------------------------------------------------------------------
    # Histogram plot formatting
    #--------------------------------------------------------------------------
    plt.xlabel(label_dict[field])
    plt.ylabel('Fraction')

    plt.legend()
    ###########################################################################


    ###########################################################################
    # Save figure?
    #--------------------------------------------------------------------------
    if save_fig:
        plt.savefig( IMAGE_DIR + '/histograms/' + field + '_hist.' + IMAGE_FORMAT,
                     format=IMAGE_FORMAT)
    ###########################################################################



###############################################################################
###############################################################################
###############################################################################



def param_CDF( void, wall, field, field_range, save_fig=False, IMAGE_DIR='', 
               IMAGE_FORMAT='eps'):
    '''
    Histogram the specified field parameter as separated by environment.


    PARAMETERS
    ==========

    void : astropy table of length n
        Table of parameters for void galaxies

    wall : astropy table of length m
        Table of parameters for wall galaxies

    field : string
        Galaxy parameter to histogram

    field_range : tuple
        Minimum and maximum of data values

    save_fig : boolean
        Flag to determine whether or not the figure should be saved.  Default 
        is False (do not save).

    IMAGE_DIR : string
        Address to directory to save file.  Default is current directory.

    IMAGE_FORMAT : string
        Format for saved image.  Default is 'eps'
    '''


    ###########################################################################
    # CDF of void and wall
    #--------------------------------------------------------------------------
    ks_stat, p_val = ks_2samp( wall[field], void[field])

    plt.hist( void[field], bins=1000, range=field_range, density=True, 
              cumulative=True, histtype='step', color='r', linewidth=lwidth, 
              label='Void')
    plt.hist( wall[field], bins=1000, range=field_range, density=True, 
              cumulative=True, histtype='step', color='k', linewidth=lwidth, 
              linestyle=':', label='Wall')
    #--------------------------------------------------------------------------
    # CDF plot formatting
    #--------------------------------------------------------------------------
    plt.xlabel(label_dict[field])
    plt.ylabel('Fraction')
    
    plt.legend(loc='upper left')

    plt.annotate( "p-val: " + "{:.{}f}".format( p_val, 3), (field_range[1] * 0.65, 0.15))
    ###########################################################################


    ###########################################################################
    # Save figure?
    #--------------------------------------------------------------------------
    if save_fig:
        plt.savefig( IMAGE_DIR + '/histograms/' + field + '_CDF.' + IMAGE_FORMAT,
                     format=IMAGE_FORMAT)
    ###########################################################################



###############################################################################
###############################################################################
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
        bins = np.linspace(hist_range[0], hist_range[1], 20)

    bin_step = bins[1] - bins[0]

    ###########################################################################
    # Initialize figure and axes
    #--------------------------------------------------------------------------
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5), tight_layout=True)
    ###########################################################################


    ###########################################################################
    # Histrogram of samples
    #--------------------------------------------------------------------------
    Ntot_void = sum(void_ratios > 0)
    Ntot_wall = sum(wall_ratios > 0)

    Nv,_ = np.histogram(void_ratios, bins=bins)
    Nw,_ = np.histogram(wall_ratios, bins=bins)

    nv = Nv/Ntot_void
    nw = Nw/Ntot_wall

    # Change integer counts to floats
    Nv = Nv.astype('float')
    Nw = Nw.astype('float')

    # Set 0-counts to equal infinity
    Nv[Nv==0] = np.infty
    Nw[Nw==0] = np.infty
    ###########################################################################


    ###########################################################################
    # Plot scatter plot of histogram in first figure
    #--------------------------------------------------------------------------
    ax1.semilogy(bins[:-1] + 0.5*bin_step, nv, 'ro', label='Void')
    ax1.semilogy(bins[:-1] + 0.5*bin_step, nw, 'k^', label='Wall')

    ax1.errorbar(bins[:-1] + 0.5*bin_step, nv, yerr=1/(Ntot_void*np.sqrt(Nv)), 
                 ecolor='r', fmt='none')
    ax1.errorbar(bins[:-1] + 0.5*bin_step, nw, yerr=1/(Ntot_wall*np.sqrt(Nw)),
                 ecolor='k', fmt='none')

    ax1.set_xlabel(r'$M_{DM}$ / $M_{*}$')
    ax1.set_ylabel('Fraction')

    ax1.set_xlim( hist_range)
    ax1.set_ylim( (0.001, 1))

    ax1.legend()
    ###########################################################################


    ###########################################################################
    # Plot histograms in second figure
    #--------------------------------------------------------------------------
    ax2.step(bins[:-1], nv, 'r', where='post', linewidth=lwidth, 
             label='Void: ' + str( Ntot_void))
    ax2.step(bins[:-1], nw, 'k', where='post', linewidth=lwidth, linestyle=':', 
             label='Wall: ' + str( Ntot_wall))

    if y_err:
        Nv = Nv.astype('float')
        Nw = Nw.astype('float')

        # Set all 0-counts to infinity
        Nv[Nv==0] = np.infty
        Nw[Nw==0] = np.infty

        ax2.errorbar(0.5*(bins[1:] + bins[:-1]), nv, yerr=Nv**-0.5/Ntot_void, ecolor='r', fmt='none')
        ax2.errorbar(0.5*(bins[1:] + bins[:-1]), nw, yerr=Nw**-0.5/Ntot_wall, ecolor='k', fmt='none')
    #--------------------------------------------------------------------------
    # Histogram plot formatting
    #--------------------------------------------------------------------------
    ax2.set_xlabel(r'$M_{DM}$ / $M_{*}$')
    ax2.set_ylabel('Galaxy fraction')
    ax2.set_title(plot_title)

    ax2.tick_params( axis='both', direction='in')
    ax2.yaxis.set_ticks_position('both')
    ax2.xaxis.set_ticks_position('both')
    ax2.set_xlim( hist_range)
    ax2.set_ylim( (0, y_max))
    #ax2.set_yscale('log')

    ax2.legend()
    ###########################################################################


    ###########################################################################
    # CDF of void and wall mass ratios
    #--------------------------------------------------------------------------
    ks_stat, p_val = ks_2samp( wall_ratios, void_ratios)

    ax3.hist( void_ratios, bins=1000, range=hist_range, density=True, 
             cumulative=True, histtype='step', color='r', linewidth=lwidth, label='Void')
    ax3.hist( wall_ratios, bins=1000, range=hist_range, density=True, 
             cumulative=True, histtype='step', color='k', linewidth=lwidth, linestyle=':', 
             label='Wall')
    #--------------------------------------------------------------------------
    # CDF plot formatting
    #--------------------------------------------------------------------------
    ax3.set_xlabel(r'$M_{DM}$ / $M_*$')
    ax3.set_ylabel('Galaxy fraction')
    ax3.set_title(plot_title)

    ax3.tick_params( axis='both', direction='in')
    ax3.yaxis.set_ticks_position('both')
    ax3.xaxis.set_ticks_position('both')
    ax3.set_xlim( hist_range)
    #ax3.set_xticks( BINS)

    ax3.legend(loc='upper left')

    ax3.text( hist_range[1] * 0.65, 0.15, "p-val: " + "{:.{}f}".format( p_val, 3))
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