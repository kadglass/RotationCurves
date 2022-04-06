
import numpy as np

import os
import gc

import matplotlib.pyplot as plt

import sys
sys.path.insert(1, '/Users/kellydouglass/Documents/Research/Rotation_curves/Yifan_Zhang/RotationCurve/2D_RC/main/')
from rotation_curve_functions import disk_vel




def plot_fitted_disk_rot_curve(gal_ID, 
                               data_table, 
                               fit_parameters, 
                               chi2,
                               IMAGE_DIR=None, 
                               IMAGE_FORMAT='eps'):
    '''
    Plot the fitted rotation curve of disk component


    Parameters:
    ===========

    gal_ID : string
        MaNGA plate number - MaNGA fiberID number

    data_table : Astropy QTable
        Table containing measured rotational velocities at given deprojected 
        radii

    fit_parameters : dictionary
        Contains the values and errors of the best-fit parameters

    chi2 : float
        Reduced chi2 of the best fit

    IMAGE_DIR : string
        Path of directory to store images

    IMAGE_FORMAT : string
        Format of saved image

    ax : matplotlib.pyplot figure axis object
        Axes handle on which to create plot

    '''


    #if ax is None:
    fig = plt.figure(figsize=(8, 5))
    ax = fig.add_axes([0.1, 0.1, 0.65, 0.8]) # [left, bottom, width, height]
    legend_ax = fig.add_axes([0.8, 0.1, 0.2, 0.8])


    ############################################################################
    # Calculate the functional form of the curve
    #---------------------------------------------------------------------------
    r_depro = np.linspace(0, data_table['radius'][-1], 10000)

    v = disk_vel(r_depro, 
                 fit_parameters['Sigma_disk'], 
                 fit_parameters['R_disk'])
    ############################################################################


    ############################################################################
    # Generate the uncertainty range of the best-fit
    #---------------------------------------------------------------------------
    cov = np.load('Pipe3D_diskMass_map_Hessians/' + gal_ID + '_cov.npy')

    N_samples = 10000

    random_sample = np.random.multivariate_normal(mean=[fit_parameters['Sigma_disk'], 
                                                        fit_parameters['R_disk']], 
                                                  cov=cov, 
                                                  size=N_samples)

    # Remove bad samples (those with negative values for any of the parameters)
    is_good_random = (random_sample[:,0] > 0) & (random_sample[:,1] > 0)
    good_randoms = random_sample[is_good_random, :]

    for i in range(len(r_depro)):
        # Calculate the values of the curve at this location
        y_sample = disk_vel(r_depro[i], good_randoms[:,0], good_randoms[:,1])

    stdevs = np.std(y_sample, axis=0)
    ############################################################################


    ############################################################################
    # Plot the fitted disk rotation curve along with its error bars.  In 
    # addition, several statistics about the goodness of fit are displayed in 
    # the lower right side of the figure.
    #---------------------------------------------------------------------------
    # Plot formating
    marker_size = 4
    errorbar_cap_thickness = 1
    errorbar_cap_size = 3

    ax.errorbar(data_table['radius'], 
                data_table['star_vel'], 
                yerr=data_table['star_vel_err'], 
                fmt='D', 
                color='cyan', 
                markersize=marker_size, 
                capthick=errorbar_cap_thickness, 
                capsize=errorbar_cap_size)
    '''
    ax.plot(r_depro, 
            disk_vel([fit_parameters['Sigma_disk'], 
                      fit_parameters['R_disk']], r_depro), 
            'c:')
    '''
    ax.plot(r_depro, v, 'c')

    ax.fill_between(r_depro, v - stdevs, v + stdevs, facecolor='aliceblue')

    ax.tick_params( axis='both', direction='in')
    ax.yaxis.set_ticks_position('both')
    ax.xaxis.set_ticks_position('both')

    ax.set_title( gal_ID + ' Fitted Disk Rotation Curve')
    ax.set_xlabel('$r$ [kpc/h]')
    ax.set_ylabel('$v_{*}$ [km/s]')

    legend_ax.get_xaxis().set_visible(False)
    legend_ax.get_yaxis().set_visible(False)
    legend_ax.patch.set_visible(False)
    legend_ax.axis('off')

    textstr = '\n'.join((
                         r'$\chi^{2}_{\nu}$: $%.3f$' % (chi2, ), 
                         r'$\Sigma_{d}$: $%.1f$ $M_{\odot}$/pc$^2$' % (fit_parameters['Sigma_disk'], ), 
                         r'$R_{d}$: $%.3f$ kpc' % (fit_parameters['R_disk'], )))

    props = dict( boxstyle='round', facecolor='cornsilk', alpha=0.6)

    legend_ax.text( 0, 0.4, textstr,
                    verticalalignment='top', horizontalalignment='left',
                    transform=legend_ax.transAxes,
                    color='black', fontsize=10, bbox=props)
    ###########################################################################


    ###########################################################################
    # Save figure
    #--------------------------------------------------------------------------
    if IMAGE_DIR is not None:
        #######################################################################
        # Create output directory if it does not already exist
        #----------------------------------------------------------------------
        if not os.path.isdir( IMAGE_DIR + '/mass_curves'):
            os.makedirs( IMAGE_DIR + '/mass_curves')
        #######################################################################


        plt.savefig( IMAGE_DIR + "/mass_curves/" + gal_ID + "_fitted_mass_curve." + IMAGE_FORMAT,
                    format=IMAGE_FORMAT)


        #######################################################################
        # Figure cleanup
        #----------------------------------------------------------------------
        plt.cla()
        plt.clf()
        plt.close()
        gc.collect()
        #######################################################################