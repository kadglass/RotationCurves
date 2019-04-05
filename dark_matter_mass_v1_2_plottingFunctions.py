from astropy.table import Table

import numpy as np

import matplotlib.pyplot as plt

from dark_matter_mass_v1_2 import rot_fit_func




def plot_fitted_rot_curve( gal_ID, fit_parameters, IMAGE_DIR=None, IMAGE_FORMAT='eps'):
    '''
    Plot the rotation curve data along with the fitted rotation curve


    Parameters:
    ===========

    gal_ID : string
        MaNGA galaxies ID information in the format [PLATE]-[FIBER_ID]

    fit_parameters : Row of astropy table
        Row from 'master_file.txt' containing the fit parameters for this 
        galaxy

    IMAGE_DIR : string
        Location to save plot

    IMAGE_FORMAT : string
        Format for saved image file.  Default is EPS.
    '''

    ###########################################################################
    # Read in galaxy's rotation curve data
    #--------------------------------------------------------------------------
    rot_curve_filename = 'rot_curve_data_files/' + gal_ID + '_rot_curve_data.txt'

    rot_curve_table = Table.read(rot_curve_filename, format='ascii.ecsv')
    #--------------------------------------------------------------------------


    ###########################################################################
    # Extract necessary data fields for plotting
    #--------------------------------------------------------------------------
    # Make sure to take abs of deprojected distance in plotting
    depro_dist = np.abs(rot_curve_table['deprojected_distance'])

    # Average rotation curve
    rot_vel_data = rot_curve_table['rot_vel_avg']
    rot_vel_data_err = rot_curve_table['rot_vel_avg_error']
    r_turn_best = fit_parameters['avg_r_turn']
    alpha_best = fit_parameters['avg_alpha']
    v_max_best = fit_parameters['avg_v_max']

    # Positive rotation curve
    pos_vel_data = rot_curve_table['max_velocity']
    pos_vel_data_err = rot_curve_table['max_velocity_error']
    r_turn_pos = fit_parameters['pos_r_turn']
    alpha_pos = fit_parameters['pos_alpha']
    v_max_pos = fit_parameters['pos_v_max']

    # Negative rotation curve
    neg_vel_data = rot_curve_table['min_velocity']
    neg_vel_data_err = rot_curve_table['min_velocity_error']
    r_turn_neg = fit_parameters['neg_r_turn']
    alpha_neg = fit_parameters['neg_alpha']
    v_max_neg = fit_parameters['neg_v_max']
    #--------------------------------------------------------------------------


    ###########################################################################
    # Plot the fitted rotation curve along with its errorbars. In addition,
    # several statistics about the goodness of fit, and mass interior to the 
    # outermost radius recorded are displayed in the lower right side of the 
    # figure.
    #--------------------------------------------------------------------------
    if v_max_best != -100 and v_max_best != -999:

        #fig, ax = plt.subplots()
        fig = plt.figure(figsize=(8,5))
        ax = fig.add_axes([0.1, 0.1, 0.65, 0.8]) # [left, bottom, width, height]
        legend_ax = fig.add_axes([0.8, 0.1, 0.2, 0.8])

        # Positive rotation curve
        pos_points,_,_ = ax.errorbar( depro_dist, pos_vel_data, yerr=pos_vel_data_err, fmt='s',
                      color='red', markersize=4, capthick=1, capsize=3)
        pos_fit, = ax.plot( np.linspace( 0, depro_dist[-1], 10000),
                  rot_fit_func( np.linspace( 0, depro_dist[-1], 10000),
                                v_max_pos, r_turn_pos, alpha_pos),
                                color='red', linestyle=':', label='Positive')

        # Negative rotation curve
        neg_points,_,_ = ax.errorbar( depro_dist, np.abs(neg_vel_data), yerr=neg_vel_data_err, fmt='^',
                      color='blue', markersize=4, capthick=1, capsize=3)
        neg_fit, = ax.plot( np.linspace( 0, depro_dist[-1], 10000),
                  rot_fit_func( np.linspace( 0, depro_dist[-1], 10000),
                                v_max_neg, r_turn_neg, alpha_neg),
                                color='blue', linestyle=':', label='Negative')

        # Average rotation curve
        avg_points,_,_ = ax.errorbar( depro_dist, rot_vel_data, yerr=rot_vel_data_err, fmt='p', 
                      color='green', markersize=4, capthick=1, capsize=3)
        avg_fit, = ax.plot( np.linspace( 0, depro_dist[-1], 10000),
             rot_fit_func(np.linspace( 0, depro_dist[-1], 10000),
                          v_max_best,
                          r_turn_best,
                          alpha_best),
                          color='green', linestyle=':', label='Average')

        ax.tick_params( axis='both', direction='in')
        ax.yaxis.set_ticks_position('both')
        ax.xaxis.set_ticks_position('both')

        ax.set_ylabel('$v_{rot}$ [km/s]')
        ax.set_xlabel('$d_{depro}$ [kpc]')
        ax.set_title( gal_ID + ' Fitted Rotation Curves')


        legend_ax.get_xaxis().set_visible(False)
        legend_ax.get_yaxis().set_visible(False)
        legend_ax.patch.set_visible(False)
        legend_ax.axis('off')

        legend_ax.legend([(pos_points, pos_fit), (neg_points, neg_fit), (avg_points, avg_fit)], ['Positive', 'Negative', 'Average'], loc=2)

        if fit_parameters['curve_used'][:3] == 'non':
            textstr = 'No mass estimate'
        else:
            textstr = '\n'.join((
                    'Curve used: %s' % (fit_parameters['curve_used'], ),
                    'Points removed: %d' % (fit_parameters['points_cut'], ),
                    r'$\chi^{2}$/ndf: $%.3f$' % ( fit_parameters[ fit_parameters['curve_used'] + '_chi_square_ndf'], ),
                    #r'$\Delta v_{max}$: $%.3f$' % ( v_max_pos - v_max_neg),
                    r'$M_{DM}$ [$M_{\odot}$]: $%9.2E$' % ( fit_parameters['Mdark'], ),
                    r'$M_{*}$ [$M_{\odot}$]: $%9.2E$' % ( fit_parameters['Mstar'], ),
                    r'$\frac{M_{DM}}{M_{*}}$: $%.3f$' % ( fit_parameters['Mdark_Mstar_ratio'], )))

        props = dict( boxstyle='round', facecolor='cornsilk', alpha=0.6)

        legend_ax.text( 0, 0.34, textstr,
                        verticalalignment='top', horizontalalignment='left',
                        transform=legend_ax.transAxes,
                        color='black', fontsize=10, bbox=props)

        if IMAGE_DIR is not None:
            plt.savefig( IMAGE_DIR + 'fitted_rotation_curves/' + gal_ID + '_fitted_rotation_curve.' + IMAGE_FORMAT,
                         format=IMAGE_FORMAT)
        else:
            plt.show()

        plt.close()
    ###########################################################################