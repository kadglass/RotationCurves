from astropy.table import QTable

import numpy as np

#import matplotlib
#matplotlib.use('TKAgg')
import matplotlib.pyplot as plt

from dark_matter_mass_v2 import rot_fit_func




def plot_fitted_rot_curve( gal_ID, 
                           fit_parameters, 
                           ROT_CURVE_FILE_DIRECTORY, 
                           IMAGE_DIR=None, 
                           IMAGE_FORMAT='eps'):
    '''
    Plot the rotation curve data along with the fitted rotation curve


    Parameters:
    ===========

    gal_ID : string
        MaNGA galaxies ID information in the format [PLATE]-[FIBER_ID]

    fit_parameters : Row of astropy table
        Row from 'master_file.txt' containing the fit parameters for this 
        galaxy

    ROT_CURVE_FILE_DIRECTORY : string
        Location of directory that holds all rotation curve files

    IMAGE_DIR : string
        Location to save plot

    IMAGE_FORMAT : string
        Format for saved image file.  Default is EPS.
    '''

    ###########################################################################
    # Read in galaxy's rotation curve data
    #--------------------------------------------------------------------------
    rot_curve_filename = ROT_CURVE_FILE_DIRECTORY + gal_ID + '_rot_curve_data.txt'

    rot_curve_table = QTable.read(rot_curve_filename, format='ascii.ecsv')
    ###########################################################################


    ###########################################################################
    # Extract necessary data fields for plotting
    #--------------------------------------------------------------------------
    # Make sure to take abs of deprojected distance in plotting
    depro_dist = np.abs(rot_curve_table['deprojected_distance'])

    # Rotation curve velocities
    pos_vel_data = rot_curve_table['max_velocity']
    neg_vel_data = rot_curve_table['min_velocity']

    # Rotation curve velocity errors
    pos_vel_data_err = rot_curve_table['max_velocity_error']
    neg_vel_data_err = rot_curve_table['min_velocity_error']

    # Fitted parameters
    r_turn = fit_parameters['r_turn'].value
    v_max = fit_parameters['v_max'].value
    ###########################################################################


    ###########################################################################
    # Plot the fitted rotation curve along with its errorbars.  In addition,
    # several statistics about the goodness of fit, and mass interior to the 
    # outermost radius recorded are displayed in the lower right side of the 
    # figure.
    #--------------------------------------------------------------------------
    if v_max != -100 and v_max != -999:

        fig = plt.figure(figsize=(8,5))
        ax = fig.add_axes([0.1, 0.1, 0.65, 0.8]) # [left, bottom, width, height]
        legend_ax = fig.add_axes([0.8, 0.1, 0.2, 0.8])

        # x-axis range
        r_depro = np.linspace( -1*depro_dist[-1].value, depro_dist[-1].value, 
                               10000)

        # Plot formating
        marker_size = 4
        errorbar_cap_thickness = 1
        errorbar_cap_size = 3

        # Positive rotation curve data
        pos_points,_,_ = ax.errorbar( depro_dist.data, pos_vel_data.data, 
                                      yerr=pos_vel_data_err.data, fmt='s',
                                      color='red', markersize=marker_size, 
                                      capthick=errorbar_cap_thickness, 
                                      capsize=errorbar_cap_size)

        # Negative rotation curve data
        neg_points,_,_ = ax.errorbar( -1*depro_dist.value, neg_vel_data.data, 
                                      yerr=neg_vel_data_err.data, fmt='^',
                                      color='blue', markersize=marker_size, 
                                      capthick=errorbar_cap_thickness, 
                                      capsize=errorbar_cap_size)

        # Fitted rotation curve
        fit, = ax.plot( r_depro, rot_fit_func(r_depro, v_max, r_turn), ':', 
                        c='orchid')

        ax.tick_params( axis='both', direction='in')
        ax.yaxis.set_ticks_position('both')
        ax.xaxis.set_ticks_position('both')

        ax.set_ylabel('$v_{rot}$ [km/s]')
        ax.set_xlabel('$r_{depro}$ [kpc]')
        ax.set_title( gal_ID + ' Fitted Rotation Curves')


        legend_ax.get_xaxis().set_visible(False)
        legend_ax.get_yaxis().set_visible(False)
        legend_ax.patch.set_visible(False)
        legend_ax.axis('off')

        legend_ax.legend([pos_points, neg_points, fit], 
                         ['Positive', 'Negative', 'Best fit'], loc=2)

        if not fit_parameters['good_fit']:
            textstr = 'No mass estimate'
        else:
            textstr = '\n'.join((
                    'Points removed: %d' % (fit_parameters['points_cut'], ),
                    r'$\chi^{2}$/ndf: $%.3f$' % ( fit_parameters[ 'chi_square_ndf'], ),
                    #r'$\Delta v_{max}$: $%.3f$' % ( v_max_pos - v_max_neg),
                    r'$R_{turn}$: $%.1f$' % ( fit_parameters[ 'r_turn'].value, ),
                    r'$V_{max}$: $%.1f$' % ( fit_parameters[ 'v_max'].value, ),
                    r'$M_{tot}$ [$M_{\odot}$]: $%9.2E$' % ( fit_parameters['Mtot'].value, ),
                    r'$M_{*}$ [$M_{\odot}$]: $%9.2E$' % ( fit_parameters['Mstar'].value, ),
                    r'$\frac{M_{DM}}{M_{*}}$: $%.3f$' % ( fit_parameters['Mdark_Mstar_ratio'], )))

        props = dict( boxstyle='round', facecolor='cornsilk', alpha=0.6)

        legend_ax.text( 0, 0.4, textstr,
                        verticalalignment='top', horizontalalignment='left',
                        transform=legend_ax.transAxes,
                        color='black', fontsize=10, bbox=props)

        if IMAGE_DIR is not None:
            plt.savefig( IMAGE_DIR + 'fitted_rotation_curves/' + gal_ID + '_fitted_rotation_curve-tanh.' + IMAGE_FORMAT,
                         format=IMAGE_FORMAT)
            plt.close()
        else:
            plt.show()
    ###########################################################################








def plot_fitted_rot_curve_mass( gal_ID, 
                                fit_parameters, 
                                data_table, 
                                ROT_CURVE_FILE_DIRECTORY,
                                DM_plot=True, 
                                IMAGE_DIR=None, 
                                IMAGE_FORMAT='eps'):
    '''
    Plot the rotation curve data along with the fitted rotation curve.  Also 
    includes the stellar mass and (optional) dark matter mass curves.


    Parameters:
    ===========

    gal_ID : string
        MaNGA galaxies ID information in the format [PLATE]-[FIBER_ID]

    fit_parameters : Row of astropy table
        Row from 'master_file.txt' containing the fit parameters for this 
        galaxy

    data_table : astropy QTable
        Contains the deprojected distance; maximum, minimum, average, stellar, 
        and dark matter velocities at that radius; difference between the 
        maximum and minimum velocities; and the stellar, dark matter, and total 
        mass interior to that radius as well as the errors associated with each 
        quantity as available

    ROT_CURVE_FILE_DIRECTORY : string
        Location of directory that holds all rotation curve files

    DM_plot : boolean
        Flag of whether or not to include the dark matter rotation curve.  
        Default value is True (include DM rotation curve).

    IMAGE_DIR : string
        Location to save plot

    IMAGE_FORMAT : string
        Format for saved image file.  Default is EPS.
    '''

    ###########################################################################
    # Read in galaxy's rotation curve data
    #--------------------------------------------------------------------------
    rot_curve_filename = ROT_CURVE_FILE_DIRECTORY + gal_ID + '_rot_curve_data.txt'

    rot_curve_table = QTable.read(rot_curve_filename, format='ascii.ecsv')
    ###########################################################################


    ###########################################################################
    # Extract necessary data fields for plotting
    #--------------------------------------------------------------------------
    # Make sure to take abs of deprojected distance in plotting
    depro_dist = np.abs(rot_curve_table['deprojected_distance'])

    # Positive rotation curve
    pos_vel_data = rot_curve_table['max_velocity']
    pos_vel_data_err = rot_curve_table['max_velocity_error']

    # Negative rotation curve
    neg_vel_data = rot_curve_table['min_velocity']
    neg_vel_data_err = rot_curve_table['min_velocity_error']

    # Best fit values
    r_turn = fit_parameters['r_turn'].value
    v_max = fit_parameters['v_max'].value
    ###########################################################################


    ###########################################################################
    # Plot the fitted rotation curve along with its errorbars. In addition,
    # several statistics about the goodness of fit, and mass interior to the 
    # outermost radius recorded are displayed in the lower right side of the 
    # figure.
    #--------------------------------------------------------------------------
    if v_max_avg != -100 and v_max_avg != -999:

        plt.figure(figsize=(7,5))

        # x-axis range
        r_depro = np.linspace( -1*depro_dist[-1].value, depro_dist[-1].value, 
                               10000)

        # Plot formating
        marker_size = 4
        errorbar_cap_thickness = 1
        errorbar_cap_size = 3
        plt.rcParams.update({'font.size': 14})

        # Stellar mass rotation curve
        Mstar, = plt.plot( data_table['deprojected_distance'], 
                           data_table['sVel_rot'], 
                           'cD', markersize=marker_size)
        plt.plot( -1*data_table['deprojected_distance'], 
                  -1*data_table['sVel_rot'], 
                  'cD', markersize=marker_size)

        # Dark matter rotation curve
        if DM_plot:
            Mdark, = plt.plot( data_table['deprojected_distance'], 
                               data_table['dmVel_rot'], 
                               'kX', 
                               markersize=marker_size)
            plt.plot( -1*data_table['deprojected_distance'], 
                      -1*data_table['dmVel_rot'], 
                      'kX', markersize=marker_size)

        # Positive rotation curve
        pos_points,_,_ = plt.errorbar( depro_dist.data, pos_vel_data.data, 
                                       yerr=pos_vel_data_err.data, fmt='s',
                                       color='red', markersize=marker_size, 
                                       capthick=errorbar_cap_thickness, 
                                       capsize=errorbar_cap_size)

        # Negative rotation curve
        neg_points,_,_ = plt.errorbar( -1*depro_dist.value, neg_vel_data.data, 
                                      yerr=neg_vel_data_err.data, fmt='^',
                                      color='blue', markersize=marker_size, 
                                      capthick=errorbar_cap_thickness, 
                                      capsize=errorbar_cap_size)

        # Fitted rotation curve
        fit, = plt.plot( r_depro, rot_fit_func(r_depro, v_max, r_turn), ':', 
                         c='violet')

        plt.tick_params( axis='both', direction='in')

        plt.ylabel('$v_{rot}$ [km/s]')
        plt.xlabel('$r_{depro}$ [kpc/$h$]')
        plt.title( gal_ID + ' Fitted Rotation Curves')


        if DM_plot:
            artist_list = [pos_points, neg_points, fit, Mstar, Mdark]
            artist_labels = ['Positive', 'Negative', 'Best fit', '$M_*$', '$M_{DM}$']
        else:
            artist_list = [pos_points, neg_points, fit, Mstar]
            artist_labels = ['Positive', 'Negative', 'Best fit', '$M_*$']
        
        plt.legend( artist_list, artist_labels, loc=2)


        if IMAGE_DIR is not None:
            plt.savefig( IMAGE_DIR + 'fitted_rotation_curves/' + gal_ID + '_fitted_rotation_curve-tanh_mass.' + IMAGE_FORMAT,
                         format=IMAGE_FORMAT)
        else:
            plt.show()

        #plt.close()
    ###########################################################################