
import numpy as np

from astropy.table import QTable

from dark_matter_mass_v1 import rot_fit_BB



################################################################################
# Loss function
#-------------------------------------------------------------------------------
def loss_func(x_data, y_data, params):
    '''
    Calculate the loss on a given galaxy, where
      loss = (1/N) sum (f(x_i) - y_i)^2
    
    
    PARAMETERS
    ==========
    
    x_data : ndarray
        X data points (radius)
        
    y_data : ndarray
        Y data points (velocity)
        
    params : list or ndarray
        Set of parameter values belonging to the best fit.
        [Vmax, Rturn, alpha]
        
        
    RETURNS
    =======
    
    loss : scalar
        The square of the difference between y_data and y_fit(x_data), 
        normalized by the number of points.
    '''
    
    
    y_fit = rot_fit_BB(x_data, params)
    
    loss = np.sum((y_fit - y_data)**2)/len(x_data)
    
    return loss
################################################################################




################################################################################
# Data
#-------------------------------------------------------------------------------
ROT_CURVE_FILE_DIRECTORY = 'Pipe3D-rot_curve_data_files/'

#filename = 'Pipe3D-master_file_vflag_10_smooth2p27.txt'
filename = 'Pipe3D-master_file_vflag_BB_chi10_alpha2p5_smooth2p27.txt'

data = QTable.read(filename, format='ascii.ecsv')
################################################################################



################################################################################
# Calculate loss for each galaxy
#-------------------------------------------------------------------------------
data['best_fit_loss'] = np.NaN*np.ones(len(data), dtype=float)


for i in range(len(data)):

    curve_used = data['curve_used'][i]

    if curve_used not in ['non', 'none']:

        #-----------------------------------------------------------------------
        # Extract best-fit parameters
        #-----------------------------------------------------------------------
        Vmax = data[curve_used + '_v_max'][i].value
        Rturn = data[curve_used + '_r_turn'][i].value
        alpha = data[curve_used + '_alpha'][i]
        #-----------------------------------------------------------------------


        #-----------------------------------------------------------------------
        # Read in rotation curve data
        #-----------------------------------------------------------------------
        plate = data['MaNGA_plate'][i]
        IFU = data['MaNGA_IFU'][i]
        gal_ID = str(plate) + '-' + str(IFU)

        rot_curve_filename = ROT_CURVE_FILE_DIRECTORY + gal_ID + '_rot_curve_data.txt'

        rot_curve_table = QTable.read(rot_curve_filename, format='ascii.ecsv')
        #-----------------------------------------------------------------------


        #-----------------------------------------------------------------------
        # Extract x, y data points
        #-----------------------------------------------------------------------
        radius = np.abs(rot_curve_table['deprojected_distance'])

        if curve_used == 'avg':
            rot_vel = rot_curve_table['rot_vel_avg']
        elif curve_used == 'pos':
            rot_vel = rot_curve_table['max_velocity']
        elif curve_used == 'neg':
            rot_vel = np.abs(rot_curve_table['min_velocity'])
        #-----------------------------------------------------------------------


        #-----------------------------------------------------------------------
        # Calculate loss
        #-----------------------------------------------------------------------
        data['best_fit_loss'][i] = loss_func(radius.data, 
                                             rot_vel.data, 
                                             [Vmax, Rturn, alpha])
        #-----------------------------------------------------------------------
################################################################################



################################################################################
# Save results
#-------------------------------------------------------------------------------
data.write(filename[:-4] + '_loss.txt', format='ascii.ecsv')
################################################################################








