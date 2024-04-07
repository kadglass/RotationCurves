

import numpy as np

import sys
# sys.path.insert(1, '/scratch/kdougla7/RotationCurves/GradientSmoothness-1.0.0/')
sys.path.insert(1, '/Users/kdouglass/Documents/Research/Rotation_curves/RotationCurves/GradientSmoothness-1.0.0/')
from GradientSmoothness import calculate_smoothness




'''
from marvin.tools import Maps


galaxy_ID = '8139-6102'





maps = Maps(plateifu=galaxy_ID)

ha_vel = maps.emline_gval_ha_6564





map_mask = ha_vel.mask

mask = np.zeros(ha_vel.shape, dtype=np.uint8)

mask[map_mask > 10000] = 1



score = how_smooth( ha_vel, mask)




print(score/num_unmasked_spaxels)
'''



################################################################################
################################################################################
################################################################################


def how_smooth(ha_vel, mask):
    '''
    Measure the smoothness of the Halpha velocity map.


    Parameters:
    ===========

    ha_vel : numpy ndarray of shape (n,n)
        Halpha velocity map

    mask : numpy boolean array of shape (n,n)
        Map mask.  Invalid spaxels are marked with True values.


    Returns:
    ========

    score : float
        Rating of how smooth the velocity map is.
    '''


    ############################################################################
    # Compute gradient
    #---------------------------------------------------------------------------
    grad_y, grad_x = np.gradient(ha_vel)
    ############################################################################


    ############################################################################
    # Calculate smoothness score
    #---------------------------------------------------------------------------
    score = calculate_smoothness( grad_x.astype(np.float32), 
                                  grad_y.astype(np.float32), 
                                  mask.astype(np.uint8), 
                                  ha_vel.shape[0], ha_vel.shape[1])
    ############################################################################


    num_unmasked_spaxels = np.sum(mask == 0)

    return score/num_unmasked_spaxels


################################################################################
################################################################################
################################################################################
