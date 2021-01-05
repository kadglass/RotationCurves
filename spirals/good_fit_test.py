
'''
Script to set the "good_fit" field based on the original fit to the formula from 
Barrera-Ballestros (BB) without introducing any loss penalty on the values of 
the individual parameters.

A galaxy is considered to have a bad fit if any of the following are true:
  - Rturn = 0
  - Rturn > 200
  - Vmax > 20,000
These galaxies will have "good_fit" = 0 (False).
'''


from astropy.table import Table

import numpy as np



################################################################################
# Data
#-------------------------------------------------------------------------------
filename = 'Pipe3D-master_file_vflag_10_smooth2p27_N2O2_noWords.txt'

data = Table.read(filename, format='ascii.commented_header')
################################################################################



################################################################################
# Set "good_fit" value for each galaxy
#-------------------------------------------------------------------------------
data['good_fit'] = np.zeros(len(data), dtype=int)


for i in range(len(data)):

    curve_used = data['curve_used'][i]

    # Positive rotation curve
    if curve_used == 1:
        Rturn_used = data['pos_r_turn'][i]
        Vmax_used = data['pos_v_max'][i]
    # Average rotation curve
    elif curve_used == 0:
        Rturn_used = data['avg_r_turn'][i]
        Vmax_used = data['avg_v_max'][i]
    # Negative rotation curve
    elif curve_used == -1:
        Rturn_used = data['neg_r_turn'][i]
        Vmax_used = data['neg_v_max'][i]

    # Is this a "good" fit?
    if Rturn_used > 0 and Rturn_used < 200 and Vmax_used < 20000:
        data['good_fit'][i] = 1
################################################################################



################################################################################
# Save results
#-------------------------------------------------------------------------------
data.write(filename[:-4] + '_goodFit.txt', format='ascii.commented_header', 
           overwrite=True)
################################################################################