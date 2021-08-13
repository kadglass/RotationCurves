'''
Position angle distribution
'''


################################################################################
# Import modules
#-------------------------------------------------------------------------------
from astropy.table import Table

import numpy as np

import matplotlib.pyplot as plt
################################################################################




################################################################################
# Read in data
#-------------------------------------------------------------------------------
data_directory = ''

data_filename = 'Pipe3D-master_file_vflag_BB_minimize_chi10_smooth2p27_mapFit_N2O2_HIdr2_noWords_v5.txt'

data = Table.read(data_directory + data_filename, 
                  format='ascii.commented_header')
################################################################################




################################################################################
# Parameter to plot
#-------------------------------------------------------------------------------
param = 'ba'

if param == 'phi':
    kinem_param = 'phi_map'

    param_min = 0
    param_max = 360
    param_bin_width = 15

    x_label = 'kinematic position angle [$^\circ$]'

elif param == 'ba':
    kinem_param = 'ba_map'

    param_min = 0
    param_max = 1
    param_bin_width = 0.05

    x_label = 'kinematic axis ratio'
################################################################################




################################################################################
# Sample criteria
#-------------------------------------------------------------------------------
bad_boolean = np.logical_or.reduce([data['M90_map'] == -99, 
                                    data['M90_disk_map'] == -99, 
                                    data['alpha_map'] > 99, 
                                    data['ba_map'] > 0.998])

sample = data[~bad_boolean]
################################################################################




################################################################################
# Separate galaxies by their CMD classification
#-------------------------------------------------------------------------------
# Green valley
gboolarray = sample['CMD_class'] == 2

# Blue cloud
bboolarray = sample['CMD_class'] == 1

# Red sequence
rboolarray = sample['CMD_class'] == 3

GVdata = sample[gboolarray]
Bdata = sample[bboolarray]
Rdata = sample[rboolarray]
################################################################################




################################################################################
# Histogram of position angle
#-------------------------------------------------------------------------------
param_bins = np.arange(param_min, param_max + param_bin_width, param_bin_width)

plt.figure()

plt.hist(Rdata[kinem_param], 
         bins=param_bins, 
         density=True, 
         histtype='step', 
         color='r', 
         linestyle=':', 
         label='Red sequence')

plt.hist(Bdata[kinem_param], 
         bins=param_bins, 
         density=True, 
         histtype='step', 
         color='b', 
         label='Blue cloud')

plt.hist(GVdata[kinem_param], 
         bins=param_bins, 
         density=True, 
         histtype='step', 
         color='g', 
         linestyle='--', 
         label='Green valley')

plt.xlabel(x_label)
plt.ylabel('fraction of galaxies')

plt.legend()

plt.tight_layout()

#plt.show()
plt.savefig(data_directory + 'Images/' + param + '_hist_v5.eps', 
            format='eps', 
            dpi=300)
################################################################################


