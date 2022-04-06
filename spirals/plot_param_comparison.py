'''
Parameter comparison (photometric v. kinematic)
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
    photo_param = 'NSA_phi'
    kinem_param = 'phi_map'
    kinem_param_err = 'phi_err_map'

    param_min = 0
    param_max = 180

    x_label = 'photometric position angle [$^\circ$]'
    y_label = 'kinematic position angle [$^\circ$]'

elif param == 'ba':
    photo_param = 'NSA_ba'
    kinem_param = 'ba_map'
    kinem_param_err = 'ba_err_map'

    param_min = 0
    param_max = 1

    x_label = 'photometric axis ratio'
    y_label = 'kinematic axis ratio'
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
# Plot parameter (photometric v. kinematic)
#-------------------------------------------------------------------------------
plt.figure()

plt.errorbar(Rdata[photo_param], 
             Rdata[kinem_param], 
             #yerr=Rdata[kinem_param_err], 
             fmt='r.', 
             fillstyle='none', 
             label='Red sequence')

plt.errorbar(Bdata[photo_param], 
             Bdata[kinem_param], 
             #yerr=Bdata[kinem_param_err], 
             fmt='b+', 
             label='Blue cloud')

plt.errorbar(GVdata[photo_param], 
             GVdata[kinem_param], 
             #yerr=GVdata[kinem_param_err], 
             fmt='g*', 
             label='Green valley')

plt.plot([param_min,param_max], [param_min,param_max], 'k', zorder=2.5)
if param == 'phi':
    plt.plot([param_min,param_max], [param_max,2*param_max], 'k', zorder=2.6)

if param == 'ba':
    plt.ylim([-0.1,1.1])

plt.xlabel(x_label)
plt.ylabel(y_label)

plt.legend()

plt.tight_layout()

#plt.show()
plt.savefig(data_directory + 'Images/' + param + '_comp_v5.eps', 
            format='eps', 
            dpi=300)
################################################################################


