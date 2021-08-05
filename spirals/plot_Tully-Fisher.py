'''
Plot the Tully-Fisher relation
'''


################################################################################
# Import modules
#-------------------------------------------------------------------------------
from astropy.table import Table

import numpy as np

from dark_matter_mass_v1 import rot_fit_BB

import matplotlib.pyplot as plt
'''
import matplotlib as mpl
COLOR = 'white'
mpl.rcParams['text.color'] = COLOR
mpl.rcParams['axes.labelcolor'] = COLOR
mpl.rcParams['xtick.color'] = COLOR
mpl.rcParams['ytick.color'] = COLOR
'''
################################################################################




################################################################################
# Read in data
#-------------------------------------------------------------------------------
data_directory = ''

data_filename = 'DRP-master_file_vflag_BB_smooth1p85_mapFit_N2O2_HIdr2_noWords_v5.txt'

data = Table.read(data_directory + data_filename, 
                  format='ascii.commented_header')
################################################################################



'''
################################################################################
# Calculate the velocity at R90
#-------------------------------------------------------------------------------
data['R90'] = data['NSA_elpetro_th90']*

data['V90'] = rot_fit_BB(data['R90'], 
                         data['Vmax_map'], 
                         data['Rturn_map'], 
                         data['alpha_map'])
################################################################################
'''



################################################################################
# Sample criteria
#-------------------------------------------------------------------------------
bad_boolean = np.logical_or.reduce([data['M90_map'] == -99, 
                                    data['M90_disk_map'] == -99, 
                                    data['alpha_map'] > 99, 
                                    data['ba_map'] > 0.998])#, 
#                                    data['V90']/data['Vmax_map'] < 0.9])

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
# Plot Tully-Fisher relation
#-------------------------------------------------------------------------------
# Formatting
tSize = 8 # text size

fig = plt.figure()

plt.errorbar(Rdata['rabsmag'], 
             Rdata['Vmax_map'], 
             #yerr=Rdata['Vmax_err_map'], 
             fmt='r.', 
             fillstyle='none', 
             label='Red sequence')

plt.errorbar(Bdata['rabsmag'], 
             Bdata['Vmax_map'], 
             #yerr=Bdata['Vmax_err_map'], 
             fmt='b+', 
             label='Blue cloud')

plt.errorbar(GVdata['rabsmag'], 
             GVdata['Vmax_map'], 
             #yerr=GVdata['Vmax_err_map'], 
             fmt='g*', 
             label='Green valley')

#plt.plot(sample['rabsmag'], sample['Vmax_map'], '.')

plt.xlim((-17,-23))
plt.ylim((30,7000))

plt.yscale('log')

plt.xlabel('$M_r$', fontsize=tSize)
plt.ylabel('$V_{max}$ [km/s]', fontsize=tSize)

plt.legend(fontsize=tSize)

fig.patch.set_facecolor('none')

ax = plt.gca()
ax.tick_params(labelsize=tSize)#, length=10., width=3.)
#ax.set_facecolor('white')

plt.tight_layout()

plt.show()
#plt.savefig(data_directory + 'Images/Tully-Fisher_CMD_v5.eps', 
#            format='eps', 
#            #transparent=True,
#            dpi=300)
################################################################################


