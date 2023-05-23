'''
Plot a random subsample of the fitted disk curves extrapolated to R90
'''


################################################################################
# Import modules
#-------------------------------------------------------------------------------
from astropy.table import Table
import astropy.constants as const

import numpy as np

from dark_matter_mass_v1 import rot_fit_BB

import sys
sys.path.insert(1, '/Users/kellydouglass/Documents/Research/Rotation_curves/Yifan_Zhang/RotationCurve/2D_RC/main/')
from rotation_curve_functions import disk_vel

import matplotlib.pyplot as plt
################################################################################




################################################################################
# Constants
#-------------------------------------------------------------------------------
H0 = 100
################################################################################




################################################################################
# Read in data
#-------------------------------------------------------------------------------
data_directory = ''

#data_filename = 'Pipe3D-master_file_vflag_BB_minimize_chi10_smooth2p27_mapFit_N2O2_HIdr2_noWords_v5.txt'
data_filename = 'DRP-master_file_vflag_BB_smooth1p85_mapFit_N2O2_HIdr2_morph_noWords_v6.txt'

data = Table.read(data_directory + data_filename, 
                  format='ascii.commented_header')
################################################################################




################################################################################
# Calculate velocity at R90
#-------------------------------------------------------------------------------
dist_to_galaxy_Mpc = (const.c.to('km/s')*data['NSA_redshift']/H0).value
dist_to_galaxy_kpc = dist_to_galaxy_Mpc*1000

data['R90_kpc'] = dist_to_galaxy_kpc*np.tan(data['NSA_elpetro_th90']*(1./60)*(1./60)*(np.pi/180))

data['V90_kms'] = rot_fit_BB(data['R90_kpc'], 
                             [data['Vmax_map'], data['Rturn_map'], data['alpha_map']])
################################################################################




################################################################################
# Calculate mass ratio
#-------------------------------------------------------------------------------
data['M90_Mdisk_ratio'] = 10**(data['M90_map'] - data['M90_disk_map'])
################################################################################




################################################################################
# Sample criteria
#-------------------------------------------------------------------------------
bad_boolean = np.logical_or.reduce([data['M90_map'] == -99, 
                                    data['M90_disk_map'] == -99, 
                                    data['alpha_map'] > 99, 
                                    data['ba_map'] > 0.998, 
                                    data['V90_kms']/data['Vmax_map'] < 0.9, 
                                    (data['Tidal'] & (data['DL_merge'] > 0.97)), 
                                    data['map_frac_unmasked'] < 0.05, 
                                    (data['map_frac_unmasked'] > 0.13) & (data['DRP_map_smoothness'] > 1.96), 
                                    (data['map_frac_unmasked'] > 0.07) & (data['DRP_map_smoothness'] > 2.9), 
                                    (data['map_frac_unmasked'] > -0.0638*data['DRP_map_smoothness'] + 0.255) & (data['DRP_map_smoothness'] > 1.96), 
                                    data['M90_Mdisk_ratio'] > 1050])

sample = data[~bad_boolean]

# Choose a random subsample of this
i_rand = np.random.rand(50)*len(sample)

random_sample = sample[i_rand.astype(int)]
################################################################################



'''
################################################################################
# Separate galaxies by their CMD classification
#-------------------------------------------------------------------------------
# Green valley
gboolarray = random_sample['CMD_class'] == 2

# Blue cloud
bboolarray = random_sample['CMD_class'] == 1

# Red sequence
rboolarray = random_sample['CMD_class'] == 3

GVdata = random_sample[gboolarray]
Bdata = random_sample[bboolarray]
Rdata = random_sample[rboolarray]
################################################################################
'''



################################################################################
# Sample disk rotation curves
#-------------------------------------------------------------------------------
tSize = 14

radii_max = np.arange(0,1,0.005)

for i in range(len(random_sample)):
    SigmaD = random_sample['Sigma_disk_map'][i]
    rD = random_sample['Rdisk_map'][i]
    Rmax = random_sample['Rmax'][i]
    R90 = random_sample['R90_kpc'][i]

    Vdisk_max = disk_vel(radii_max*Rmax, SigmaD, rD)

    if random_sample['CMD_class'][i] == 1:
        line_color = 'b'
    elif random_sample['CMD_class'][i] == 2:
        line_color = 'lime'
    else:
        line_color = 'r'

    plt.plot(radii_max, Vdisk_max, color=line_color, linewidth=1)

    if R90 > Rmax:
        radii_90 = np.linspace(1,R90/Rmax,50)
        Vdisk_90 = disk_vel(radii_90*Rmax, SigmaD, rD)
        plt.plot(radii_90, Vdisk_90, color=line_color, linestyle='dashed', linewidth=1)

# Create place-holder lines for legend
plt.plot([-1, 0], [-1, 0], 'r', label='Red sequence')
plt.plot([-1, 0], [-1, 0], 'lime', label='Green valley')
plt.plot([-1, 0], [-1, 0], 'b', label='Blue cloud')

plt.xlim([0,2.5])
plt.ylim(ymin=0)

plt.xlabel('$r/R_{max}$', fontsize=tSize)
plt.ylabel('$V_*$ [km/s]', fontsize=tSize)

ax = plt.gca()
ax.tick_params(labelsize=tSize)

plt.legend(fontsize=tSize-4)

plt.tight_layout()

#plt.show()
plt.savefig(data_directory + 'Images/DRP-Pipe3D/mass_curves/random_mass_curves_v6.eps', 
            format='eps', 
            dpi=120)
################################################################################




