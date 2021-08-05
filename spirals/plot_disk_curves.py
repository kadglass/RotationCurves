'''
Plot a random subsample of the fitted disk curves extrapolated to R90
'''


################################################################################
# Import modules
#-------------------------------------------------------------------------------
from astropy.table import Table

import numpy as np

import sys
sys.path.insert(1, '/Users/kellydouglass/Documents/Research/Rotation_curves/Yifan_Zhang/RotationCurve/')
from rotation_curve_functions import disk_vel

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
# Sample criteria
#-------------------------------------------------------------------------------
bad_boolean = np.logical_or.reduce([data['M90_map'] == -99, 
                                    data['M90_disk_map'] == -99, 
                                    data['alpha_map'] > 99, 
                                    data['ba_map'] > 0.998])

sample = data[~bad_boolean]

# Choose a random subsample of this
i_rand = np.random.rand(50)*len(sample)

random_sample = sample[i_rand.astype(int)]
################################################################################




################################################################################
# Convert radius to pc
#-------------------------------------------------------------------------------
H_0 = 100      # Hubble's Constant in units of h km/s/Mpc
c = 299792.458 # Speed of light in units of km/s

dist_to_galaxy_Mpc = c*random_sample['NSA_redshift']/H_0
dist_to_galaxy_kpc = dist_to_galaxy_Mpc*1000

random_sample['R90_kpc'] = dist_to_galaxy_kpc*np.tan(random_sample['NSA_elpetro_th90']*(1./60)*(1./60)*(np.pi/180))
################################################################################



'''
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
'''



################################################################################
# Sample disk rotation curves
#-------------------------------------------------------------------------------
radii_max = np.linspace(0,1,50)

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

plt.xlim([-0.05,2.5])

plt.xlabel('$r/R_{max}$')
plt.ylabel('$V_*$ [km/s]')

plt.tight_layout()

#plt.show()
plt.savefig(data_directory + 'Images/DRP-Pipe3D/mass_curves/random_mass_curves.eps', 
            format='eps', 
            dpi=300)
################################################################################




