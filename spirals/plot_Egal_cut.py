'''
Plot the map smoothness v. the fraction of masked spaxels to show the boundary 
within which most straggling elliptical galaxies are located.
'''


################################################################################
# Import modules
#-------------------------------------------------------------------------------
from astropy.table import Table
import astropy.constants as const

import numpy as np

from dark_matter_mass_v1 import rot_fit_BB

import matplotlib.pyplot as plt
################################################################################




################################################################################
# Constants
#-------------------------------------------------------------------------------
H0 = 100
################################################################################




################################################################################
# Import data
#-------------------------------------------------------------------------------
data_filename = 'DRP-master_file_vflag_BB_smooth1p85_mapFit_N2O2_HIdr2_morph_v6.txt'

data = Table.read(data_filename, format='ascii.commented_header')
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
# Select only the galaxies with good fits
#-------------------------------------------------------------------------------
bad_boolean = np.logical_or.reduce([np.isnan(data['M90_map']), 
                                    np.isnan(data['M90_disk_map']), 
                                    data['alpha_map'] > 99, 
                                    data['ba_map'] > 0.998, 
                                    data['V90_kms']/data['Vmax_map'] < 0.9, 
                                    (data['Tidal'] & (data['DL_merge'] > 0.97)), 
                                    data['map_frac_unmasked'] < 0.05, 
                                    data['M90_Mdisk_ratio'] > 1050])

good_galaxies = data[~bad_boolean]
################################################################################




################################################################################
# Plot
#-------------------------------------------------------------------------------
tSize = 14

plt.figure(tight_layout=True)

plt.plot(good_galaxies['DRP_map_smoothness'], good_galaxies['map_frac_unmasked'], '.')

plt.vlines(1.96, 0.13, 0.6, colors='k')
plt.hlines(0.07, 2.9, 3.6, colors='k')
plt.plot([1.96, 2.9], [0.13, 0.07], 'k')

plt.xlim(0.25, 3.55)
plt.ylim(0.04, 0.55)

plt.xlabel('smoothness score', fontsize=tSize)
plt.ylabel('fraction of unmasked data', fontsize=tSize)

#plt.show()
plt.savefig('Images/Egal_cut.eps', format='eps', dpi=120)
################################################################################



