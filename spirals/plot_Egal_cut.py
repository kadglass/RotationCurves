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
from matplotlib.collections import PatchCollection
from matplotlib.patches import Rectangle, Polygon
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
# Separate galaxies by their CMD classification
#-------------------------------------------------------------------------------
# Green valley
gboolarray = good_galaxies['CMD_class'] == 2

# Blue cloud
bboolarray = good_galaxies['CMD_class'] == 1

# Red sequence
rboolarray = good_galaxies['CMD_class'] == 3

GVdata = good_galaxies[gboolarray]
Bdata = good_galaxies[bboolarray]
Rdata = good_galaxies[rboolarray]
################################################################################




################################################################################
# Plot
#-------------------------------------------------------------------------------
tSize = 14

fig, ax = plt.subplots(1)

remove1 = Rectangle((1.96,0.13), 3.6-1.96, 0.45)
remove2 = Rectangle((2.9, 0.07), 3.6-2.9, 0.13-0.07)
remove3 = Polygon(np.array([[1.96, 0.13], [2.9, 0.07], [2.9, 0.13]]), True)

pc = PatchCollection([remove1, remove2, remove3], 
                     facecolor='mistyrose', 
                     #alpha=0.1, 
                     edgecolor='none')

ax.add_collection(pc)

plt.plot(Rdata['DRP_map_smoothness'], Rdata['map_frac_unmasked'], 'r.', 
         fillstyle='none', label='Red sequence')
plt.plot(Bdata['DRP_map_smoothness'], Bdata['map_frac_unmasked'], 'b+', 
         label='Blue cloud')
plt.plot(GVdata['DRP_map_smoothness'], GVdata['map_frac_unmasked'], 'g*', 
         markersize=4, label='Green valley')

plt.vlines(1.96, 0.13, 0.6, colors='k')
plt.hlines(0.07, 2.9, 3.6, colors='k')
plt.plot([1.96, 2.9], [0.13, 0.07], 'k')

plt.xlim(0.25, 3.55)
plt.ylim(0.04, 0.55)

plt.xlabel('smoothness score', fontsize=tSize)
plt.ylabel('fraction of unmasked data', fontsize=tSize)

ax = plt.gca()
ax.tick_params(labelsize=tSize)

plt.legend(fontsize=tSize-4)

plt.tight_layout()

#plt.show()
plt.savefig('Images/Egal_cut_v6.eps', format='eps', dpi=120)
################################################################################



