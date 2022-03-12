'''
Plot the correlation between our estimate of Mtot and that calculated with S_K.
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

data_filename = 'DRP-master_file_vflag_BB_smooth1p85_mapFit_N2O2_HIdr2_morph_SK_v6.txt'

data = Table.read(data_directory + data_filename, 
                  format='ascii.commented_header')
################################################################################




################################################################################
# Calculate the velocity at R90
#-------------------------------------------------------------------------------
# Convert r from arcsec to kpc
#-------------------------------------------------------------------------------
H_0 = 100      # Hubble's Constant in units of h km/s/Mpc
c = 299792.458 # Speed of light in units of km/s

dist_to_galaxy_Mpc = c*data['NSA_redshift']/H_0
dist_to_galaxy_kpc = dist_to_galaxy_Mpc*1000

data['R90_kpc'] = dist_to_galaxy_kpc*np.tan(data['NSA_elpetro_th90']*(1./60)*(1./60)*(np.pi/180))
#-------------------------------------------------------------------------------

data['V90_kms'] = rot_fit_BB(data['R90_kpc'], 
                             [data['Vmax_map'], 
                              data['Rturn_map'], 
                              data['alpha_map']])
################################################################################




################################################################################
# Calculate the mass ratio
#-------------------------------------------------------------------------------
data['M90_Mdisk_ratio'] = 10**(data['M90_map'] - data['M90_disk_map'])
################################################################################




################################################################################
# Sample criteria
#-------------------------------------------------------------------------------
bad_boolean = np.logical_or.reduce([np.isnan(data['M90_map']), 
                                    np.isnan(data['M90_disk_map']), 
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
# Plot Tully-Fisher relations
#-------------------------------------------------------------------------------
# Formatting
tSize = 14 # text size

plt.figure()

RS = plt.errorbar(np.log10(Rdata['Mdyn90_SK']), 
                  Rdata['M90_map'], 
                  #yerr=Rdata['Vmax_err_map'], 
                  fmt='r.', 
                  fillstyle='none', 
                  zorder=0, 
                  label='Red sequence')

BC = plt.errorbar(np.log10(Bdata['Mdyn90_SK']), 
                  Bdata['M90_map'], 
                  #yerr=Bdata['Vmax_err_map'], 
                  fmt='b+', 
                  zorder=1, 
                  label='Blue cloud')

GV = plt.errorbar(np.log10(GVdata['Mdyn90_SK']), 
                  GVdata['M90_map'], 
                  #yerr=GVdata['Vmax_err_map'], 
                  fmt='g*', 
                  markersize=4, 
                  zorder=2, 
                  label='Green valley')

# 1:1
o2o, = plt.plot([9,14], [9,14], 'k:', zorder=3, label='$y = x$')

plt.xlim((9,14))
plt.ylim((7.75,14.1))

plt.ylabel(r'log($M_{tot}/M_\odot$)', fontsize=tSize)
plt.xlabel(r'log($M_{dyn}/M_\odot$)', fontsize=tSize)

plt.legend(handles=[BC, GV, RS, o2o], fontsize=tSize-4)#, loc='lower right')

ax = plt.gca()
ax.tick_params(labelsize=tSize)#, length=10., width=3.)

#fig.patch.set_facecolor('none')

plt.tight_layout()

#plt.show()
plt.savefig(data_directory + 'Images/Mcomp_SK_CMD_v6.eps', 
            format='eps', 
            #transparent=True,
            dpi=120)
################################################################################


