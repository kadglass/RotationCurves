'''
Plot the baryonic Tully-Fisher relation (M* v. V)
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

data_filename = 'DRP-master_file_vflag_BB_smooth1p85_mapFit_N2O2_HIdr2_morph_noWords_v6.txt'

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
################################################################################




################################################################################
# Adjust Vmax values by galaxy thickness
#-------------------------------------------------------------------------------
q0 = 0.2

cosi2 = (sample['ba_map']**2 - q0**2)/(1 - q0**2)

cosi2[cosi2 < 0] = 0

sample['Vmax_map_corrected'] = sample['Vmax_map']*np.sin(np.arccos(sample['ba_map']))/np.sqrt(1 - cosi2)
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
# Best-fits from other papers
#-------------------------------------------------------------------------------
logM = np.arange(8, 13, 1)

AvilaReese08 = -0.65 + 0.27*logM
AquinoOrtiz18 = -1.00 + 0.30*logM
AquinoOrtiz20 = -1.17 + 0.31*logM
################################################################################




################################################################################
# Plot the baryonic Tully-Fisher relation (M* v. V)
#-------------------------------------------------------------------------------
# Formatting
tSize = 14 # text size

fig = plt.figure()

plt.errorbar(np.log10(Rdata['Vmax_map_corrected']), 
             #np.log10(Rdata['Vmax_map']), 
             #np.log10(Rdata['V90_kms']), 
             #Rdata['M90_disk_map'], 
             np.log10(Rdata['NSA_Mstar']), 
             #yerr=Rdata['Vmax_err_map'], 
             fmt='r.', 
             fillstyle='none', 
             label='Red sequence')

plt.errorbar(np.log10(Bdata['Vmax_map_corrected']), 
             #np.log10(Bdata['Vmax_map']), 
             #np.log10(Bdata['V90_kms']), 
             #Bdata['M90_disk_map'], 
             np.log10(Bdata['NSA_Mstar']), 
             #yerr=Bdata['Vmax_err_map'], 
             fmt='b+', 
             label='Blue cloud')

plt.errorbar(np.log10(GVdata['Vmax_map_corrected']), 
             #np.log10(GVdata['Vmax_map']), 
             #np.log10(GVdata['V90_kms']), 
             #GVdata['M90_disk_map'], 
             np.log10(GVdata['NSA_Mstar']), 
             #yerr=GVdata['Vmax_err_map'], 
             fmt='g*', 
             label='Green valley')

# Avila-Reese08
plt.plot(AvilaReese08, logM, '--', c='gray', label='Avila-Reese et al. (2008)')

# Aquino-Ortiz18
plt.plot(AquinoOrtiz18, logM, 'k:', label='Aquino-Ortiz et al. (2018)')

# Aquino-Ortiz20
#plt.plot(AquinoOrtiz20, logM, '-.', c='lightgray', 
#         label='Aquino-Ortiz et al. (2020)')

plt.ylim((8,12))
plt.xlim((1.5,3.75))

#plt.xscale('log')

plt.ylabel(r'log($M_*/M_\odot$)', fontsize=tSize)
plt.xlabel('log($V_{max}$ [km/s])', fontsize=tSize)

plt.legend(fontsize=tSize)

fig.patch.set_facecolor('none')

ax = plt.gca()
ax.tick_params(labelsize=tSize-2)#, length=10., width=3.)
#ax.set_facecolor('white')

plt.tight_layout()

plt.show()
#plt.savefig(data_directory + 'Images/bTFR_CMD_v6.eps', 
#            format='eps', 
            #transparent=True,
#            dpi=120)
################################################################################