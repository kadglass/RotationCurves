'''
Plot the TFR and the bTFR in a single figure window
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
v = np.logspace(1.5, 3.75, 20)

AvilaReese08 = -0.650 + 0.274*logM

Ferrero17_M = 8.63e8*((v/50)**4.1)*np.exp(-(v/50)**0.432)

AquinoOrtiz18 = -0.41 + 0.25*logM
#AquinoOrtiz20 = -1.17 + 0.31*logM
################################################################################




################################################################################
# Plot Tully-Fisher relations
#-------------------------------------------------------------------------------
# Formatting
tSize = 14 # text size

fig = plt.figure()

#-------------------------------------------------------------------------------
# TFR
#-------------------------------------------------------------------------------
plt.subplot(122)

plt.errorbar(np.log10(Rdata['Vmax_map']), 
             Rdata['rabsmag'], 
             #yerr=Rdata['Vmax_err_map'], 
             fmt='r.', 
             fillstyle='none', 
             label='Red sequence')

plt.errorbar(np.log10(Bdata['Vmax_map']), 
             Bdata['rabsmag'], 
             #yerr=Bdata['Vmax_err_map'], 
             fmt='b+', 
             label='Blue cloud')

plt.errorbar(np.log10(GVdata['Vmax_map']), 
             GVdata['rabsmag'], 
             #yerr=GVdata['Vmax_err_map'], 
             fmt='g*', 
             markersize=4, 
             label='Green valley')

#plt.plot(sample['rabsmag'], sample['Vmax_map'], '.')

plt.ylim((-17,-23))
plt.xlim((1.5,3.75))

#plt.xscale('log')

plt.ylabel('$M_r$', fontsize=tSize)
plt.xlabel('log($V_{max}$ [km/s])', fontsize=tSize)

ax = plt.gca()
ax.tick_params(labelsize=tSize)#, length=10., width=3.)
#ax.set_facecolor('white')
#-------------------------------------------------------------------------------


#-------------------------------------------------------------------------------
# Baryonic TFR
#-------------------------------------------------------------------------------
plt.subplot(121)

RS = plt.errorbar(np.log10(Rdata['Vmax_map']), 
                  Rdata['M90_disk_map'], 
                  #yerr=Rdata['Vmax_err_map'], 
                  fmt='r.', 
                  fillstyle='none', 
                  zorder=0, 
                  label='Red sequence')

BC = plt.errorbar(np.log10(Bdata['Vmax_map']), 
                  Bdata['M90_disk_map'], 
                  #yerr=Bdata['Vmax_err_map'], 
                  fmt='b+', 
                  zorder=1, 
                  label='Blue cloud')

GV = plt.errorbar(np.log10(GVdata['Vmax_map']), 
                  GVdata['M90_disk_map'], 
                  #yerr=GVdata['Vmax_err_map'], 
                  fmt='g*', 
                  markersize=4, 
                  zorder=2, 
                  label='Green valley')

# Avila-Reese08
AR08, = plt.plot(AvilaReese08, logM, '--', c='lightgray', zorder=3, 
                 label='Avila-Reese et al. (2008)')

# Ferrero17
F17, = plt.plot(np.log10(v), np.log10(Ferrero17_M), 'k-.', zorder=5,
                label='Ferrero et al. (2017)')

# Aquino-Ortiz18
AO18, = plt.plot(AquinoOrtiz18, logM, ':', c='gray', zorder=4, 
                 label='Aquino-Ortiz et al. (2018)')

# Aquino-Ortiz20
#AO20, = plt.plot(AquinoOrtiz20, logM, '-.', c='lightgray', 
#                 label='Aquino-Ortiz et al. (2020)')

plt.ylim((8,12))
plt.xlim((1.5,3.75))

#plt.xscale('log')

plt.ylabel(r'log($M_{90,disk}/M_\odot$)', fontsize=tSize)
plt.xlabel('log($V_{max}$ [km/s])', fontsize=tSize)

plt.legend(handles=[BC, GV, RS, AR08, F17, AO18], 
           fontsize=tSize-4, 
           ncol=2, 
           loc='upper left', 
           bbox_to_anchor=(0, 1.25, 2.4, 0.102), 
           mode='expand', 
           borderaxespad=0)

ax = plt.gca()
ax.tick_params(labelsize=tSize)#, length=10., width=3.)
#-------------------------------------------------------------------------------

fig.patch.set_facecolor('none')

plt.tight_layout()

#plt.show()
plt.savefig(data_directory + 'Images/Tully-Fisher_both_CMD_v6.eps', 
            format='eps', 
            #transparent=True,
            dpi=120)
################################################################################


