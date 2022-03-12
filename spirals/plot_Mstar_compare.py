'''
Compare our estimates of M90_disk with Pipe3D's stellar mass estimates.
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
# Our fits
data_filename = 'DRP-master_file_vflag_BB_smooth1p85_mapFit_N2O2_HIdr2_morph_v6.txt'

data = Table.read(data_filename, format='ascii.commented_header')

# Pipe3D
P3D_folder = '/Users/kellydouglass/Documents/Research/data/SDSS/dr15/manga/spectro/pipe3d/'
P3D_filename = P3D_folder + 'v2_4_3/2.4.3/manga.Pipe3D-v2_4_3.fits'

P3D = Table.read(P3D_filename, format='fits')
################################################################################




################################################################################
# Join the two tables together
#-------------------------------------------------------------------------------
data['P3D_logMstar'] = np.nan

for i in range(len(data)):

    gal_name = 'manga-' + str(data['MaNGA_plate'][i]) + '-' + str(data['MaNGA_IFU'][i])

    i_P3D = np.argwhere(P3D['mangaid'] == gal_name)

    if len(i_P3D) > 0:
        data['P3D_logMstar'][i] = P3D['log_mass'][i_P3D[0]]
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
                                    (data['map_frac_unmasked'] > 0.13) & (data['DRP_map_smoothness'] > 1.96), 
                                    (data['map_frac_unmasked'] > 0.07) & (data['DRP_map_smoothness'] > 2.9), 
                                    (data['map_frac_unmasked'] > -0.0638*data['DRP_map_smoothness'] + 0.255) & (data['DRP_map_smoothness'] > 1.96), 
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
# Compare M90_disk and M* from Pipe3D
#-------------------------------------------------------------------------------
tSize=14

plt.figure(tight_layout=True)

plt.plot(Rdata['P3D_logMstar'], Rdata['M90_disk_map'], 'r.', fillstyle='none', 
         label='Red sequence')
plt.plot(Bdata['P3D_logMstar'], Bdata['M90_disk_map'], 'b+', 
         label='Blue cloud')
plt.plot(GVdata['P3D_logMstar'], GVdata['M90_disk_map'], 'g*', markersize=4, 
         label='Green valley')
plt.plot([5,12], [5,12], 'k:', label='$y = x$')

plt.xlim(7,12)
plt.ylim(7,12)

plt.xlabel('log($M_*/M_\odot$) [Pipe3D]', fontsize=tSize)
plt.ylabel('log($M_d(R_{90})/M_\odot$) [this work]', fontsize=tSize)

ax = plt.gca()
ax.tick_params(labelsize=tSize)

plt.legend(fontsize=tSize-4)

#plt.show()
plt.savefig('Images/M90disk_MstarP3D_compare_v6.eps', format='eps', dpi=120)
################################################################################




