'''
Plot the BPT diagram
'''


################################################################################
# Import modules
#-------------------------------------------------------------------------------
from astropy.table import Table

import numpy as np

from extract_KIAS2_functions import match_by_index

import matplotlib.pyplot as plt
################################################################################




################################################################################
# Read in data
#-------------------------------------------------------------------------------
data_directory = ''

data_filename = 'Pipe3D-master_file_vflag_BB_minimize_chi10_smooth2p27_mapFit_N2O2_HIdr2_noWords_v5.txt'

data = Table.read(data_directory + data_filename, 
                  format='ascii.commented_header')


flux_directory = '/Users/kellydouglass/Documents/Drexel/Research/Data/'

flux_filename = 'kias1033_5_Martini_MPAJHU_flux_oii.txt'

flux = Table.read(flux_directory + flux_filename, 
                  format='ascii.commented_header')
################################################################################




################################################################################
# Sample criteria
#-------------------------------------------------------------------------------
bad_boolean = np.logical_or.reduce([data['M90_map'] == -99, 
                                    data['M90_disk_map'] == -99, 
                                    data['alpha_map'] > 99, 
                                    data['ba_map'] > 0.998]) 
                                    #data['Z12logOH'] > 0])

sample = data[~bad_boolean]
################################################################################




################################################################################
# Extract flux values and calculate flux ratios
#-------------------------------------------------------------------------------
sample = match_by_index(sample, 
                        flux, 
                        ['NII_6584_FLUX', 
                         'H_ALPHA_FLUX', 
                         'OIII_5007_FLUX', 
                         'H_BETA_FLUX'])

sample['N2_Halpha'] = sample['NII_6584_FLUX']/sample['H_ALPHA_FLUX']
sample['O3_Hbeta'] = sample['OIII_5007_FLUX']/sample['H_BETA_FLUX']
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
# Separate galaxies by their categories (as defined by their luminosity and 
# metallicity)
#-------------------------------------------------------------------------------
# Category 1
cat1_boolarray = sample['Z12logOH'] < 0.4*sample['rabsmag'] + 17.1

# Category 3
cat3_boolarray = sample['Z12logOH'] > 0.4*sample['rabsmag'] + 17.9

cat1_data = sample[cat1_boolarray]
cat2_data = sample[~np.logical_or(cat1_boolarray, cat3_boolarray)]
cat3_data = sample[cat3_boolarray]
################################################################################




################################################################################
# Reference lines
#-------------------------------------------------------------------------------
N2Ha = np.linspace(-1.4,0.3,100)

# Kauffmann et al. (2003)
Ka03 = 1.3 + 0.61/(N2Ha[N2Ha < 0] - 0.05)

# Kewley et al. (2001)
Ke01 = 1.19 + 0.61/(N2Ha - 0.47)
################################################################################




################################################################################
# Plot BPT diagram, colored by CMD
#-------------------------------------------------------------------------------
plt.figure()

plt.plot(np.log10(Rdata['N2_Halpha']), 
         np.log10(Rdata['O3_Hbeta']), 
         'r.', 
         fillstyle='none', 
         label='Red sequence')

plt.plot(np.log10(Bdata['N2_Halpha']), 
         np.log10(Bdata['O3_Hbeta']), 
         'b+', 
         label='Blue cloud')

plt.plot(np.log10(GVdata['N2_Halpha']), 
         np.log10(GVdata['O3_Hbeta']), 
         'g*', 
         label='Green valley')

plt.plot(N2Ha, Ke01, 'k--', label='Kewley et al. (2001)')
plt.plot(N2Ha[N2Ha < 0], Ka03, 'k:', label='Kauffmann et al. (2003)')

plt.ylim([-1.5, 1.5])
plt.xlim([-1.4, 0.5])

plt.xlabel(r'log([NII]/H$\alpha$)')
plt.ylabel(r'log([OIII]/H$\beta$)')

plt.legend()

plt.tight_layout()

plt.show()
#plt.savefig(data_directory + 'Images/BPT_v5.eps', format='eps', dpi=300)
################################################################################




################################################################################
# Plot BPT diagram, colored by category
#-------------------------------------------------------------------------------
plt.figure()

plt.plot(np.log10(cat1_data['N2_Halpha']), 
         np.log10(cat1_data['O3_Hbeta']), 
         'd',
         markersize=2,
         color='darkviolet', 
         fillstyle='none', 
         label='Category 1')

plt.plot(np.log10(cat2_data['N2_Halpha']), 
         np.log10(cat2_data['O3_Hbeta']), 
         's', 
         markersize=2,
         color='teal',
         label='Category 2')

plt.plot(np.log10(cat3_data['N2_Halpha']), 
         np.log10(cat3_data['O3_Hbeta']), 
         '1', 
         #markersize=2,
         color='orange',
         label='Category 3')

plt.plot(N2Ha, Ke01, 'k--', label='Kewley et al. (2001)')
plt.plot(N2Ha[N2Ha < 0], Ka03, 'k:', label='Kauffmann et al. (2003)')

plt.ylim([-1.5, 1.5])
plt.xlim([-1.4, 0.5])

plt.xlabel(r'log([NII]/H$\alpha$)')
plt.ylabel(r'log([OIII]/H$\beta$)')

plt.legend()

plt.tight_layout()

plt.show()
#plt.savefig(data_directory + 'Images/BPT_v5.eps', format='eps', dpi=300)
################################################################################







