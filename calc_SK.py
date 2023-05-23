'''
Calculate the S_K value (as defined in Aquino-Ortiz et al., 2018) for all 
specified galaxies.

This module assumes that the velocity maps have already been fitted.
'''



################################################################################
# Import modules
#-------------------------------------------------------------------------------
from astropy.table import Table
import astropy.constants as const

import numpy as np
import numpy.ma as ma

import sys
sys.path.insert(1, '/Users/kellydouglass/Documents/Research/Rotation_curves/RotationCurves/spirals/')
from DRP_rotation_curve import extract_data
from DRP_vel_map_functions import build_map_mask
################################################################################




################################################################################
# Constants
#-------------------------------------------------------------------------------
H0 = 100      # Hubble's Constant in units of h km/s/Mpc
c = 299792.458 # Speed of light in units of km/s
G = 4.30091E-6 # Gravitation constant in units of (km/s)^2 pc/Msun
################################################################################




################################################################################
# Directories
#-------------------------------------------------------------------------------
MANGA_FOLDER = '/Users/kellydouglass/Documents/Research/data/SDSS/dr16/manga/spectro/'

VEL_MAP_FOLDER = MANGA_FOLDER + 'analysis/v2_4_3/2.2.1/HYB10-GAU-MILESHC/'
################################################################################




################################################################################
# Import data
#-------------------------------------------------------------------------------
data_filename = 'spirals/DRP-master_file_vflag_BB_smooth1p85_mapFit_N2O2_HIdr2_morph_v6.txt'

data = Table.read(data_filename, format='ascii.commented_header')
################################################################################




################################################################################
# For each galaxy, we need to calculate its average velocity dispersion.
#-------------------------------------------------------------------------------
data['sigma'] = np.nan

for i in range(len(data)):

    gal_ID = str(data['MaNGA_plate'][i]) + '-' + str(data['MaNGA_IFU'][i])

    ############################################################################
    # Read in the velocity dispersion map
    #---------------------------------------------------------------------------
    maps = extract_data(VEL_MAP_FOLDER, gal_ID, ['Ha_sigma', 'Ha_vel', 'Ha_flux'])
    ############################################################################


    if maps is not None:
        ########################################################################
        # Build mask used for velocity map fit
        #-----------------------------------------------------------------------
        map_mask = build_map_mask(gal_ID, 
                                  data['map_fit_flag'][i], 
                                  ma.array(maps['Ha_vel'], mask=maps['Ha_vel_mask']), 
                                  ma.array(maps['Ha_flux'], mask=maps['Ha_flux_mask'] + maps['Ha_vel_mask']), 
                                  ma.array(maps['Ha_flux_ivar'], mask=maps['Ha_flux_mask'] + maps['Ha_vel_mask']), 
                                  ma.array(maps['Ha_sigma'], mask=maps['Ha_sigma_mask'] + maps['Ha_vel_mask']))
        ########################################################################


        ########################################################################
        # Calculate (unweighted) average velocity dispersion
        #-----------------------------------------------------------------------
        data['sigma'][i] = ma.mean(ma.array(maps['Ha_sigma'], mask=map_mask))
        ########################################################################
################################################################################




################################################################################
# Calculate SK value
#-------------------------------------------------------------------------------
K = 0.5

data['SK'] = K*data['Vmax_map']**2 + data['sigma']**2
################################################################################




################################################################################
# Calculate mass based on SK
#-------------------------------------------------------------------------------
#eta = 2.5 # Cappellari06
eta = 1.8 # AquinoOrtiz18

dist_to_galaxy_Mpc = c*data['NSA_redshift']/H0
dist_to_galaxy_kpc = dist_to_galaxy_Mpc*1000

data['R50_kpc'] = dist_to_galaxy_kpc*np.tan(data['NSA_elpetro_th50']*(1./60)*(1./60)*(np.pi/180))
data['R90_kpc'] = dist_to_galaxy_kpc*np.tan(data['NSA_elpetro_th90']*(1./60)*(1./60)*(np.pi/180))

data['Mdyn50_SK'] = eta*data['R50_kpc']*data['SK']/G
data['Mdyn90_SK'] = eta*data['R90_kpc']*data['SK']/G
################################################################################




################################################################################
# Save data
#-------------------------------------------------------------------------------
data.write(data_filename[:-4] + '_SK.txt', 
           format='ascii.commented_header', 
           overwrite=True)
################################################################################




