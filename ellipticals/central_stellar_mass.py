import numpy as np
import matplotlib.pyplot as plt
from astropy.table import Table
import numpy.ma as ma

from IO_data import *


H_0 = 100      # Hubble's Constant in units of h km/s/Mpc
c = 299792.458 # Speed of light in units of km/s
G = 4.30091E-6 # Gravitation constant in units of (km/s)^2 kpc/Msun
MANGA_SPAXEL_SIZE = 0.5*(1/60)*(1/60)*(np.pi/180)  # spaxel size (0.5") in radians

MANGA_FOLDER = '/Users/nityaravi/Documents/Research/RotationCurves/data/manga/'
IMAGE_DIR = MANGA_FOLDER + 'Ellipticals_Images/'
MAP_FOLDER = MANGA_FOLDER + 'DR17/'
PIPE3D_FOLDER = MANGA_FOLDER +'Pipe3D/'
DRP_FILENAME = MANGA_FOLDER + 'output_files/DR17/CURRENT_MASTER_TABLE/Elliptical_sphdisk_refitspirals_BPT_v4.fits'
OUT_FILENAME = MANGA_FOLDER + 'output_files/DR17/CURRENT_MASTER_TABLE/Elliptical_sphdisk_refitspirals_BPT_v5.fits'

################################################################################
# Paths for Bluehive
################################################################################

# MANGA_FOLDER = '/scratch/kdougla7/data/SDSS/dr17/manga/spectro/'
# IMAGE_DIR = '/scratch/nravi3/ellipticals/'
# MAP_FOLDER = MANGA_FOLDER + 'analysis/v3_1_1/3.1.0/HYB10-MILESHC-MASTARSSP/'
# PIPE3D_FOLDER = MANGA_FOLDER + 'pipe3d/'
# #update
# DRP_FILENAME = '/scratch/nravi3/ellipticals/Elliptical_StelVelDispDAPMeanSigma_Mvir_smoothness_lt_2_dipole_vflag_comoving_mratio_refitspirals.fits'
# OUT_FILENAME = '/scratch/nravi3/ellipticals/Elliptical_StelVelDispDAPMeanSigma_Mvir_smoothness_lt_2_dipole_vflag_comoving_mratio_refitspirals.fits'
# COV_DIR = '/scratch/nravi3/ellipticals/elliptical_stellar_mass_cov/'

################################################################################
################################################################################

RUN_ALL_GALAXIES = True

################################################################################
# Open the DRPall file
#-------------------------------------------------------------------------------
DRP_table = Table.read(DRP_FILENAME, format='fits')

DRP_index = {}

for i in range(len(DRP_table)):
    gal_ID = DRP_table['plateifu'][i]

    DRP_index[gal_ID] = i

if RUN_ALL_GALAXIES:
    FILE_IDS = list(DRP_index.keys())
################################################################################

DRP_table['sum_M_star_0p1_R90'] = 0.
DRP_table['count_M_star_0p1_R90'] = 0.

for gal_ID in FILE_IDS:

    i_DRP = DRP_index[gal_ID]

    if DRP_table['Mvir'][i_DRP] > 0:

        # extract maps

        pipe3d_maps = extract_Pipe3d_data(PIPE3D_FOLDER, gal_ID, ['sMass'])
        if pipe3d_maps is None:
            print('No Pipe3D data for ', gal_ID)
            continue

        maps = extract_data(MAP_FOLDER, gal_ID, ['flux'])
        if maps is None:
            print('No data for ', gal_ID)
            continue

        # extract NSA vals from table

        ba = DRP_table['nsa_elpetro_ba'][i_DRP]
        phi = DRP_table['nsa_elpetro_phi'][i_DRP]
        z = DRP_table['nsa_z'][i_DRP]
        r90_arcsec = DRP_table['nsa_elpetro_th90'][i_DRP]

        # mask maps

        sMass_mask = np.isnan(pipe3d_maps['sMass_density'])
        msMass_density = ma.array(pipe3d_maps['sMass_density'], mask = sMass_mask)
        sMass_err_mask = np.isnan(pipe3d_maps['sMass_density_err'])
        msMass_density_err = ma.array(pipe3d_maps['sMass_density_err'], mask=sMass_err_mask)

        mflux = ma.array(maps['mflux'], mask=sMass_mask)
        optical_center = np.unravel_index(ma.argmax(mflux), mflux.shape)

        # r90 ellipse mask

        dist_to_galaxy_Mpc = c*z/H_0
        dist_to_galaxy_kpc = dist_to_galaxy_Mpc*1000
        pix_scale_factor = dist_to_galaxy_kpc*np.tan(MANGA_SPAXEL_SIZE)

        array_length = msMass_density.shape[0]  # y-coordinate distance
        array_width = msMass_density.shape[1]  # x-coordinate distance

        X_RANGE = np.arange(0, array_width, 1)
        Y_RANGE = np.arange(0, array_length, 1)
        X_COORD, Y_COORD = np.meshgrid( X_RANGE, Y_RANGE)

        phi_elip = (90 - phi)*np.pi/180.

        x_diff = X_COORD - optical_center[1]
        y_diff = Y_COORD - optical_center[0]

        ellipse = (x_diff*np.cos(phi_elip) - y_diff*np.sin(phi_elip))**2 \
                + (x_diff*np.sin(phi_elip) + y_diff*np.cos(phi_elip))**2 \
                / ba**2


        r90_spax = r90_arcsec/0.5
        ellipse_mask = ellipse > (r90_spax/10)**2

        center_msMass = ma.array(msMass_density, mask=np.logical_or(sMass_mask, ellipse_mask))
        cent_mass = np.log10(np.sum(10**center_msMass)) 

        spax_count = len(center_msMass[~center_msMass.mask])

        DRP_table['sum_M_star_0p1_R90'][i_DRP] = cent_mass
        DRP_table['count_M_star_0p1_R90'][i_DRP] = spax_count



DRP_table.write(OUT_FILENAME, format='fits', overwrite=True)