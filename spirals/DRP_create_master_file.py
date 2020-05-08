'''
Create (or overwrite) master file.
'''


################################################################################
# Import modules
#-------------------------------------------------------------------------------
import os.path

import glob

from astropy.io import fits

import numpy as np

from DRP_rotation_curve import match_to_DRPall, write_master_file
################################################################################




################################################################################
# 'LOCAL_PATH' should be updated depending on the file structure.  It is set to 
# 'os.path.dirname(__file__)' when working on a local system.
#
# ATTN: 'MANGA_FOLDER' must be manually altered according to the data release
#       being ran.
#-------------------------------------------------------------------------------
LOCAL_PATH = os.path.dirname(__file__)
if LOCAL_PATH == '':
    LOCAL_PATH = './'

MANGA_FOLDER = LOCAL_PATH + '../data/MaNGA/MaNGA_DR16/HYB10-GAU-MILESHC/'
################################################################################




################################################################################
# Create list of .fits file names to extract a rotation curve from.
#-------------------------------------------------------------------------------
files = glob.glob( MANGA_FOLDER + '*/manga-*-MAPS-HYB10-GAU-MILESHC.fits.gz')
################################################################################




################################################################################
# Extract the length of the 'files' array
#-------------------------------------------------------------------------------
N_files = len( files)
################################################################################




################################################################################
# Open DRPall master catalog and extract the data structures for RA; DEC; the 
# axes ratio of b/a (obtained via elliptical sersic fit); phi, the angle of 
# rotation in the two-dimensional, observational plane (obtained via elliptical 
# sersic fit); the redshift; and the absolute magnitude in the r-band.
#-------------------------------------------------------------------------------
DRPall_filename = LOCAL_PATH + '../data/MaNGA/drpall-v2_4_3.fits'
general_data = fits.open( DRPall_filename)

plateIFU_all = general_data[1].data['plateifu']

ra_all = general_data[1].data['objra']
dec_all = general_data[1].data['objdec']
z_all = general_data[1].data['nsa_z']

axis_ratio_all = general_data[1].data['nsa_elpetro_ba']
phi_all = general_data[1].data['nsa_elpetro_phi']

rabsmag_all = general_data[1].data['nsa_elpetro_absmag'][:,4] # SDSS r-band
mStar_all = general_data[1].data['nsa_elpetro_mass']

manga_target_galaxy_all = general_data[1].data['mngtarg1'] != 0
data_quality_all = general_data[1].data['drp3qual'] < 10000

general_data.close()
################################################################################




################################################################################
# Initialize the master arrays that create the structure of the master file.
#-------------------------------------------------------------------------------
manga_plate_master = -1 * np.ones( N_files, dtype=int)
manga_IFU_master = -1 * np.ones( N_files, dtype=int)

ra_master = -1. * np.ones( N_files)
dec_master = -1. * np.ones( N_files)
z_master = -1. * np.ones( N_files)

axis_ratio_master = -1. * np.ones( N_files)
phi_master = -1. * np.ones( N_files)

rabsmag_master = np.zeros( N_files, dtype=float)
mStar_master = -1. * np.ones( N_files)

manga_target_galaxy_master = np.zeros( N_files, dtype=bool)
manga_dq_master = np.zeros( N_files, dtype=bool)
################################################################################




################################################################################
# 
#-------------------------------------------------------------------------------
for i in range(N_files):

    file_name = files[i]

    ############################################################################
    # Extract [plate]-[IFU] of galaxy from file name
    #---------------------------------------------------------------------------
    gal_ID = file_name[ file_name.find('manga-') + 6 : file_name.find('-MAPS')]

    manga_plate, manga_IFU = gal_ID.split('-')
    ############################################################################


    ############################################################################
    # Add the MaNGA catalog information to the master arrays.
    #---------------------------------------------------------------------------
    manga_plate_master[i] = manga_plate
    manga_IFU_master[i] = manga_IFU
    ############################################################################


    ############################################################################
    # Find the galaxy in the DRPall file, and extract necessary information
    #---------------------------------------------------------------------------
    DRPall_gal_idx = match_to_DRPall( gal_ID, plateIFU_all)

    ra_master[i] = ra_all[ DRPall_gal_idx]
    dec_master[i] = dec_all[ DRPall_gal_idx]
    z_master[i] = z_all[ DRPall_gal_idx]

    axis_ratio_master[i] = axis_ratio_all[ DRPall_gal_idx]
    phi_master[i] = phi_all[ DRPall_gal_idx]

    rabsmag_master[i] = rabsmag_all[ DRPall_gal_idx]
    mStar_master[i] = mStar_all[ DRPall_gal_idx]

    manga_target_galaxy_master[i] = manga_target_galaxy_all[ DRPall_gal_idx]
    manga_dq_master[i] = data_quality_all[ DRPall_gal_idx]
    ############################################################################
################################################################################




###############################################################################
# Remove all objects that either are not a galaxy target or do not have good 
# data.
#------------------------------------------------------------------------------
keep_obj = np.logical_and(manga_target_galaxy_master, manga_dq_master)

manga_plate_good = manga_plate_master[keep_obj]
manga_IFU_good = manga_IFU_master[keep_obj]

ra_good = ra_master[keep_obj]
dec_good = dec_master[keep_obj]
z_good = z_master[keep_obj]

axis_ratio_good = axis_ratio_master[keep_obj]
phi_good = phi_master[keep_obj]

rabsmag_good = rabsmag_master[keep_obj]
mStar_good = mStar_master[keep_obj]
###############################################################################





###############################################################################
# Build master file that contains identifying information for each galaxy as 
# well as scientific information as taken from the DRPall catalog.
#-------------------------------------------------------------------------------
write_master_file( manga_plate_good, manga_IFU_good,
                   ra_good, dec_good, z_good,
                   axis_ratio_good, phi_good, 
                   mStar_good, rabsmag_good, 
                   LOCAL_PATH, MASTER_FILENAME='DRPall-master_file.txt')
print("MASTER FILE WRITTEN")
###############################################################################











