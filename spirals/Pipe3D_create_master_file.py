'''
Create (or overwrite) master file.
'''


################################################################################
# Import modules
#-------------------------------------------------------------------------------
import os.path

import glob

from astropy.io import fits
from astropy.coordinates import SkyCoord
import astropy.units as u

import numpy as np

from Pipe3D_rotation_curve import extract_data, match_to_NSA, write_master_file
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

MANGA_FOLDER = LOCAL_PATH + '../data/MaNGA/MaNGA_DR15/pipe3d/'
################################################################################




################################################################################
# Create list of .fits file names to extract a rotation curve from.
#-------------------------------------------------------------------------------
files = glob.glob( MANGA_FOLDER + '*/manga-*.Pipe3D.cube.fits.gz')
################################################################################




################################################################################
# Extract the length of the 'files' array
#-------------------------------------------------------------------------------
N_files = len( files)
################################################################################




################################################################################
# Open NASA-Sloan-Atlas (NSA) master catalog and extract the data structures for 
# RA; DEC; the axes ratio of b/a (obtained via sersic fit); phi, the angle of 
# rotation in the two-dimensional, observational plane (obtained via Sersic 
# fit); the redshift distance calculated from the shift in H-alpha flux; and the 
# absolute magnitude in the r-band.
#
# Note: The NSA RA and DEC are passed to a SkyCoord object to match galaxies to 
#       the NSA catalog index.
#-------------------------------------------------------------------------------
nsa_catalog_filename = '/Users/kellydouglass/Documents/Drexel/Research/Data/NSA/nsa_v1_0_1.fits'
nsa_catalog = fits.open( nsa_catalog_filename)

nsa_axis_ratio_all = nsa_catalog[1].data['SERSIC_BA']
nsa_phi_EofN_deg_all = nsa_catalog[1].data['SERSIC_PHI']
nsa_z_all = nsa_catalog[1].data['Z']
#nsa_zdist_all = nsa_catalog[1].data['ZDIST']
#nsa_zdist_all_err = nsa_catalog[1].data['ZDIST_ERR']
nsa_absmag_all = nsa_catalog[1].data['ELPETRO_ABSMAG']
nsa_ra_all = nsa_catalog[1].data['RA']
nsa_dec_all = nsa_catalog[1].data['DEC']
nsa_plate_all = nsa_catalog[1].data['PLATE']
nsa_fiberID_all = nsa_catalog[1].data['FIBERID']
nsa_mjd_all = nsa_catalog[1].data['MJD']
nsaID_all = nsa_catalog[1].data['NSAID']
if nsa_catalog_filename[-6] == '1':
    #nsa_mStar_all = nsa_catalog[1].data['SERSIC_MASS']
    nsa_mStar_all = nsa_catalog[1].data['ELPETRO_MASS']
else:
    nsa_mStar_all = nsa_catalog[1].data['MASS']

nsa_catalog.close()

catalog_coords = SkyCoord( ra=nsa_ra_all*u.degree, dec=nsa_dec_all*u.degree)
################################################################################




################################################################################
# Initialize the master arrays that create the structure of the master file.
#-------------------------------------------------------------------------------
manga_plate_master = -1 * np.ones( N_files)
manga_IFU_master = -1 * np.ones( N_files)

axis_ratio_master = -1. * np.ones( N_files)
phi_master = -1. * np.ones( N_files)
z_master = -1. * np.ones( N_files)
#zdist_master = -1. * np.ones( N_files)
#zdist_err_master = -1. * np.ones( N_files)
mStar_master = -1. * np.ones( N_files)
rabsmag_master = np.zeros( N_files)

ra_master = -1. * np.ones( N_files)
dec_master = -1. * np.ones( N_files)
nsa_plate_master = -1 * np.ones( N_files)
nsa_fiberID_master = -1 * np.ones( N_files)
nsa_mjd_master = -1 * np.ones( N_files)
nsaID_master = -1 * np.ones( N_files)
################################################################################




################################################################################
# 
#-------------------------------------------------------------------------------
for i in range(N_files):

    file_name = files[i]

    ############################################################################
    # Extract [plate]-[IFU] of galaxy from file name
    #---------------------------------------------------------------------------
    gal_ID = file_name[ file_name.find('manga-') + 6 : file_name.find('.Pipe3D')]

    manga_plate, manga_IFU = gal_ID.split('-')
    ############################################################################


    ############################################################################
    # Extract the necessary data from the .fits file.
    #---------------------------------------------------------------------------
    _, _, _, _, _, _, _, gal_ra, gal_dec = extract_data( file_name)
    print( gal_ID, " EXTRACTED")
    ############################################################################


    ############################################################################
    # Add the MaNGA catalog information to the master arrays.
    #---------------------------------------------------------------------------
    manga_plate_master[i] = manga_plate
    manga_IFU_master[i] = manga_IFU
    ############################################################################


    ############################################################################
    # Match the galaxy's RA and DEC from the to the NSA catalog index, and pull
    # out the matched data from the NSA catalog.
    #---------------------------------------------------------------------------
    nsa_gal_idx = match_to_NSA( gal_ra, gal_dec, catalog_coords)
    print(gal_ID, " MATCHED")

    axis_ratio = nsa_axis_ratio_all[ nsa_gal_idx]
    phi_EofN_deg = nsa_phi_EofN_deg_all[ nsa_gal_idx] * u.degree
    z = nsa_z_all[ nsa_gal_idx]
    #zdist = zdist_all[ nsa_gal_idx]
    #zdist_err = zdist_all_err[ nsa_gal_idx]
    mStar = nsa_mStar_all[ nsa_gal_idx] * u.M_sun
    rabsmag = nsa_absmag_all[ nsa_gal_idx][4] # SDSS r-band

    nsa_ra = nsa_ra_all[ nsa_gal_idx]
    nsa_dec = nsa_dec_all[ nsa_gal_idx]
    nsa_plate = nsa_plate_all[ nsa_gal_idx]
    nsa_fiberID = nsa_fiberID_all[ nsa_gal_idx]
    nsa_mjd = nsa_mjd_all[ nsa_gal_idx]
    nsaID = nsaID_all[ nsa_gal_idx]
    ############################################################################


    ############################################################################
    # Add the NSA catalog information to the master arrays.
    #---------------------------------------------------------------------------
    axis_ratio_master[i] = axis_ratio
    phi_master[i] = phi_EofN_deg / u.degree
    z_master[i] = z
    #zdist_master[i] = zdist
    #zdist_err_master[i] = zdist_err
    mStar_master[i] = mStar / u.M_sun
    rabsmag_master[i] = rabsmag

    ra_master[i] = nsa_ra
    dec_master[i] = nsa_dec
    nsa_plate_master[i] = nsa_plate
    nsa_fiberID_master[i] = nsa_fiberID
    nsa_mjd_master[i] = nsa_mjd
    nsaID_master[i] = nsaID
    ############################################################################

    print('\n')
################################################################################





###############################################################################
# Build master file that contains identifying information for each galaxy
# as well as scientific information as taken from the DRPall catalog.
#------------------------------------------------------------------------------
write_master_file( manga_plate_master, manga_IFU_master,
                   nsa_plate_master, nsa_fiberID_master, nsa_mjd_master, nsaID_master, 
                   ra_master, dec_master, z_master,
                   axes_ratio_master, phi_master, 
                   mStar_master, rabsmag_master, 
                   LOCAL_PATH, MASTER_FILENAME='master_file_vflag_10.txt')
print("MASTER FILE WRITTEN")
###############################################################################











