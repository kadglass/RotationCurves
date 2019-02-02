#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 09 2018
@author: Jacob A. Smith

Main script file for 'rotation_curve_vX_X.'
"""

import datetime
START = datetime.datetime.now()

import glob, os.path, warnings
import numpy as np
from astropy.io import fits
from astropy.coordinates import SkyCoord
import astropy.units as u

warnings.simplefilter('ignore', np.RankWarning)

from rotation_curve_v2_1 import extract_data, \
                                match_to_NSA, \
                                calc_rot_curve, \
                                write_rot_curve, \
                                write_master_file

###############################################################################
# 'LOCAL_PATH' should be updated depending on the file structure (e.g. if
#    working in bluehive). It is set to 'os.path.dirname(__file__)' when
#    working on a local system.
#
# In addition, 'LOCAL_PATH' is aliased as 'SCRATCH_PATH' if
#    'WORKING_IN_BLUEHIVE' is set to True. This is done because of how the data
#    folders are kept separate from the python script files in bluehive.
#
# This block can be altered if desired, but the conditional below is tailored
#    for use with bluehive.
#------------------------------------------------------------------------------
WORKING_IN_BLUEHIVE = True

if WORKING_IN_BLUEHIVE:
    LOCAL_PATH = '/home/jsm171'
    SCRATCH_PATH = '/scratch/jsm171'

    IMAGE_DIR = SCRATCH_PATH + '/images'
    MANGA_FOLDER = SCRATCH_PATH + '/manga_files'
    ROT_CURVE_MASTER_FOLDER = SCRATCH_PATH + '/rot_curve_data_files'

else:
    LOCAL_PATH = os.path.dirname(__file__)

    IMAGE_DIR = LOCAL_PATH + '/images'
    MANGA_FOLDER = LOCAL_PATH + '/manga_files'
    ROT_CURVE_MASTER_FOLDER = LOCAL_PATH + '/rot_curve_data_files'

ROT_CURVE_DATA_INDICATOR = '_rot_curve_data'
GAL_STAT_DATA_INDICATOR = '_gal_stat_data'
###############################################################################


###############################################################################
# Create list of .fits file names to extract a rotation curve from.
#
# IMPORTANT: rot_curve_main.py must be run outside the folder that
#            houses the plate folders. The default folder is
#            '/manga_files'.
#------------------------------------------------------------------------------
files = glob.glob( MANGA_FOLDER + '/*manga-*Pipe3D.cube.fits.gz')
###############################################################################


###############################################################################
# Code to isolate files and run it through all of the functions from
# rotation_curve_vX_X.
# ~   ~   ~   ~   ~   ~   ~   ~   ~   ~   ~   ~   ~   ~   ~   ~   ~   ~   ~   ~
#DATA_RELEASES = ['dr14']
#FILE_IDS = ['7957-12701']
#
#files = []
#for data_release in DATA_RELEASES:
#    for file_name in FILE_IDS:
#        files.append( MANGA_FOLDER \
#        + '/' + data_release + '-manga-' + file_name + '.Pipe3D.cube.fits.gz')
###############################################################################


###############################################################################
# Print the list of file names.
#------------------------------------------------------------------------------
#print("files:", files)
###############################################################################


###############################################################################
# Open NASA-Sloan-Atlas (NSA) master catalog and extract the data structurers
#    for RA; DEC; the axes ratio of b/a (obtained via sersic fit); phi, the
#    angle of rotation in the two-dimensional, observational plane (obtained
#    via sersic fit); and the redshift distance calculated from the shift in
#    H-alpha flux.
#
# Note: The NSA RA and DEC are passed to a SkyCoord object to better match
#       galaxies to the NSA catalog index.
#------------------------------------------------------------------------------
if WORKING_IN_BLUEHIVE:
    nsa_catalog = fits.open( SCRATCH_PATH + '/nsa_v0_1_2.fits')
else:
    nsa_catalog = fits.open( LOCAL_PATH + '/nsa_v0_1_2.fits')

nsa_axes_ratio_all = nsa_catalog[1].data['SERSIC_BA']
nsa_phi_EofN_deg_all = nsa_catalog[1].data['SERSIC_PHI']
nsa_zdist_all = nsa_catalog[1].data['ZDIST']
nsa_zdist_all_err = nsa_catalog[1].data['ZDIST_ERR']
nsa_mStar_all = nsa_catalog[1].data['MASS']

nsa_ra_all = nsa_catalog[1].data['RA']
nsa_dec_all = nsa_catalog[1].data['DEC']
nsa_plate_all = nsa_catalog[1].data['PLATE']
nsa_fiberID_all = nsa_catalog[1].data['FIBERID']
nsa_mjd_all = nsa_catalog[1].data['MJD']
nsaID_all = nsa_catalog[1].data['NSAID']

nsa_catalog.close()

catalog_coords = SkyCoord( ra = nsa_ra_all*u.degree,
                             dec = nsa_dec_all*u.degree)
###############################################################################


###############################################################################
# # Initialize the master arrays that create the structure of the master file.
#------------------------------------------------------------------------------
manga_data_release_master = []
manga_plate_master = []
manga_fiberID_master = []

nsa_axes_ratio_master = []
nsa_phi_master = []
nsa_zdist_master = []
nsa_zdist_err_master = []
nsa_mStar_master = []

nsa_ra_master = []
nsa_dec_master = []
nsa_plate_master = []
nsa_fiberID_master = []
nsa_mjd_master = []
nsaID_master = []
###############################################################################


###############################################################################
# This for loop runs through the necessary calculations to calculte and write
#    the rotation curve for all of the galaxies in the MaNGA survey.
# ~   ~   ~   ~   ~   ~   ~   ~   ~   ~   ~   ~   ~   ~   ~   ~   ~   ~   ~   ~
for file_name in files:
    ###########################################################################
    # file_id is a simplified string that identifies each file that is run
    #    through the algorithm. The file_id name scheme is [PLATE]-[FIBER ID].
    #--------------------------------------------------------------------------
    gal_ID = file_name[ file_name.find(MANGA_FOLDER) \
                               + len( MANGA_FOLDER) + 1: file_name.find('-')] \
                       + file_name[file_name.find(MANGA_FOLDER) \
                               + len(MANGA_FOLDER) + 11: \
                               file_name.find('.Pipe3D.cube.fits.gz')]
#    print( gal_ID)
    ###########################################################################


    ###########################################################################
    # Extract the necessary data from the .fits file.
    #--------------------------------------------------------------------------
    Ha_vel, Ha_vel_error, v_band, v_band_err, sMass_density, \
    manga_plate, manga_fiberID, gal_ra, gal_dec = extract_data( file_name)
    print( gal_ID, " EXTRACTED")
    ###########################################################################


    ###########################################################################
    # Add the MaNGA catalog information to the master arrays.
    #--------------------------------------------------------------------------
    manga_data_release_master.append( gal_ID[ 0: 4])
    manga_plate_master.append( manga_plate)
    manga_fiberID_master.append( manga_fiberID)
    ###########################################################################


    ###########################################################################
    # Match the galaxy's RA and DEC from the to the NSA catalog index, and pull
    # out the matched data from the NSA catalog.
    #--------------------------------------------------------------------------
    nsa_gal_idx = match_to_NSA( gal_ra, gal_dec, catalog_coords)
    print(gal_ID, " MATCHED")

    axes_ratio = nsa_axes_ratio_all[ nsa_gal_idx]
    phi_EofN_deg = nsa_phi_EofN_deg_all[ nsa_gal_idx] * u.degree
    zdist = nsa_zdist_all[ nsa_gal_idx]
    zdist_err = nsa_zdist_all_err[ nsa_gal_idx]
    mStar = nsa_mStar_all[ nsa_gal_idx] * u.M_sun

    nsa_ra = nsa_ra_all[ nsa_gal_idx]
    nsa_dec = nsa_dec_all[ nsa_gal_idx]
    nsa_plate = nsa_plate_all[ nsa_gal_idx]
    nsa_fiberID = nsa_fiberID_all[ nsa_gal_idx]
    nsa_mjd = nsa_mjd_all[ nsa_gal_idx]
    nsaID = nsaID_all[ nsa_gal_idx]
    ###########################################################################


    ###########################################################################
    # Add the NSA catalog information to the master arrays.
    #--------------------------------------------------------------------------
    nsa_axes_ratio_master.append( axes_ratio)
    nsa_phi_master.append( phi_EofN_deg / u.degree)
    nsa_zdist_master.append( zdist)
    nsa_zdist_err_master.append( zdist_err)
    nsa_mStar_master.append( mStar / u.M_sun)

    nsa_ra_master.append( nsa_ra)
    nsa_dec_master.append( nsa_dec)
    nsa_plate_master.append( nsa_plate)
    nsa_fiberID_master.append( nsa_fiberID)
    nsa_mjd_master.append( nsa_mjd)
    nsaID_master.append( nsaID)
    ###########################################################################


    ###########################################################################
    # Extract rotation curve data for the .fits file in question and create an
    #    astropy Table containing said data.
    #--------------------------------------------------------------------------
    rot_data_table, gal_stat_table = calc_rot_curve( Ha_vel, Ha_vel_error, \
                                       v_band, v_band_err, sMass_density, \
                                       axes_ratio, phi_EofN_deg, zdist, \
                                       zdist_err, gal_ID, IMAGE_DIR)
    print(gal_ID, " ROT CURVE CALCULATED")
    ###########################################################################


    ###########################################################################
    # Write the rotation curve data to a text file in ascii format.
    #
    # IMPORTANT: rot_curve_main.py writes the data files into the default
    #            folder 'rot_curve_data_files'. It also saves the file with the
    #            default extension '_rot_curve_data'.
    #--------------------------------------------------------------------------
    write_rot_curve( rot_data_table, gal_stat_table,
                    gal_ID,
                    ROT_CURVE_MASTER_FOLDER,
                    ROT_CURVE_DATA_INDICATOR, GAL_STAT_DATA_INDICATOR)
    print(gal_ID, " WRITTEN")
    ###########################################################################

    print("\n")
# ~ # ~ # ~ # ~ # ~ # ~ # ~ # ~ # ~ # ~ # ~ # ~ # ~ # ~ # ~ # ~ # ~ # ~ # ~ # ~


###############################################################################
# Build master file that contains identifying information for each galaxy
#   as well as scientific information as taken from the NSA catalog.
#------------------------------------------------------------------------------
write_master_file( manga_plate_master, manga_fiberID_master,
                  manga_data_release_master,
                  nsa_plate_master, nsa_fiberID_master, nsa_mjd_master,
                  nsaID_master, nsa_ra_master, nsa_dec_master,
                  nsa_axes_ratio_master, nsa_phi_master, nsa_zdist_master,
                  nsa_mStar_master,
                  LOCAL_PATH)
print("MASTER FILE WRITTEN")
###############################################################################



###############################################################################
# Clock the program's run time to check performance.
#------------------------------------------------------------------------------
FINISH = datetime.datetime.now()
print("Runtime:", FINISH - START)
###############################################################################