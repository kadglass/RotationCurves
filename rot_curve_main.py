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
'''
import matplotlib.pyplot as plt
import pickle, psutil
process = psutil.Process(os.getpid())
memory_list = []
'''
warnings.simplefilter('ignore', np.RankWarning)

###############################################################################
# File format for saved images
#------------------------------------------------------------------------------
image_format = 'eps'
###############################################################################


###############################################################################
# 'LOCAL_PATH' should be updated depending on the file structure (e.g. if
#    working in bluehive). It is set to 'os.path.dirname(__file__)' when
#    working on a local system.
#
# In addition, 'LOCAL_PATH' is altered and 'SCRATCH_PATH' is added if
#    'WORKING_IN_BLUEHIVE' is set to True. This is done because of how the data
#    folders are kept separate from the python script files in bluehive. For
#    BlueHive to run, images cannot be generated with $DISPLAY keys; therefore,
#    'matplotlib' is imported and 'Agg' is used. This must be done before
#    'matplotlib.pyplot' is imported.
#
# This block can be altered if desired, but the conditional below is tailored
#    for use with bluehive.
#
# ATTN: 'MANGA_FOLDER' must be manually altered according to the data release
#       being ran.
#------------------------------------------------------------------------------
WORKING_IN_BLUEHIVE = True

if WORKING_IN_BLUEHIVE:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    LOCAL_PATH = '/home/jsm171'
    SCRATCH_PATH = '/scratch/jsm171'

    IMAGE_DIR = SCRATCH_PATH + '/images'
    MANGA_FOLDER = SCRATCH_PATH + '/manga_files/dr15'
    ROT_CURVE_MASTER_FOLDER = SCRATCH_PATH + '/rot_curve_data_files'

else:
    LOCAL_PATH = os.path.dirname(__file__)
    if LOCAL_PATH == '':
        LOCAL_PATH = '.'

    IMAGE_DIR = LOCAL_PATH + '/images'
    MANGA_FOLDER = LOCAL_PATH + '/manga_files/dr15'
    ROT_CURVE_MASTER_FOLDER = LOCAL_PATH + '/rot_curve_data_files'

ROT_CURVE_DATA_INDICATOR = '_rot_curve_data'
GAL_STAT_DATA_INDICATOR = '_gal_stat_data'

# Create output directories if they do not already exist
if not os.path.isdir( IMAGE_DIR):
    os.makedirs( IMAGE_DIR)
if not os.path.isdir( ROT_CURVE_MASTER_FOLDER):
    os.makedirs( ROT_CURVE_MASTER_FOLDER)
###############################################################################


###############################################################################
# Import functions from 'rotation_curve_vX_X.'
#------------------------------------------------------------------------------
from rotation_curve_v2_1 import extract_data, \
                                match_to_NSA, \
                                calc_rot_curve, \
                                write_rot_curve, \
                                write_master_file
###############################################################################


###############################################################################
# Create list of .fits file names to extract a rotation curve from.
#
# IMPORTANT: rot_curve_main.py must be run outside the folder that
#            houses the plate folders. The default folder is
#            '/manga_files'.
#------------------------------------------------------------------------------
files = glob.glob( MANGA_FOLDER + '/manga-*Pipe3D.cube.fits.gz')
###############################################################################


###############################################################################
# Code to isolate files and run it through all of the functions from
# rotation_curve_vX_X.
# ~   ~   ~   ~   ~   ~   ~   ~   ~   ~   ~   ~   ~   ~   ~   ~   ~   ~   ~   ~
#FILE_IDS = ['10001-12703']
#
#files = []
#for file_name in FILE_IDS:
#    files.append( MANGA_FOLDER \
#    + '/manga-' + file_name + '.Pipe3D.cube.fits.gz')
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
    #nsa_catalog = fits.open('/Users/kellydouglass/Documents/Drexel/Research/Data/nsa_v0_1_2.fits')

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


iteration_times = []
i = 0


###############################################################################
# This for loop runs through the necessary calculations to calculte and write
#    the rotation curve for all of the galaxies in the MaNGA survey.
# ~   ~   ~   ~   ~   ~   ~   ~   ~   ~   ~   ~   ~   ~   ~   ~   ~   ~   ~   ~
for file_name in files:
    iteration_start = datetime.datetime.now()

    ###########################################################################
    # file_id is a simplified string that identifies each file that is run
    #    through the algorithm. The file_id name scheme is [PLATE]-[FIBER ID].
    #--------------------------------------------------------------------------
    gal_ID = file_name[ file_name.find(MANGA_FOLDER) \
                           + len( MANGA_FOLDER) + 7: file_name.find('.Pipe3D')]

#    print( gal_ID)
    ###########################################################################


    ###########################################################################
    # Extract the necessary data from the .fits file.
    #--------------------------------------------------------------------------
    Ha_vel, Ha_vel_error, v_band, v_band_err, sMass_density, \
    manga_plate, manga_fiberID, gal_ra, gal_dec = extract_data( file_name)
    print( gal_ID, " EXTRACTED")
    ###########################################################################

    '''
    ###########################################################################
    # Add the MaNGA catalog information to the master arrays.
    #--------------------------------------------------------------------------
    manga_data_release_master.append( gal_ID[ 0: 4])
    manga_plate_master.append( manga_plate)
    manga_fiberID_master.append( manga_fiberID)
    ###########################################################################
    '''

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

    '''
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
    '''

    ###########################################################################
    # Extract rotation curve data for the .fits file in question and create an
    #    astropy Table containing said data.
    #--------------------------------------------------------------------------
    rot_data_table, gal_stat_table = calc_rot_curve( Ha_vel, Ha_vel_error, \
                                       v_band, v_band_err, sMass_density, \
                                       axes_ratio, phi_EofN_deg, zdist, \
                                       zdist_err, gal_ID, IMAGE_DIR, \
                                       image_format)
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

    iteration_end = datetime.datetime.now() - iteration_start
    print("ITERATION TIME:", iteration_end)
    iteration_times.append( iteration_end)

    i += 1
    if i == 1000:
        break

    '''
    print('Loop number:', loop_num)
    print('manga_data_release_master length:', len(manga_data_release_master), len(pickle.dumps(manga_data_release_master)))
    print('manga_plate_master length:', len(manga_plate_master), len(pickle.dumps(manga_plate_master)))
    print('manga_fiberID_master length:', len(manga_fiberID_master), len(pickle.dumps(manga_fiberID_master)))
    print('nsa_axes_ratio_master length:', len(nsa_axes_ratio_master), len(pickle.dumps(nsa_axes_ratio_master)))
    print('nsa_phi_master length:', len(nsa_phi_master), len(pickle.dumps(nsa_phi_master)))
    print('nsa_zdist_master length:', len(nsa_zdist_master), len(pickle.dumps(nsa_zdist_master)))
    print('nsa_zdist_err_master length:', len(nsa_zdist_err_master), len(pickle.dumps(nsa_zdist_err_master)))
    print('nsa_mStar_master length:', len(nsa_mStar_master), len(pickle.dumps(nsa_mStar_master)))
    print('nsa_ra_master length:', len(nsa_ra_master), len(pickle.dumps(nsa_ra_master)))
    print('nsa_dec_master length:', len(nsa_dec_master), len(pickle.dumps(nsa_dec_master)))
    print('nsa_plate_master length:', len(nsa_plate_master), len(pickle.dumps(nsa_plate_master)))
    print('nsa_fiberID_master length:', len(nsa_fiberID_master), len(pickle.dumps(nsa_fiberID_master)))
    print('nsa_mjd_master length:', len(nsa_mjd_master), len(pickle.dumps(nsa_mjd_master)))
    print('nsaID_master length:', len(nsaID_master), len(pickle.dumps(nsaID_master)))
    print('Memory usage (bytes):', process.memory_info().rss)

    memory_list.append( process.memory_info().rss)
    '''

    print("\n")
# ~ # ~ # ~ # ~ # ~ # ~ # ~ # ~ # ~ # ~ # ~ # ~ # ~ # ~ # ~ # ~ # ~ # ~ # ~ # ~


###############################################################################
# Build master file that contains identifying information for each galaxy
#   as well as scientific information as taken from the NSA catalog.
#------------------------------------------------------------------------------
#write_master_file( manga_plate_master, manga_fiberID_master,
#                  manga_data_release_master,
#                  nsa_plate_master, nsa_fiberID_master, nsa_mjd_master,
#                  nsaID_master, nsa_ra_master, nsa_dec_master,
#                  nsa_axes_ratio_master, nsa_phi_master, nsa_zdist_master,
#                  nsa_mStar_master,
#                  LOCAL_PATH)
#print("MASTER FILE WRITTEN")
###############################################################################


iteration_clock_fig = plt.figure()
plt.title( 'Iteration Time VS File Index')
plt.plot( iteration_times)
plt.savefig( IMAGE_DIR + "iteration_clock")
plt.close()
del iteration_clock_fig


###############################################################################
# Clock the program's run time to check performance.
#------------------------------------------------------------------------------
FINISH = datetime.datetime.now()
print("Runtime:", FINISH - START)
###############################################################################

'''
###############################################################################
# Plot memory usage for each galaxy
#------------------------------------------------------------------------------
plt.figure()
plt.plot(memory_list, '.')
plt.xlabel('Iteration number')
plt.ylabel('Memory usage [bytes]')
plt.show()
###############################################################################
'''