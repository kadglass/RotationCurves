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
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import pickle, psutil
process = psutil.Process(os.getpid())
memory_list = []
'''
warnings.simplefilter('ignore', np.RankWarning)

###############################################################################
# File format for saved images
#------------------------------------------------------------------------------
IMAGE_FORMAT = 'eps'
###############################################################################

###############################################################################
# Boolean variable to specify if the script is being run in Bluehive.
#------------------------------------------------------------------------------
WORKING_IN_BLUEHIVE = False
RUN_ALL_GALAXIES = True
###############################################################################

###############################################################################
# List of files (in "[MaNGA_plate]-[MaNGA_fiberID]" format) to be ran through
#    the individual galaxy version of this script.
#------------------------------------------------------------------------------
FILE_IDS = ['7443-12705']
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
if WORKING_IN_BLUEHIVE:
    LOCAL_PATH = '/home/jsm171'
    SCRATCH_PATH = '/scratch/jsm171'

    IMAGE_DIR = SCRATCH_PATH + '/images/'
    MANGA_FOLDER = SCRATCH_PATH + '/manga_files/dr15/'
    ROT_CURVE_MASTER_FOLDER = SCRATCH_PATH + '/rot_curve_data_files/'

else:
    LOCAL_PATH = os.path.dirname(__file__)
    if LOCAL_PATH == '':
        LOCAL_PATH = '.'

    IMAGE_DIR = LOCAL_PATH + '/Images/'
    MANGA_FOLDER = LOCAL_PATH + '../data/MaNGA/MaNGA_DR15/pipe3d/'
    ROT_CURVE_MASTER_FOLDER = LOCAL_PATH + '/rot_curve_data_files/'

# Create output directories if they do not already exist
if not os.path.isdir( IMAGE_DIR):
    os.makedirs( IMAGE_DIR)
if not os.path.isdir( ROT_CURVE_MASTER_FOLDER):
    os.makedirs( ROT_CURVE_MASTER_FOLDER)
###############################################################################


###############################################################################
# Import functions from 'Pipe3D_rotation_curve'
#------------------------------------------------------------------------------
from Pipe3D_rotation_curve import extract_data, \
                                  match_to_NSA, \
                                  calc_rot_curve, \
                                  write_rot_curve, \
                                  write_master_file
###############################################################################

if RUN_ALL_GALAXIES:
    ###########################################################################
    # Create list of .fits file names to extract a rotation curve from.
    #--------------------------------------------------------------------------
    files = glob.glob( MANGA_FOLDER + '*/manga-*.Pipe3D.cube.fits.gz')
    ###########################################################################


else:
    ###########################################################################
    # Code to isolate files and run it through all of the functions from
    # rotation_curve_vX_X.
    # ~   ~   ~   ~   ~   ~   ~   ~   ~   ~   ~   ~   ~   ~   ~   ~   ~   ~   ~
    files = []

    for file_name in FILE_IDS:
        [plate, fiberID] = file_name.split('-')
        files.append( MANGA_FOLDER + plate + '/manga-' + file_name + '.Pipe3D.cube.fits.gz')
    ###########################################################################


###############################################################################
# Extract the length of the 'files' array for future use in creating the
#    'master_table.'
#------------------------------------------------------------------------------
N_files = len( files)
###############################################################################


###############################################################################
# Open NASA-Sloan-Atlas (NSA) master catalog and extract the data structures
#    for RA; DEC; the axes ratio of b/a (obtained via sersic fit); phi, the
#    angle of rotation in the two-dimensional, observational plane (obtained
#    via sersic fit); the redshift distance calculated from the shift in
#    H-alpha flux; and the absolute magnitude in the r-band.
#
# Note: The NSA RA and DEC are passed to a SkyCoord object to match galaxies to 
#       the NSA catalog index.
#------------------------------------------------------------------------------
if WORKING_IN_BLUEHIVE:
    nsa_catalog_filename = SCRATCH_PATH + '/nsa_v1_0_1.fits'
else:
    #nsa_catalog_filename = LOCAL_PATH + '/nsa_v1_0_1.fits'
    nsa_catalog_filename = '/Users/kellydouglass/Documents/Drexel/Research/Data/nsa_v1_0_1.fits'
nsa_catalog = fits.open( nsa_catalog_filename)

nsa_axes_ratio_all = nsa_catalog[1].data['SERSIC_BA']
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
    nsa_mStar_all = nsa_catalog[1].data['ELPETRO_LOGMASS']
else:
    nsa_mStar_all = nsa_catalog[1].data['MASS']

nsa_catalog.close()

catalog_coords = SkyCoord( ra=nsa_ra_all*u.degree, dec=nsa_dec_all*u.degree)
###############################################################################


###############################################################################
# # Initialize the master arrays that create the structure of the master file.
#------------------------------------------------------------------------------
manga_plate_master = -1 * np.ones( N_files)
manga_fiberID_master = -1 * np.ones( N_files)

axes_ratio_master = -1. * np.ones( N_files)
phi_master = -1. * np.ones( N_files)
z_master = -1. * np.ones( N_files)
#zdist_master = -1. * np.ones( N_files)
#zdist_err_master = -1. * np.ones( N_files)
mStar_master = -1. * np.ones( N_files)
rabsmag_master = np.zeroes( N_files)

ra_master = -1. * np.ones( N_files)
dec_master = -1. * np.ones( N_files)
nsa_plate_master = -1 * np.ones( N_files)
nsa_fiberID_master = -1 * np.ones( N_files)
nsa_mjd_master = -1 * np.ones( N_files)
nsaID_master = -1 * np.ones( N_files)
###############################################################################

'''
###############################################################################
# Create an array to store the time spent on each iteration of the fot-loop.
#    This is used to clock the algorithm for analysis.
#------------------------------------------------------------------------------
iteration_times = []
###############################################################################
'''

###############################################################################
# This for loop runs through the necessary calculations to calculte and write
#    the rotation curve for all of the galaxies in the 'files' array.
# ~   ~   ~   ~   ~   ~   ~   ~   ~   ~   ~   ~   ~   ~   ~   ~   ~   ~   ~   ~
num_masked_gal = 0 # Number of completely masked galaxies

for i in range( len( files)):
    #iteration_start = datetime.datetime.now()
    file_name = files[i]
    #print( file_name)
    
    ###########################################################################
    # file_id is a simplified string that identifies each file that is run
    #    through the algorithm. The file_id name scheme is [PLATE]-[FIBER ID].
    #--------------------------------------------------------------------------
    gal_ID = file_name[ file_name.find('manga-') + 6 : file_name.find('.Pipe3D')]
    ###########################################################################


    ###########################################################################
    # Extract the necessary data from the .fits file.
    #--------------------------------------------------------------------------
    galaxy_target, good_galaxy, Ha_vel, Ha_vel_error, v_band, v_band_err, 
    sMass_density, manga_plate, manga_fiberID, gal_ra, gal_dec = extract_data( file_name)
    print( gal_ID, " EXTRACTED")
    ###########################################################################
    
    
    ###########################################################################
    # Add the MaNGA catalog information to the master arrays.
    #--------------------------------------------------------------------------
    manga_plate_master[i] = manga_plate
    manga_fiberID_master[i] = manga_fiberID
    ###########################################################################
    

    ###########################################################################
    # Match the galaxy's RA and DEC from the to the NSA catalog index, and pull
    # out the matched data from the NSA catalog.
    #--------------------------------------------------------------------------
    nsa_gal_idx = match_to_NSA( gal_ra, gal_dec, catalog_coords)
    print(gal_ID, " MATCHED")

    axes_ratio = axes_ratio_all[ nsa_gal_idx]
    phi_EofN_deg = phi_EofN_deg_all[ nsa_gal_idx] * u.degree
    z = z_all[ nsa_gal_idx]
    #zdist = zdist_all[ nsa_gal_idx]
    #zdist_err = zdist_all_err[ nsa_gal_idx]
    mStar = mStar_all[ nsa_gal_idx] * u.M_sun
    rabsmag = absmag_all[ nsa_gal_idx][4] # SDSS r-band

    ra = ra_all[ nsa_gal_idx]
    dec = dec_all[ nsa_gal_idx]
    nsa_plate = nsa_plate_all[ nsa_gal_idx]
    nsa_fiberID = nsa_fiberID_all[ nsa_gal_idx]
    nsa_mjd = nsa_mjd_all[ nsa_gal_idx]
    nsaID = nsaID_all[ nsa_gal_idx]
    ###########################################################################

    
    ###########################################################################
    # Add the NSA catalog information to the master arrays.
    #--------------------------------------------------------------------------
    axes_ratio_master[i] = axes_ratio
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
    ###########################################################################
    
    
    if galaxy_target and good_galaxy:
        ########################################################################
        # Extract rotation curve data for the .fits file in question and create 
        # an astropy Table containing said data.
        #-----------------------------------------------------------------------
        rot_data_table, gal_stat_table, num_masked_gal = calc_rot_curve( Ha_vel, 
                                                                         Ha_vel_error, 
                                                                         v_band, 
                                                                         v_band_err, 
                                                                         sMass_density, 
                                                                         axes_ratio, 
                                                                         phi_EofN_deg, 
                                                                         z, gal_ID, 
                                                                         #IMAGE_DIR, 
                                                                         #IMAGE_FORMAT, 
                                                                         num_masked_gal=num_masked_gal)
        print(gal_ID, " ROT CURVE CALCULATED")
        ########################################################################
    
    
        ########################################################################
        # Write the rotation curve data to a text file in ascii format.
        #
        # IMPORTANT: rot_curve_main.py writes the data files into the default
        #            folder 'rot_curve_data_files'. It also saves the file with 
        #            the default extension '_rot_curve_data'.
        #-----------------------------------------------------------------------
        write_rot_curve( rot_data_table, gal_stat_table, gal_ID, ROT_CURVE_MASTER_FOLDER)

        print(gal_ID, " WRITTEN")
        ########################################################################
    
    '''
    ###########################################################################
    # Clock the current iteration and append the time to 'iteration_times'
    #    which is plotted below.
    #--------------------------------------------------------------------------
    iteration_end = datetime.datetime.now() - iteration_start
    print("ITERATION TIME:", iteration_end)
    iteration_times.append( iteration_end.total_seconds())
    ###########################################################################
    
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

'''
###############################################################################
# Histogram the iteration time for each loop.
#------------------------------------------------------------------------------
#BINS = np.linspace( 0, 18, 37)

iteration_clock_fig = plt.figure()
plt.title('Iteration Time Histogram')
plt.xlabel('Iteration Time [sec]')
plt.ylabel('Percentage of Galaxies')
#plt.xticks( np.arange( 0, 19, 1))
plt.hist( iteration_times,
#         BINS,
         color='indianred', density=True)
plt.savefig( IMAGE_DIR + "/histograms/iteration_clock_hist",
            format=image_format)
plt.show()
plt.close()
del iteration_clock_fig
###############################################################################
'''

###############################################################################
# Build master file that contains identifying information for each galaxy
# as well as scientific information as taken from the DRPall catalog.
#------------------------------------------------------------------------------
write_master_file( manga_plate_master, manga_fiberID_master,
                  nsa_plate_master, nsa_fiberID_master, nsa_mjd_master, nsaID_master, 
                  ra_master, dec_master, z_master,
                  axes_ratio_master, phi_master, 
                  mStar_master, rabsmag_master, 
                  LOCAL_PATH)
print("MASTER FILE WRITTEN")
###############################################################################


###############################################################################
# Print number of galaxies that were completely masked
#------------------------------------------------------------------------------
print('There were', num_masked_gal, 'galaxies that were completely masked.')
###############################################################################


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