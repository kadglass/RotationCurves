################################################################################
# IMPORT MODULES
#-------------------------------------------------------------------------------
import datetime
START = datetime.datetime.now()

import glob, os.path, warnings

import numpy as np

from astropy.table import QTable
import astropy.units as u

from DRP_rotation_curve import extract_data, \
                               extract_Pipe3d_data, \
                               calc_rot_curve, \
                               write_rot_curve

warnings.simplefilter('ignore', np.RankWarning)
################################################################################



################################################################################
# File format for saved images
#-------------------------------------------------------------------------------
IMAGE_FORMAT = 'eps'
################################################################################



################################################################################
# List of files (in "[MaNGA_plate]-[MaNGA_IFU]" format) to be ran through the
# individual galaxy version of this script.
# 
# If RUN_ALL_GALAXIES is set to True, then code will ignore what is in FILE_IDS.
#-------------------------------------------------------------------------------
FILE_IDS = ['8713-6101']
RUN_ALL_GALAXIES = True
################################################################################



################################################################################
# 'LOCAL_PATH' should be updated depending on the file structure (e.g. if
# working in bluehive). It is set to 'os.path.dirname(__file__)' when working on 
# a local system.
#
# ATTN: 'MANGA_FOLDER' must be manually altered according to the data release
#       being ran.
#-------------------------------------------------------------------------------
LOCAL_PATH = os.path.dirname(__file__)
if LOCAL_PATH == '':
    LOCAL_PATH = './'

if RUN_ALL_GALAXIES:
    IMAGE_DIR = LOCAL_PATH + 'Images/DRP/'

    # Create directory if it does not already exist
    if not os.path.isdir( IMAGE_DIR):
        os.makedirs( IMAGE_DIR)
else:
    IMAGE_DIR = None

MANGA_FOLDER = LOCAL_PATH + '../data/MaNGA/MaNGA_DR16/HYB10-GAU-MILESHC/'
PIPE3D_folder = LOCAL_PATH + '../data/MaNGA/MaNGA_DR15/pipe3d/'
ROT_CURVE_MASTER_FOLDER = LOCAL_PATH + 'DRP-rot_curve_data_files/'
MASTER_FILENAME = 'DRPall-master_file.txt'

# Create output directory if it does not already exist
if not os.path.isdir( ROT_CURVE_MASTER_FOLDER):
    os.makedirs( ROT_CURVE_MASTER_FOLDER)
################################################################################



################################################################################
# Open the master file
#-------------------------------------------------------------------------------
master_table = QTable.read( MASTER_FILENAME, format='ascii.ecsv')


master_index = {}

for i in range(len(master_table)):
    plate = master_table['MaNGA_plate'][i]
    IFU = master_table['MaNGA_IFU'][i]

    master_index[str(plate) + '-' + str(IFU)] = i
################################################################################



################################################################################
# Create a list of galaxy IDs for which to extract a rotation curve.
#-------------------------------------------------------------------------------
if RUN_ALL_GALAXIES:
    
    N_files = len(master_table)

    FILE_IDS = list(master_index.keys())

else:

    N_files = len(FILE_IDS)
################################################################################



################################################################################
# Calculate and write the rotation curve for all of the galaxies in the 'files' 
# array.
#-------------------------------------------------------------------------------
num_masked_gal = 0 # Number of completely masked galaxies

for gal_ID in FILE_IDS:
    
    ############################################################################
    # gal_id is a simplified string that identifies each file that is run
    # through the algorithm.  The gal_id name scheme is [PLATE]-[IFU].
    #---------------------------------------------------------------------------
    manga_plate, manga_IFU = gal_ID.split('-')

    file_name = MANGA_FOLDER + manga_plate + '/manga-' + gal_ID + '-MAPS-HYB10-GAU-MILESHC.fits.gz'
    ############################################################################


    ############################################################################
    # Extract the necessary data from the .fits files.
    #---------------------------------------------------------------------------
    Ha_vel, Ha_vel_ivar, Ha_vel_mask, r_band, r_band_ivar = extract_data( file_name)
    sMass_density = extract_Pipe3d_data( PIPE3D_folder, gal_ID)

    print( gal_ID, " EXTRACTED")
    ############################################################################


    ############################################################################
    # Extract the necessary data from the master table.
    #---------------------------------------------------------------------------
    i_master = master_index[gal_ID]

    axis_ratio = master_table['ba'][ i_master]
    phi_EofN_deg = master_table['phi'][ i_master]

    z = master_table['redshift'][ i_master]
    ############################################################################
    
    
    ############################################################################
    # Extract rotation curve data for the .fits file in question and create an 
    # astropy Table containing said data.
    #---------------------------------------------------------------------------
    rot_data_table, gal_stat_table, num_masked_gal = calc_rot_curve( Ha_vel, 
                                                                     Ha_vel_ivar, 
                                                                     Ha_vel_mask, 
                                                                     r_band, 
                                                                     r_band_ivar, 
                                                                     sMass_density, 
                                                                     axis_ratio, 
                                                                     phi_EofN_deg, 
                                                                     z, gal_ID, 
                                                                     IMAGE_DIR=IMAGE_DIR, 
                                                                     #IMAGE_FORMAT=IMAGE_FORMAT, 
                                                                     num_masked_gal=num_masked_gal)
    print(gal_ID, " ROT CURVE CALCULATED")
    ############################################################################


    ############################################################################
    # Write the rotation curve data to a text file in ascii format.
    #---------------------------------------------------------------------------
    write_rot_curve( rot_data_table, 
                     gal_stat_table, 
                     gal_ID, 
                     ROT_CURVE_MASTER_FOLDER)

    print(gal_ID, " WRITTEN")
    ############################################################################

    print("\n")
################################################################################



################################################################################
# Print number of galaxies that were completely masked
#-------------------------------------------------------------------------------
if RUN_ALL_GALAXIES:
    print('There were', num_masked_gal, 'galaxies that were completely masked.')
################################################################################


################################################################################
# Clock the program's run time to check performance.
#-------------------------------------------------------------------------------
FINISH = datetime.datetime.now()
print("Runtime:", FINISH - START)
################################################################################

