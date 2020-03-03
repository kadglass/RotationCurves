################################################################################
# IMPORT MODULES
#-------------------------------------------------------------------------------
import datetime
START = datetime.datetime.now()

import glob, os.path, warnings

import numpy as np

from astropy.io import fits
import astropy.units as u

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
FILE_IDS = ['7443-12705']
RUN_ALL_GALAXIES = False
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
    IMAGE_DIR = LOCAL_PATH + 'Images/'

    # Create directory if it does not already exist
    if not os.path.isdir( IMAGE_DIR):
        os.makedirs( IMAGE_DIR)
else:
    IMAGE_DIR = None

MANGA_FOLDER = LOCAL_PATH + '../data/MaNGA/MaNGA_DR16/HYB10-GAU-MILESHC/'
PIPE3D_folder = LOCAL_PATH + '../data/MaNGA/MaNGA_DR15/pipe3d/'
ROT_CURVE_MASTER_FOLDER = LOCAL_PATH + 'DRP_rot_curve_data_files/'

# Create output directory if it does not already exist
if not os.path.isdir( ROT_CURVE_MASTER_FOLDER):
    os.makedirs( ROT_CURVE_MASTER_FOLDER)
################################################################################



################################################################################
# Import functions from 'DRP_rotation_curve'
#-------------------------------------------------------------------------------
from DRP_rotation_curve import extract_data, \
                               extract_Pipe3d_data, \
                               match_to_DRPall, \
                               calc_rot_curve, \
                               write_rot_curve, \
                               write_master_file
################################################################################



if RUN_ALL_GALAXIES:
    ############################################################################
    # Create list of .fits file names to extract a rotation curve from.
    #---------------------------------------------------------------------------
    files = glob.glob( MANGA_FOLDER + '*/manga-*-MAPS-HYB10-GAU-MILESHC.fits.gz')
    ############################################################################

else:
    ############################################################################
    # Code to isolate select files and run it through all of the functions from
    # DRP_rotation_curve.
    #---------------------------------------------------------------------------
    files = []

    for file_name in FILE_IDS:
        [plate, fiberID] = file_name.split('-')
        files.append( MANGA_FOLDER + plate + '/manga-' + file_name + '-MAPS-HYB10-GAU-MILESHC.fits.gz')
    ############################################################################



################################################################################
# Extract the length of the 'files' array for future use in creating the
# 'master_table.'
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

axes_ratio_all = general_data[1].data['nsa_elpetro_ba']
phi_EofN_deg_all = general_data[1].data['nsa_elpetro_phi']

rabsmag_all = general_data[1].data['nsa_elpetro_absmag'][:,4] # SDSS r-band
mStar_all = general_data[1].data['nsa_elpetro_mass']

manga_target_galaxy_all = general_data[1].data['mngtarg1'] != 0
data_quality_all = general_data[1].data['drp3qual'] < 10000

general_data.close()
################################################################################



################################################################################
# Initialize the master arrays that create the structure of the master file.
#-------------------------------------------------------------------------------
manga_plate_master = -1 * np.ones( N_files)
manga_IFU_master = -1 * np.ones( N_files)

axes_ratio_master = -1. * np.ones( N_files)
phi_master = -1. * np.ones( N_files)
z_master = -1. * np.ones( N_files)
mStar_master = -1. * np.ones( N_files)
rabsmag_master = np.zeros( N_files)

ra_master = -1. * np.ones( N_files)
dec_master = -1. * np.ones( N_files)
################################################################################



################################################################################
# Calculate and write the rotation curve for all of the galaxies in the 'files' 
# array.
#-------------------------------------------------------------------------------
num_masked_gal = 0 # Number of completely masked galaxies

for i in range( len( files)):
    #iteration_start = datetime.datetime.now()
    file_name = files[i]
    #print( file_name)
    
    ############################################################################
    # file_id is a simplified string that identifies each file that is run
    # through the algorithm.  The file_id name scheme is [PLATE]-[IFU].
    #---------------------------------------------------------------------------
    gal_ID = file_name[ file_name.find('manga-') + 6 : file_name.find('-MAPS')]

    [manga_plate, manga_IFU] = gal_ID.split('-')
    ############################################################################


    ############################################################################
    # Extract the necessary data from the .fits files.
    #---------------------------------------------------------------------------
    Ha_vel, Ha_vel_ivar, Ha_vel_mask, r_band, r_band_ivar = extract_data( file_name)
    sMass_density = extract_Pipe3d_data( PIPE3D_folder, gal_ID)
    print( gal_ID, " EXTRACTED")
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
    print(gal_ID, " MATCHED")

    axes_ratio = axes_ratio_all[ DRPall_gal_idx]
    phi_EofN_deg = phi_EofN_deg_all[ DRPall_gal_idx] * u.degree
    z = z_all[ DRPall_gal_idx]

    mStar = mStar_all[ DRPall_gal_idx] * u.M_sun
    rabsmag = rabsmag_all[ DRPall_gal_idx]

    ra = ra_all[ DRPall_gal_idx]
    dec = dec_all[ DRPall_gal_idx]

    galaxy_target = manga_target_galaxy_all[ DRPall_gal_idx]
    good_galaxy = data_quality_all[ DRPall_gal_idx]
    ############################################################################

    
    ############################################################################
    # Add the DRPall catalog information to the master arrays.
    #---------------------------------------------------------------------------
    axes_ratio_master[i] = axes_ratio
    phi_master[i] = phi_EofN_deg / u.degree
    z_master[i] = z
    mStar_master[i] = mStar / u.M_sun
    rabsmag_master[i] = rabsmag

    ra_master[i] = ra
    dec_master[i] = dec
    ############################################################################
    
    
    if galaxy_target and good_galaxy:
        ########################################################################
        # Extract rotation curve data for the .fits file in question and create 
        # an astropy Table containing said data.
        #-----------------------------------------------------------------------
        rot_data_table, gal_stat_table, num_masked_gal = calc_rot_curve( Ha_vel, 
                                                                         Ha_vel_ivar, 
                                                                         Ha_vel_mask, 
                                                                         r_band, 
                                                                         r_band_ivar, 
                                                                         sMass_density, 
                                                                         axes_ratio, 
                                                                         phi_EofN_deg, 
                                                                         z, gal_ID, 
                                                                         IMAGE_DIR=IMAGE_DIR, 
                                                                         #IMAGE_FORMAT, 
                                                                         num_masked_gal=num_masked_gal)
        print(gal_ID, " ROT CURVE CALCULATED")
        ########################################################################
    
    
        ########################################################################
        # Write the rotation curve data to a text file in ascii format.
        #
        # IMPORTANT: DRP_rot_curve_main.py writes the data files into the 
        #            default folder 'DRP_rot_curve_data_files'. It also saves 
        #            the file with the default extension '_rot_curve_data'.
        #-----------------------------------------------------------------------
        write_rot_curve( rot_data_table, 
                         gal_stat_table, 
                         gal_ID, 
                         ROT_CURVE_MASTER_FOLDER)

        print(gal_ID, " WRITTEN")
        ########################################################################


    print("\n")
################################################################################



################################################################################
# Build master file that contains identifying information for each galaxy as 
# well as scientific information as taken from the DRPall catalog.
#-------------------------------------------------------------------------------
if RUN_ALL_GALAXIES:
    write_master_file( manga_plate_master, manga_IFU_master,
                      ra_master, dec_master, z_master,
                      axes_ratio_master, phi_master, 
                      mStar_master, rabsmag_master, 
                      LOCAL_PATH)
    print("MASTER FILE WRITTEN")
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

