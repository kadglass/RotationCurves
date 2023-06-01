'''
Main script to extract and fit the stellar (disk) mass rotation curve in disk 
galaxies.
'''


from time import time
START = time()

import os

import numpy as np
import numpy.ma as ma

from astropy.table import Table

from file_io import add_disk_columns, fillin_output_table

from DRP_rotation_curve import extract_data, extract_Pipe3d_data

from DRP_vel_map_functions import build_map_mask

from disk_mass import calc_mass_curve, fit_mass_curve

#import matplotlib.pyplot as plt

from rotation_curve_functions import disk_mass



################################################################################
# Constants
#-------------------------------------------------------------------------------
H_0 = 100      # Hubble's Constant in units of h km/s/Mpc
c = 299792.458 # Speed of light in units of km/s
################################################################################




################################################################################
# File format for saved images
#-----------------------------------------------s--------------------------------
IMAGE_FORMAT = 'eps'
################################################################################



################################################################################
# List of files (in "[MaNGA plate]-[MaNGA IFU]" format) to be run through the 
# individual galaxy version of this script.
#
# If RUN_ALL_GALAXIES is set to True, then the code will ignore what is in 
# FILE_IDS
#-------------------------------------------------------------------------------


'''
FILE_IDS =  ['10001-3702','7815-3702',  '7990-6104',
        '7992-6104', '8077-12704', '8077-6102',
        '8082-12702', '8082-9102',  '8146-1901',
        '8150-9102', '8153-1901', '8155-12701', '8158-1901',
        '8255-12704', '8257-9102',
        '8262-3702', '8262-9102', '8311-3703',
        '8320-9101', '8322-1901',
        '8322-3701', '8329-12701', '8330-12703', '8332-3702',
        '8332-9102', '8335-12705', '8338-12701', '8440-6104',
        '8450-6102', '8452-12705',
        '8453-3704', '8455-3701', '8459-1901', '8462-9101',
        '8465-9102', '8466-12702', '8483-6101', '8547-6102',
        '8548-12704', '8551-12705', '8552-12701', 
         '8588-6101', '8592-6101',
        '8595-3703', '8600-1901', '8600-3704',
        '8601-12702', '8603-12704', '8603-6103', '8603-6104',
        '8604-12702', '8604-9102', '8624-12702', '8624-12703',
        '8626-12701', '8626-12702', '8626-3702', '8626-3703',
        '8712-6101', '8713-6104',  '8727-3701',
        '8932-9102', '8935-6104', '8945-12701', '8950-12705',
        '8952-6104','8979-6102', '8987-3701',
        '8989-9102', '8993-6104', '8996-3703', '8997-12704', 
        '9027-12701', '9029-12702', '9029-6102',
        '9036-9102', '9041-12701', '9042-12703',
        '9044-6101', '9047-6104', '9049-6104', '9050-3704',
        '9050-9101', '9085-12703', 
        '9095-9102',
        '9196-6103', '9485-12705', '9485-3701', '9486-12701',
        '9486-12702', '9500-1901', '9508-12705',
        '9508-6101', '9508-6104', '9865-9102', '9871-6101',
        '9881-12705', '9881-3702', '9888-12704', '9888-12705',
        '8092-3701', '11744-6103', '7977-3704',
        '7977-3703', '12085-6103', '12090-1901', '12085-9102',
        '9863-1902',
        '8089-12701', '9495-6101', '9506-1901', '10844-3703', '12769-6104', 
        '10513-1901', '12700-12702',
        '9493-6103', '9494-3701', '10219-1901', '8087-9102',
        '10514-3701', '8934-3701']'''

FILE_IDS = ['8082-12702']


RUN_ALL_GALAXIES = False
TEXT_OUT = True



fit_function = 'bulge'
################################################################################



################################################################################
# 'LOCAL_PATH' should be updated depending on the file structure (e.g. if
# working in bluehive).  It is set to 'os.path.dirname(__file__)' when working 
# on a local system.
#
# ATTN: 'MANGA_FOLDER' must be manually altered according to the data release
#       being analyzed.
#-------------------------------------------------------------------------------
LOCAL_PATH = os.path.dirname(__file__)
if LOCAL_PATH == '':
    LOCAL_PATH = './'

if RUN_ALL_GALAXIES:
    IMAGE_DIR = LOCAL_PATH + 'Images/DRP-Pipe3d/'
    #IMAGE_DIR = '/home/idies/workspace/Storage/nityaravi/OutputFiles/'

    # Create directory if it does not already exist
    if not os.path.isdir( IMAGE_DIR):
        os.makedirs( IMAGE_DIR)
else:
    IMAGE_DIR = '/Users/nityaravi/Documents/Research/RotationCurves/data/manga/Images/DiskMass/bulge_chi2_test/'
    #IMAGE_DIR = LOCAL_PATH + 'Images/DRP-Pipe3d/'
    #IMAGE_DIR = '/home/idies/workspace/Storage/nityaravi/OutputFiles/'



# old versions
# MANGA_FOLDER = '/Users/kellydouglass/Documents/Research/data/SDSS/dr16/manga/spectro/'
# SDSS_FOLDER = '/Users/kellydouglass/Documents/Research/data/SDSS/'
# MANGA_FOLDER = '/home/kelly/Documents/Data/SDSS/dr16/manga/spectro/'
# SDSS_FOLDER = '/home/kelly/Documents/Data/SDSS/'
# MASS_MAP_FOLDER = SDSS_FOLDER + 'dr15/manga/spectro/pipe3d/v2_4_3/2.4.3/'
# VEL_MAP_FOLDER = SDSS_FOLDER + 'dr16/manga/spectro/analysis/v2_4_3/2.2.1/HYB10-GAU-MILESHC/'



# for sciserver
# MANGA_FOLDER = '/home/idies/workspace/sdss_sas/dr17/manga/spectro/'
# NSA_FILENAME = '/home/idies/workspace/Storage/nityaravi/RotationCurves/nsa_v1_0_1.fits'
# MASS_MAP_FOLDER = MANGA_FOLDER + 'pipe3d/v3_1_1/3.1.1/'
# VEL_MAP_FOLDER = MANGA_FOLDER + 'analysis/v3_1_1/3.1.0/HYB10-MILESHC-MASTARSSP/'


# for nitya's local machine
MANGA_FOLDER = '/Users/nityaravi/Documents/Research/RotationCurves/data/manga/'
NSA_FILENAME = '/Users/nityaravi/Documents/Research/RotationCurves/data/nsa_v1_0_1.fits'
MASS_MAP_FOLDER = MANGA_FOLDER + 'Pipe3D/'
VEL_MAP_FOLDER = MANGA_FOLDER + 'DR17/'



# for bluehive
# MANGA_FOLDER = '/scratch/kdougla7/data/SDSS/dr17/manga/spectro/'
# NSA_FILENAME = '/scratch/kdougla7/data/NSA/nsa_v1_0_1.fits'
# MASS_MAP_FOLDER = MANGA_FOLDER + 'pipe3d/'
# VEL_MAP_FOLDER = MANGA_FOLDER + 'analysis/v3_1_1/3.1.0/HYB10-MILESHC-MASTARSSP/'


MASS_CURVE_MASTER_FOLDER = MASS_MAP_FOLDER + 'Pipe3d-mass_curve_data_files/'
# MASS_CURVE_MASTER_FOLDER = IMAGE_DIR
if not os.path.isdir(MASS_CURVE_MASTER_FOLDER):
    os.makedirs(MASS_CURVE_MASTER_FOLDER)

GALAXIES_FILENAME = '/Users/nityaravi/Documents/Research/RotationCurves/data/manga/output_files/DRP_HaVel_map_results_BB_smooth_lt_2.0_.fits'
#GALAXIES_FILENAME = '/home/idies/workspace/Storage/nityaravi/RotationCurves/DRP_HaVel_map_results_BB_smooth_lt_2.0_.fits'
#DRP_FILENAME = '/home/idies/workspace/sdss_sas/dr17/manga/spectro/redux/v3_1_1/drpall-v3_1_1.fits'

DRP_FILENAME = MANGA_FOLDER + 'DR17/' + 'drpall-v3_1_1.fits'
################################################################################



################################################################################
# Open the DRPall file
#-------------------------------------------------------------------------------
DRP_table = Table.read( DRP_FILENAME, format='fits')


DRP_index = {}

for i in range(len(DRP_table)):
    gal_ID = DRP_table['plateifu'][i]

    DRP_index[gal_ID] = i
################################################################################



################################################################################
# Open the galaxies file
#-------------------------------------------------------------------------------
#galaxies_table = Table.read( GALAXIES_FILENAME, format='ascii.ecsv')
galaxies_table = Table.read(GALAXIES_FILENAME, format='fits')


galaxies_index = {}

for i in range(len(galaxies_table)):
    gal_ID = galaxies_table['plateifu'][i]

    galaxies_index[gal_ID] = i
################################################################################



################################################################################
# Create a list of galaxy IDs for which to fit the mass rotation curve.
#-------------------------------------------------------------------------------
if RUN_ALL_GALAXIES:
    
    N_files = len(galaxies_table)

    FILE_IDS = list(galaxies_index.keys())

    galaxies_table = add_disk_columns(galaxies_table)

else:

    N_files = len(FILE_IDS)
    galaxies_table = add_disk_columns(galaxies_table)
################################################################################




################################################################################
# Fit the rotation curve for the stellar mass density map for all of the 
# galaxies in the 'files' array.
#-------------------------------------------------------------------------------
for gal_ID in FILE_IDS:
    
    ############################################################################
    # Extract the necessary data from the .fits files.
    #---------------------------------------------------------------------------
    maps = extract_data(VEL_MAP_FOLDER, 
                        gal_ID, 
                        ['Ha_vel', 'r_band', 'Ha_flux', 'Ha_sigma'])
    sMass_density, sMass_density_err = extract_Pipe3d_data(MASS_MAP_FOLDER, gal_ID)

    if maps is None or sMass_density is None:
        print('\n')
        continue

    print( gal_ID, "extracted")
    ############################################################################


    i_gal = galaxies_index[gal_ID]
    i_DRP = DRP_index[gal_ID]


    ########################################################################
    # Extract the necessary data from the galaxies table.
    #-----------------------------------------------------------------------
    if np.isfinite(galaxies_table['phi'][i_gal]):

        axis_ratio = galaxies_table['ba'][i_gal]
        axis_ratio_err = galaxies_table['ba_err'][i_gal]

        phi_EofN_deg = galaxies_table['phi'][i_gal]
        phi_EofN_deg_err = galaxies_table['phi_err'][i_gal]

        center_x = galaxies_table['x0'][i_gal]
        center_x_err = galaxies_table['x0_err'][i_gal]

        center_y = galaxies_table['y0'][i_gal]
        center_y_err = galaxies_table['y0_err'][i_gal]

        fit_flag = galaxies_table['fit_flag'][i_gal]

        ########################################################################
        # Create mask based on fit method
        #-----------------------------------------------------------------------
        map_mask = build_map_mask(gal_ID, 
                                  fit_flag, 
                                  ma.array(maps['Ha_vel'], mask=maps['Ha_vel_mask']), 
                                  ma.array(maps['Ha_flux'], mask=maps['Ha_flux_mask']), 
                                  ma.array(maps['Ha_flux_ivar'], mask=maps['Ha_flux_mask']), 
                                  ma.array(maps['Ha_sigma'], mask=maps['Ha_sigma_mask']))
        ########################################################################

    else:

        axis_ratio = DRP_table['nsa_elpetro_ba'][i_DRP]
        axis_ratio_err = np.NaN

        phi_EofN_deg = DRP_table['nsa_elpetro_phi'][i_DRP]
        phi_EofN_deg_err = np.NaN

        center_x = None
        center_x_err = None

        center_y = None
        center_y_err = None

        ########################################################################
        # Galaxy did not return a successful fit, so we are just going to use 
        # the Ha_vel map mask
        #-----------------------------------------------------------------------
        map_mask = maps['Ha_vel_mask']
        ########################################################################

    z = galaxies_table['nsa_z'][i_gal]

    R90 = galaxies_table['nsa_elpetro_th90'][i_gal] # arcsec
    ########################################################################
    
    
    ########################################################################
    # Extract rotation curve data for the .fits file in question and create 
    # an astropy Table containing said data.
    #-----------------------------------------------------------------------
    start = time()
    
    mass_data_table = calc_mass_curve(sMass_density,
                                      sMass_density_err, 
                                      maps['r_band'], 
                                      map_mask, 
                                      center_x,
                                      center_y,
                                      axis_ratio, 
                                      phi_EofN_deg, 
                                      z, 
                                      gal_ID, 
                                      IMAGE_DIR=IMAGE_DIR, 
                                      IMAGE_FORMAT=IMAGE_FORMAT)
                                                 
    extract_time = time() - start
    
    print(gal_ID, "mass curve calculated", extract_time)
    ########################################################################


    if len(mass_data_table) > 3:
        ####################################################################
        # Fit the stellar mass rotation curve to the disk velocity function.
        #-------------------------------------------------------------------
        start = time()

        param_outputs = fit_mass_curve(mass_data_table, 
                                       gal_ID,
                                       fit_function, 
                                       IMAGE_DIR=IMAGE_DIR,
                                       IMAGE_FORMAT=IMAGE_FORMAT
                                       )

        print(param_outputs)
        fit_time = time() - start

        print(gal_ID, 'mass curve fit', fit_time)
        ####################################################################
        

        ####################################################################
        # Estimate the total disk mass within the galaxy
        #-------------------------------------------------------------------
        if param_outputs is not None:

            ####################################################################
            # Convert R90 from arcsec to kpc
            #-------------------------------------------------------------------
            dist_to_galaxy_Mpc = c*z/H_0
            dist_to_galaxy_kpc = dist_to_galaxy_Mpc*1000

            R90_kpc = dist_to_galaxy_kpc*np.tan(R90*(1./60)*(1./60)*(np.pi/180))
            ####################################################################

            M90_disk, M90_disk_err = disk_mass(param_outputs, R90_kpc)
            M_disk, M_disk_err = disk_mass(param_outputs, 3.5*R90_kpc)
        ####################################################################


        if RUN_ALL_GALAXIES or TEXT_OUT:

            ################################################################
            # Write the extracted mass curve to a text file in ascii format.
            #---------------------------------------------------------------
            mass_data_table.write(MASS_CURVE_MASTER_FOLDER + gal_ID + '.txt', 
                                  format='ascii.commented_header', 
                                  overwrite=True)
            ################################################################


            if param_outputs is not None:
                ############################################################
                # Write the best-fit values and calculated parameters to a 
                # text  file in ascii format.
                #-----------------------------------------------------------
                galaxies_table = fillin_output_table(galaxies_table, 
                                                     param_outputs, 
                                                     i_DRP)
                galaxies_table = fillin_output_table(galaxies_table, 
                                                     M90_disk, 
                                                     i_DRP, 
                                                     col_name='M90_disk')
                galaxies_table = fillin_output_table(galaxies_table, 
                                                     M90_disk_err, 
                                                     i_DRP, 
                                                     col_name='M90_disk_err')
                galaxies_table = fillin_output_table(galaxies_table,
                                                     M_disk,
                                                     i_DRP,
                                                     col_name='M_disk')
                galaxies_table = fillin_output_table(galaxies_table,
                                                     M_disk_err,
                                                     i_DRP,
                                                     col_name='M_disk_err')
                ############################################################

            print(gal_ID, "written")

        else:
            ################################################################
            # Print output to terminal if not analyzing all galaxies
            #---------------------------------------------------------------
            print(param_outputs)

            if param_outputs is not None:
                print('M90_disk:', M90_disk, '+/-', M90_disk_err)
            ################################################################


    print("\n")
################################################################################



################################################################################
# Save the output_table
#-------------------------------------------------------------------------------
if RUN_ALL_GALAXIES or TEXT_OUT:

    #galaxies_filename, extension = GALAXIES_FILENAME.split('.')

    galaxies_table.write('/Users/nityaravi/Documents/Research/RotationCurves/disk_bulge_mass_test.fits', 
                         format='fits', #'ascii.commented_header', 
                         overwrite=True)
################################################################################



################################################################################
# Clock the program's run time to check performance.
#-------------------------------------------------------------------------------
FINISH = time()
print("Runtime:", FINISH - START)
################################################################################






