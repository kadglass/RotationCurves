################################################################################
# IMPORT MODULES
#-------------------------------------------------------------------------------
from time import time
START = time()

import os

from multiprocessing import Process, Queue, Value
from queue import Empty

import numpy as np
import numpy.ma as ma

from astropy.table import Table

from file_io import add_disk_columns, fillin_output_table

from DRP_rotation_curve import extract_data, extract_Pipe3d_data

from DRP_vel_map_functions import build_map_mask

from disk_mass import calc_mass_curve, fit_mass_curve

#import matplotlib.pyplot as plt
import sys
#sys.path.insert(1, '/home/kelly/Documents/RotationCurves/')
sys.path.insert(1, '/scratch/nravi3/RotationCurves/')
from rotation_curve_functions import disk_mass



################################################################################
################################################################################
################################################################################

def process_1_galaxy(job_queue, i,
                     return_queue,
                     VEL_MAP_FOLDER,
                     MASS_MAP_FOLDER,
                     IMAGE_DIR,
                     IMAGE_FORMAT,
                     DRP_index,
                     DRP_table,
                     fit_function,
                     galaxies_index,
                     galaxies_table)

    outfile = open('Process_' + str(i) + '_output.txt', 'wt')
    sys.stdout = outfile
    sys.stderr = outfile

    while True:
        try: 
            gal_ID = job_queue.get(timeout=1.0)
        except Empty:
        
            print('Queue is empty!', flush=True)
        
            outfile.close()
            sys.stdout = sys.__stdout__
            sys.stderr = sys.__stderr__
            
            print('Worker', i, 'redirected stdout and stderr.', flush=True)
            
            return_queue.close()
            
            print('Worker', i, 'closed the return queue.', flush=True)
            
            return_queue.join_thread()
            
            print('Worker', i, 'joined the return queue.', flush=True)
            
            job_queue.close()
            
            print('Worker', i, 'closed the job queue.', flush=True)
            
            job_queue.join_thread()
            
            print('Worker', i, 'returned successfully', datetime.datetime.now(), flush=True)
            return

        maps = extract_data(VEL_MAP_FOLDER, 
                        gal_ID, 
                        ['Ha_vel', 'r_band', 'Ha_flux', 'Ha_sigma'])
        sMass_density, sMass_density_err = extract_Pipe3d_data(MASS_MAP_FOLDER, gal_ID)

        if maps is None or sMass_density is None:
            print('\n', flush=True)
            continue

        print( gal_ID, "extracted", flush=True)
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

        try:
    
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

        except:
            print(gal_ID, 'CRASHED! <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<', flush=True)
                    
            raise
                                                 
        extract_time = time() - start
    
        print(gal_ID, "mass curve calculated", extract_time, flush=True)
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

            print(param_outputs, flush=True)
            fit_time = time() - start

            print(gal_ID, 'mass curve fit', fit_time, flush=True)
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
        else:
            M90_disk = None
            M90_disk_err = None
            M_disk = None 
            M_disk_err = None


        ################################################################
         # Write the extracted mass curve to a text file in ascii format.
        #---------------------------------------------------------------
        mass_data_table.write(MASS_CURVE_MASTER_FOLDER + gal_ID + '.txt', 
                              format='ascii.commented_header', 
                              overwrite=True)
        ################################################################


        print('\n', flush=True)

        ########################################################################
        # Add output values to return queue
        #-----------------------------------------------------------------------
        output_tuple = (param_outputs, M90_disk, M90_disk_err, M_disk, M_disk_err, i_DRP)
        return_queue.put(output_tuple)
        ########################################################################




################################################################################
################################################################################
################################################################################





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


FILE_IDS = ['8082-12702']



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


IMAGE_DIR = '/scratch/nravi3/Images/Pipe3d/'

if not os.path.isdir( IMAGE_DIR):
    os.makedirs( IMAGE_DIR)


# for bluehive
MANGA_FOLDER = '/scratch/kdougla7/data/SDSS/dr17/manga/spectro/'
NSA_FILENAME = '/scratch/kdougla7/data/NSA/nsa_v1_0_1.fits'
MASS_MAP_FOLDER = MANGA_FOLDER + 'pipe3d/'
VEL_MAP_FOLDER = MANGA_FOLDER + 'analysis/v3_1_1/3.1.0/HYB10-MILESHC-MASTARSSP/'

MASS_CURVE_MASTER_FOLDER  = '/scratch/nravi3/Pipe3d-mass_curve_data_files/'
if not os.path.isdir(MASS_CURVE_MASTER_FOLDER):
    os.makedirs(MASS_CURVE_MASTER_FOLDER)

GALAXIES_FILENAME = '/scratch/nravi3/DRP_HaVel_map_results_BB_smooth_lt_2.0_.fits'
DRP_FILENAME = MANGA_FOLDER + 'redux/v3_1_1/drpall-v3_1_1.fits'
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

# for test purposes, only take first 5 galaxies
galaxies_table = galaxies_table[0:5]


galaxies_index = {}

for i in range(len(galaxies_table)):
    gal_ID = galaxies_table['plateifu'][i]

    galaxies_index[gal_ID] = i
################################################################################



################################################################################
# Create a list of galaxy IDs for which to fit the mass rotation curve.
#-------------------------------------------------------------------------------
N_files = len(galaxies_table)

FILE_IDS = list(galaxies_index.keys())

galaxies_table = add_disk_columns(galaxies_table)


job_queue = Queue()
return_queue = Queue()

num_tasks = len(FILE_IDS)

# Load jobs into queue
for i,gal_ID in enumerate(FILE_IDS):
        
    job_queue.put(gal_ID)

    if i > 10:
        num_tasks = 12
        break


print('Starting processes', datetime.datetime.now(), flush=True)

processes = []

for i in range(2):

    p = Process(target=process_1_galaxy, args=(job_queue, i, 
                                               return_queue, 
                                               VEL_MAP_FOLDER,
                                               MASS_MAP_FOLDER, 
                                               IMAGE_DIR, 
                                               IMAGE_FORMAT, 
                                               DRP_index, 
                                               DRP_table, 
                                               vel_function, 
                                               fit_function,
                                               galaxies_index,
                                               galaxies_table))

    
    p.start()

    processes.append(p)

print('Populating output table', datetime.datetime.now(), flush=True)




################################################################################
# Iterate through the populated return queue to fill in the table
#-------------------------------------------------------------------------------
num_processed = 0

print(num_tasks)

while num_processed < num_tasks:

    try:
        return_tuple = return_queue.get(timeout=1.0)
    except:
        continue

    param_outputs, M90_disk, M90_disk_err, i_DRP = return_tuple

    if param_outputs is not None:
        galaxies_table = fillin_output_table(galaxies_table, 
                                                     param_outputs, 
                                                     i_DRP)

    if M90_disk is not None:
        galaxies_table = fillin_output_table(galaxies_table, 
                                                     M90_disk, 
                                                     i_DRP, 
                                                     col_name='M90_disk')

                        
    if M90_disk_err is not None:
        galaxies_table = fillin_output_table(galaxies_table, 
                                                     M90_disk_err, 
                                                     i_DRP, 
                                                     col_name='M90_disk_err')

    if M_disk is not None:
        galaxies_table = fillin_output_table(galaxies_table, 
                                                     M_disk, 
                                                     i_DRP, 
                                                     col_name='M_disk')

    if M_disk_err is not None:
        galaxies_table = fillin_output_table(galaxies_table, 
                                                     M_disk_err, 
                                                     i_DRP, 
                                                     col_name='M_disk_err')

    num_processed += 1
    
    print(num_processed)

print('Finished populating output table', datetime.datetime.now(), flush=True)


for p in processes:
    p.join(None)

################################################################################
# Save the output_table
#-------------------------------------------------------------------------------
galaxies_table.write('/scratch/nravi3/Pipe3d_disk_bulge' + '.fits', 
                format='fits', overwrite=True)
################################################################################

################################################################################
# Clock the program's run time to check performance.
#-------------------------------------------------------------------------------
FINISH = datetime.datetime.now()
print("Runtime:", FINISH - START, flush=True)
################################################################################