################################################################################
# IMPORT MODULES
#-------------------------------------------------------------------------------
import datetime
START = datetime.datetime.now()

import os.path, warnings

import numpy as np

from astropy.table import Table
import astropy.units as u

from multiprocessing import Process, Queue, Value
from queue import Empty

from ctypes import c_long

from file_io import add_columns, fillin_output_table

from DRP_rotation_curve import extract_data

from DRP_vel_map import fit_vel_map, estimate_total_mass

import sys
sys.path.insert(1, '/home/kelly/Documents/RotationCurves/')
#sys.path.insert(1, '/scratch/kdougla7/RotationCurves/')
from mapSmoothness_functions import how_smooth

#warnings.simplefilter('ignore', np.RankWarning)
#warnings.simplefilter('ignore', RuntimeWarning)
################################################################################





################################################################################
################################################################################
################################################################################

def process_1_galaxy(job_queue, i, 
                     return_queue, 
                     num_masked_gal, 
                     num_not_smooth, 
                     num_missing_photo,
                     VEL_MAP_FOLDER, 
                     IMAGE_DIR, 
                     IMAGE_FORMAT, 
                     DRP_index, 
                     map_smoothness_max, 
                     DRP_table, 
                     vel_function, 
                     V_type, 
                     NSA_index, 
                     NSA_table):
    '''
    Main body of for-loop for processing one galaxy.
    '''
    
    ############################################################################
    # Open file to which we can write all the print statements.
    #---------------------------------------------------------------------------
    outfile = open('Process_' + str(i) + '_output.txt', 'wt')
    sys.stdout = outfile
    sys.stderr = outfile
    ############################################################################
    
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
        
        
        ########################################################################
        # Extract the necessary data from the NSA table.
        #-----------------------------------------------------------------------
        i_DRP = DRP_index[gal_ID]
        
        NSA_ID = DRP_table['nsa_nsaid'][i_DRP]
        ########################################################################
        
        
        ########################################################################
        # Confirm that object is a galaxy target
        #-----------------------------------------------------------------------
        if (DRP_table['mngtarg1'][i_DRP] <= 0) or (gal_ID in ['9037-9102']):
            
            print(gal_ID, 'is not a galaxy target.\n', flush=True)
            
            output_tuple = (None, None, None, None, None, None)
            return_queue.put(output_tuple)
            
            continue
        ########################################################################
        
            
        ########################################################################
        # Extract the necessary data from the .fits files.
        #-----------------------------------------------------------------------
        if V_type == 'Ha':
            maps = extract_data(VEL_MAP_FOLDER, 
                                gal_ID, 
                                ['Ha_vel', 'Ha_flux', 'Ha_sigma', 'r_band'])
            
        elif V_type == 'star':
            maps = extract_data(VEL_MAP_FOLDER, 
                                gal_ID, 
                                ['star_vel', 'Ha_flux', 'Ha_sigma', 'r_band'])
                
        else:
            print('Unknown velocity data type (V_type).')
            
            output_tuple = (None, None, None, None, None, None)
            return_queue.put(output_tuple)
            
            continue
            
            
        if maps is None:
        
            print('No data for', gal_ID, '\n', flush=True)
            
            output_tuple = (None, None, None, None, None, None)
            return_queue.put(output_tuple)
            
            continue
        
        
        print( gal_ID, "extracted", flush=True)
        ########################################################################


        ########################################################################
        # Calculate degree of smoothness of velocity map
        #-----------------------------------------------------------------------
        can_fit = True
        
        if V_type == 'Ha':
            map_smoothness = how_smooth( maps['Ha_vel'], maps['Ha_vel_mask'])
            
            map_smoothness_5sigma = how_smooth(maps['Ha_vel'], 
                                               np.logical_or(maps['Ha_vel_mask'] > 0, 
                                                             np.abs(maps['Ha_flux']*np.sqrt(maps['Ha_flux_ivar'])) < 5))
                                                             
            if (map_smoothness > map_smoothness_max) and (map_smoothness_5sigma > map_smoothness_max):
                can_fit = False
        ########################################################################


        if can_fit:
            ####################################################################
            # Extract the necessary data from the DRP table.
            #-------------------------------------------------------------------
            axis_ratio = DRP_table['nsa_elpetro_ba'][i_DRP]
            phi_EofN_deg = DRP_table['nsa_elpetro_phi'][i_DRP]

            z = DRP_table['nsa_z'][i_DRP]
            ####################################################################
            
            
            if axis_ratio > -9999:
                ################################################################
                # Extract rotation curve data for the .fits file in question and 
                # create an astropy Table containing said data.
                #---------------------------------------------------------------
                start = datetime.datetime.now()
                
                try:
                    param_outputs, masked_gal_flag, fit_flag = fit_vel_map(maps[V_type + '_vel'], 
                                                                           maps[V_type + '_vel_ivar'], 
                                                                           maps[V_type + '_vel_mask'], 
                                                                           maps['Ha_sigma'], 
                                                                           maps['Ha_sigma_ivar'], 
                                                                           maps['Ha_sigma_mask'], 
                                                                           maps['Ha_flux'], 
                                                                           maps['Ha_flux_ivar'], 
                                                                           maps['Ha_flux_mask'], 
                                                                           maps['r_band'], 
                                                                           maps['r_band_ivar'], 
                                                                           axis_ratio, 
                                                                           phi_EofN_deg, 
                                                                           z, 
                                                                           gal_ID, 
                                                                           vel_function, 
                                                                           V_type=V_type,
                                                                           IMAGE_DIR=IMAGE_DIR, 
                                                                           IMAGE_FORMAT=IMAGE_FORMAT
                                                                           )
                except:
                    print(gal_ID, 'CRASHED! <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<', 
                          flush=True)
                    
                    raise
                    
                fit_time = datetime.datetime.now() - start

                with num_masked_gal.get_lock():
                    num_masked_gal.value += masked_gal_flag

                print(gal_ID, "velocity map fit", fit_time, flush=True)
                ################################################################


                ################################################################
                # Estimate the total mass within the galaxy
                #---------------------------------------------------------------
                i_NSA = NSA_index[NSA_ID]

                R90 = NSA_table['ELPETRO_TH90_R'][i_NSA]
                
                if param_outputs is not None:
                    mass_outputs = estimate_total_mass([param_outputs['v_max'], 
                                                        param_outputs['r_turn'], 
                                                        param_outputs['alpha']], 
                                                       R90, 
                                                       z, 
                                                       vel_function, 
                                                       gal_ID)
                ################################################################
                
            else:
                print(gal_ID, 'is missing photometric measurements.', 
                      flush=True)
                
                with num_missing_photo.get_lock():
                    num_missing_photo.value += 1
                
                param_outputs = None
                mass_outputs = None
                fit_flag = None
                R90 = None

        else:
            print(gal_ID, "is not smooth enough to fit.", flush=True)

            with num_not_smooth.get_lock():
                num_not_smooth.value += 1

            param_outputs = None
            mass_outputs = None
            fit_flag = None
            
            if NSA_ID >= 0:
                R90 = NSA_table['ELPETRO_TH90_R'][NSA_index[NSA_ID]]
            else:
                R90 = None


        print('\n', flush=True)

        ########################################################################
        # Add output values to return queue
        #-----------------------------------------------------------------------
        output_tuple = (map_smoothness, param_outputs, mass_outputs, fit_flag, R90, i_DRP)
        return_queue.put(output_tuple)
        ########################################################################

        
################################################################################
################################################################################
################################################################################






################################################################################
# File format for saved images
#-------------------------------------------------------------------------------
IMAGE_FORMAT = 'eps'
################################################################################



################################################################################
# Fitting routine restrictions
#-------------------------------------------------------------------------------
# Maximum allowed map smoothness score
map_smoothness_max = 1.85

# Velocity function to use (options are 'BB' or 'tanh')
vel_function = 'BB'

# Velocity map to use
V_type = 'Ha'
#V_type = 'star'
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

IMAGE_DIR = LOCAL_PATH + 'Images/DRP/'

# Create directory if it does not already exist
if not os.path.isdir( IMAGE_DIR):
    os.makedirs( IMAGE_DIR)

MANGA_FOLDER = '/home/kelly/Documents/Data/SDSS/dr16/manga/spectro/'
#MANGA_FOLDER = '/scratch/kdougla7/data/SDSS/dr16/manga/spectro/'
VEL_MAP_FOLDER = MANGA_FOLDER + 'analysis/v2_4_3/2.2.1/HYB10-GAU-MILESHC/'
DRP_FILENAME = MANGA_FOLDER + 'redux/v2_4_3/drpall-v2_4_3.fits'

NSA_FILENAME = '/home/kelly/Documents/Data/NSA/nsa_v1_0_1.fits'
#NSA_FILENAME = '/scratch/kdougla7/data/NSA/nsa_v1_0_1.fits'
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
# Open the NSA file
#-------------------------------------------------------------------------------
NSA_table = Table.read( NSA_FILENAME, format='fits')


NSA_index = {}

for i in range(len(NSA_table)):
    NSA_ID = NSA_table['NSAID'][i]

    NSA_index[NSA_ID] = i
################################################################################



################################################################################
# Create a list of galaxy IDs for which to extract a rotation curve.
#-------------------------------------------------------------------------------
N_files = len(DRP_table)

FILE_IDS = list(DRP_index.keys())

DRP_table = add_columns(DRP_table, vel_function)
################################################################################



################################################################################
# Fit the rotation curve for the H-alpha velocity map for all of the galaxies in 
# the 'files' array.
#-------------------------------------------------------------------------------
#num_masked_gal = 0 # Number of completely masked galaxies
#num_not_smooth = 0 # Number of galaxies which do not have smooth velocity maps

num_masked_gal = Value(c_long)
num_not_smooth = Value(c_long)
num_missing_photo = Value(c_long)

with num_masked_gal.get_lock():
    num_masked_gal.value = 0

with num_not_smooth.get_lock():
    num_not_smooth.value = 0
    
with num_missing_photo.get_lock():
    num_missing_photo.value = 0


job_queue = Queue()
return_queue = Queue()

num_tasks = len(FILE_IDS)

# Load jobs into queue
for i,gal_ID in enumerate(FILE_IDS):
        
    job_queue.put(gal_ID)
    '''
    if i > 10:
        num_tasks = 12
        break
    '''
    


print('Starting processes', datetime.datetime.now(), flush=True)

processes = []

for i in range(12): # This number is the number of processes

    p = Process(target=process_1_galaxy, args=(job_queue, i, 
                                               return_queue, 
                                               num_masked_gal, 
                                               num_not_smooth, 
                                               num_missing_photo,
                                               VEL_MAP_FOLDER, 
                                               IMAGE_DIR, 
                                               IMAGE_FORMAT, 
                                               DRP_index, 
                                               map_smoothness_max, 
                                               DRP_table, 
                                               vel_function, 
                                               V_type, 
                                               NSA_index, 
                                               NSA_table))
    
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

    ############################################################################
    # Write the best-fit values and calculated parameters to a text file in 
    # ascii format.
    #---------------------------------------------------------------------------
    map_smoothness, param_outputs, mass_outputs, fit_flag, R90, i_DRP = return_tuple
    
    #print('Writing', i_DRP, flush=True)

    if map_smoothness is not None:
        DRP_table = fillin_output_table(DRP_table, 
                                        map_smoothness, 
                                        i_DRP, 
                                        col_name='smoothness_score')
    
    if R90 is not None:
        DRP_table = fillin_output_table(DRP_table, 
                                        R90, 
                                        i_DRP, 
                                        col_name='nsa_elpetro_th90')

    if param_outputs is not None:
        DRP_table = fillin_output_table(DRP_table, param_outputs, i_DRP)
        DRP_table = fillin_output_table(DRP_table, mass_outputs, i_DRP)
        DRP_table = fillin_output_table(DRP_table, 
                                        fit_flag, 
                                        i_DRP, 
                                        col_name='fit_flag')
    ############################################################################
    
    num_processed += 1
    
    print(num_processed)

    #print("\n")
    
#job_queue.close()
#job_queue.join_thread()
    
print('Finished populating output table', datetime.datetime.now(), flush=True)
################################################################################


# Go through all the processes and join them back to the parent.
for p in processes:
    p.join(None)


################################################################################
# Save the output_table
#-------------------------------------------------------------------------------
DRP_table.write('DRP_vel_map_results_' + vel_function + '_smooth_lt_' + str(map_smoothness_max) + '.fits', 
                format='fits', overwrite=True)
################################################################################


################################################################################
# Print number of galaxies that were completely masked
#-------------------------------------------------------------------------------
print('There were', num_masked_gal.value, 'galaxies that were completely masked.', flush=True)
print('There were', num_not_smooth.value, 'galaxies without smooth velocity maps.', flush=True)
print('There were', num_missing_photo.value, 'galaxies missing photometry.', flush=True)
################################################################################


################################################################################
# Clock the program's run time to check performance.
#-------------------------------------------------------------------------------
FINISH = datetime.datetime.now()
print("Runtime:", FINISH - START, flush=True)
################################################################################

