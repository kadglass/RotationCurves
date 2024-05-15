################################################################################
# IMPORT MODULES
#-------------------------------------------------------------------------------
import datetime
START = datetime.datetime.now()

import os.path, warnings
import sys

from multiprocessing import Process, Queue, Value
from queue import Empty

import numpy as np

from astropy.table import Table
import astropy.units as u

from IO_data import extract_data, add_cols, extract_Pipe3d_data
from elliptical_virial_mass import calculate_virial_mass

# bluehive 
sys.path.insert(1, '/home/nravi3/Documents/RotationCurves/')

# sys.path.insert(1, '/Users/nityaravi/Documents/GitHub/RotationCurves/')
from mapSmoothness_functions import how_smooth

################################################################################
################################################################################
################################################################################

def process_1_galaxy(job_queue, i, 
                     return_queue, 
                     MAP_FOLDER,
                     PIPE3D_FOLDER, 
                     IMAGE_DIR, 
                     IMAGE_FORMAT, 
                     DRP_index, 
                     map_smoothness_min, 
                     DRP_table):
    
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
        
        
       
        
        i_DRP = DRP_index[gal_ID]
        
        ########################################################################
        # Check that ID is a galaxy target
        #-----------------------------------------------------------------------
        if DRP_table['mngtarg1'][i_DRP] > 0:

            ########################################################################
            # Check T-Type
            #-----------------------------------------------------------------------
            if DRP_table['DL_ttype'][i_DRP] >= 0:

                print(gal_ID, ' not an ETG\n')
                output_tuple = (None, None, None, None, None, None)
                return_queue.put(output_tuple)

                continue
            
            ########################################################################
            # Extract the necessary data from the .fits files.
            #-----------------------------------------------------------------------

            maps = extract_data(MAP_FOLDER, gal_ID, ['flux', 'star_sigma', 'Ha'])

            if maps is None:
                print('No data for ', gal_ID, '\n', flush=True)

                output_tuple = (None, None, None, None, None, None)
                return_queue.put(output_tuple)

                
                continue


            pipe3d_maps = extract_Pipe3d_data(PIPE3D_FOLDER,gal_ID,'sMass')

            if pipe3d_maps is None:
                print('No pipe3d data for ', gal_ID, '\n', flush=True)
            
                output_tuple = (None, None, None, None, None, None)
                return_queue.put(output_tuple)

                
                continue

            print( gal_ID, "extracted", flush=True)

            ########################################################################
            # Calculate degree of smoothness of velocity map
            #-----------------------------------------------------------------------

            map_smoothness = how_smooth(maps['Ha_vel'], maps['Ha_vel_mask'])
            map_smoothness_5sigma = how_smooth(maps['Ha_vel'],
                                                np.logical_or(
                                                    maps['Ha_vel_mask'] > 0,
                                                    np.abs(maps['Ha_flux']*np.sqrt(maps['Ha_flux_ivar'])) < 5
                                                    )
                                                    )
            if (map_smoothness < map_smoothness_min) and (map_smoothness_5sigma < map_smoothness_min):
                print(gal_ID, ' map is not smooth\n', flush=True)

                output_tuple = (None, None, None, None, map_smoothness, i_DRP)
                return_queue.put(output_tuple)
                continue

            ########################################################################
            # grab necessary data and calculate virial mass
            #-----------------------------------------------------------------------

            nsa_z = DRP_table['nsa_z'][i_DRP]
            r50 = DRP_table['nsa_elpetro_th50_r'][i_DRP]  

            start = datetime.datetime.now()

            try:

                Mvir, Mvir_err, star_sigma, star_sigma_err = calculate_virial_mass(gal_ID,
                                                                            maps['star_sigma'],
                                                                            maps['star_sigma_ivar'],
                                                                            maps['star_sigma_corr'],
                                                                            maps['star_sigma_mask'],
                                                                            maps['mflux'],
                                                                            pipe3d_maps['sMass_density'],
                                                                            pipe3d_maps['sMass_density_err'],
                                                                            maps['Ha_vel'],
                                                                            maps['Ha_vel_mask'],
                                                                            r50,
                                                                            nsa_z,
                                                                            IMAGE_DIR,
                                                                            IMAGE_FORMAT)

            except:
                print(gal_ID, 'CRASHED! <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<', 
                          flush=True)
                    
                raise
            
            calc_time = datetime.datetime.now() - start

            print(gal_ID, ' calc time: ', calc_time, '\n', flush=True)

        else:
            print(gal_ID, ' is not a galaxy target', flush=True)
            
            Mvir = None
            Mvir_err = None
            star_sigma = None
            star_sigma_err = None
            map_smoothness = None

        print('\n', flush=True)

        ########################################################################
        # Add output values to return queue
        #-----------------------------------------------------------------------
        output_tuple = (Mvir, Mvir_err, star_sigma, star_sigma_err, map_smoothness, i_DRP)
        return_queue.put(output_tuple)


################################################################################
################################################################################################################################################################
################################################################################



IMAGE_FORMAT = 'png'

################################################################################
# Paths for Bluehive
################################################################################

MANGA_FOLDER = '/scratch/kdougla7/data/SDSS/dr17/manga/spectro/'
IMAGE_DIR = '/scratch/nravi3/ellipticals/'
MAP_FOLDER = MANGA_FOLDER + 'analysis/v3_1_1/3.1.0/HYB10-MILESHC-MASTARSSP/'
PIPE3D_FOLDER = MANGA_FOLDER + 'pipe3d/'
DRP_FILENAME = '/scratch/nravi3/ellipticals/Elliptical_StelVelDispDAPMeanSigma_Mvir_smoothness_lt_2.fits'
NSA_FILENAME = '/scratch/kdougla7/data/NSA/nsa_v1_0_1.fits'


################################################################################
################################################################################

map_smoothness_min = 2.0
CLEAR_COLS = True  # zeros out columns in table

START = datetime.datetime.now()


################################################################################
# Open the DRPall file
#-------------------------------------------------------------------------------
DRP_table = Table.read(DRP_FILENAME, format='fits')

DRP_index = {}

# for testing take first 20 galaxies
# for i in range(len(DRP_table)):
for i in range(20):
    gal_ID = DRP_table['plateifu'][i]

    DRP_index[gal_ID] = i
################################################################################

if CLEAR_COLS:
    DRP_table = add_cols(DRP_table, ['smoothness_score', 
                                     'Mvir', 
                                     'Mvir_err',
                                     'star_sigma',
                                     'star_sigma_err'])


N_files = len(DRP_table)
FILE_IDS = list(DRP_index.keys())


job_queue = Queue()
return_queue = Queue()

num_tasks = len(FILE_IDS)

# Load jobs into queue
for i,gal_ID in enumerate(FILE_IDS):
        
    job_queue.put(gal_ID)

    #if i > 10:
    #    num_tasks = 12
    #    break


print('Starting processes', datetime.datetime.now(), flush=True)

processes = []

for i in range(12):

    p = Process(target=process_1_galaxy, args=(job_queue, i, 
                                               return_queue, 
                                               MAP_FOLDER,
                                               PIPE3D_FOLDER, 
                                               IMAGE_DIR, 
                                               IMAGE_FORMAT, 
                                               DRP_index, 
                                               map_smoothness_min, 
                                               DRP_table))
    
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
    Mvir, Mvir_err, star_sigma, star_sigma_err, map_smoothness, i_DRP = return_tuple
    
    #print('Writing', i_DRP, flush=True)

    if map_smoothness is not None:
        DRP_table['smoothness_score'][i_DRP] = map_smoothness
    
    if Mvir is not None:
        DRP_table['Mvir'][i_DRP] = Mvir
        DRP_table['Mvir_err'][i_DRP] = Mvir_err
        DRP_table['star_sigma'][i_DRP] = star_sigma
        DRP_table['star_sigma_err'][i_DRP] = star_sigma_err

    num_processed += 1

    if num_processed % 5 == 0:
        DRP_table.write('/scratch/nravi3/ellipticals/Elliptical_StelVelDispDAPMeanSigma_Mvir_smoothness_lt_2.fits', 
                format='fits', overwrite=True)
        print('Table written ', num_processed, flush=True)
    
    print(num_processed, ': ', i_DRP)

print('Finished populating output table', datetime.datetime.now(), flush=True)
################################################################################


# Go through all the processes and join them back to the parent.
for p in processes:
    p.join(None)


################################################################################
# Save the output_table
#-------------------------------------------------------------------------------
DRP_table.write('/scratch/nravi3/ellipticals/Elliptical_StelVelDispDAPMeanSigma_Mvir_smoothness_lt_2.fits', 
                format='fits', overwrite=True)
################################################################################



################################################################################
# Clock the program's run time to check performance.
#-------------------------------------------------------------------------------
FINISH = datetime.datetime.now()
print("Runtime:", FINISH - START, flush=True)
################################################################################

