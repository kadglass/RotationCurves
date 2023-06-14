################################################################################
# IMPORT MODULES
#-------------------------------------------------------------------------------
import datetime
START = datetime.datetime.now()

import os.path, warnings

import numpy as np

from astropy.table import Table
import astropy.units as u
import astropy.constants as const

from multiprocessing import Process, Queue, Value
from queue import Empty

from ctypes import c_long

from file_io import add_star_columns, fillin_output_table

from DRP_rotation_curve import extract_data

from Pipe3D_starVel_map import extract_Pipe3D_data

from DRP_vel_map import fit_vel_map, estimate_total_mass

from dark_matter_mass_v1 import rot_fit_BB

import sys

#warnings.simplefilter('ignore', np.RankWarning)
#warnings.simplefilter('ignore', RuntimeWarning)
################################################################################





################################################################################
################################################################################
################################################################################

def process_1_galaxy(job_queue, i, 
                     return_queue, 
                     num_masked_gal, 
                     num_missing_photo,
                     DRP_FOLDER, 
                     PIPE3D_FOLDER, 
                     IMAGE_DIR, 
                     IMAGE_FORMAT, 
                     DRP_index, 
                     DRP_table, 
                     vel_function):
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
        
        #NSA_ID = DRP_table['nsa_nsaid'][i_DRP]
        ########################################################################
        
        """
        ########################################################################
        # Confirm that object is a galaxy target
        #-----------------------------------------------------------------------
        if DRP_table['mngtarg1'][i_DRP] <= 0:
            
            print(gal_ID, 'is not a galaxy target.\n', flush=True)
            
            output_tuple = (None, None, None, None, None)
            return_queue.put(output_tuple)
            
            continue
        ########################################################################
        """
            
        ########################################################################
        # Extract the necessary data from the .fits files.
        #-----------------------------------------------------------------------
        _, _, _, r_band, r_band_ivar, Ha_flux, Ha_flux_ivar, Ha_flux_mask, Ha_sigma, Ha_sigma_ivar, Ha_sigma_mask = extract_data(DRP_FOLDER, gal_ID)
        
        star_vel, star_vel_err = extract_Pipe3D_data(PIPE3D_FOLDER, gal_ID)
        """
        if Ha_vel is None:
        
            print('No data for', gal_ID, '\n', flush=True)
            
            output_tuple = (None, None, None, None, None)
            return_queue.put(output_tuple)
            
            continue
        """
        print( gal_ID, "extracted", flush=True)
        ########################################################################
        
        
        ########################################################################
        # Construct mask for the stellar velocity map
        #
        # Following Aquino-Ortiz et al. (2020), we require a maximum error of 
        # 25 km/s in the derived velocities.
        #-----------------------------------------------------------------------
        star_vel_mask = (star_vel_err > 25) | np.isnan(star_vel) | (star_vel == 0)
        ########################################################################


        ########################################################################
        # Extract the necessary data from the DRP table.
        #-----------------------------------------------------------------------
        axis_ratio = DRP_table['NSA_ba'][i_DRP]
        phi_EofN_deg = DRP_table['NSA_phi'][i_DRP]

        z = DRP_table['NSA_redshift'][i_DRP]
        ########################################################################
        
        
        if axis_ratio > -9999:
            ####################################################################
            # Subtract systemic velocity from stellar velocity map
            #-------------------------------------------------------------------
            star_vel -= (z*const.c.to('km/s')).value
            ####################################################################
            
            
            ####################################################################
            # Extract rotation curve data for the .fits file in question and 
            # create an astropy Table containing said data.
            #-------------------------------------------------------------------
            start = datetime.datetime.now()
            
            try:
                param_outputs, masked_gal_flag, fit_flag = fit_vel_map(star_vel, 
                                                                       1/star_vel_err**2, 
                                                                       star_vel_mask, 
                                                                       Ha_sigma, 
                                                                       Ha_sigma_ivar, 
                                                                       Ha_sigma_mask, 
                                                                       Ha_flux, 
                                                                       Ha_flux_ivar, 
                                                                       Ha_flux_mask, 
                                                                       r_band, 
                                                                       r_band_ivar, 
                                                                       axis_ratio, 
                                                                       phi_EofN_deg, 
                                                                       z, 
                                                                       gal_ID, 
                                                                       vel_function, 
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
            ####################################################################


            ####################################################################
            # Estimate the total mass within the galaxy
            #-------------------------------------------------------------------
            R90 = DRP_table['NSA_elpetro_th90'][i_DRP]
            
            if param_outputs is not None:
                mass_outputs = estimate_total_mass([param_outputs['v_max'], 
                                                    param_outputs['r_turn'], 
                                                    param_outputs['alpha']], 
                                                   R90, 
                                                   z, 
                                                   vel_function, 
                                                   gal_ID)
            else:
                mass_outputs = None
            ####################################################################
            
        else:
            print(gal_ID, 'is missing photometric measurements.', 
                  flush=True)
            
            with num_missing_photo.get_lock():
                num_missing_photo.value += 1
            
            param_outputs = None
            mass_outputs = None


        print('\n', flush=True)

        ########################################################################
        # Add output values to return queue
        #-----------------------------------------------------------------------
        output_tuple = (param_outputs, mass_outputs, i_DRP)
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
# Velocity function to use (options are 'BB' or 'tanh')
vel_function = 'BB'
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

IMAGE_DIR = LOCAL_PATH + 'Images/Pipe3D_star/'

# Create directory if it does not already exist
if not os.path.isdir( IMAGE_DIR):
    os.makedirs( IMAGE_DIR)

MANGA_FOLDER = '/home/kelly/Documents/Data/SDSS/'

PIPE3D_FOLDER = MANGA_FOLDER + 'dr15/manga/spectro/pipe3d/v2_4_3/2.4.3/'
DRP_FOLDER = MANGA_FOLDER + 'dr16/manga/spectro/analysis/v2_4_3/2.2.1/HYB10-GAU-MILESHC/'
#DRP_FILENAME = MANGA_FOLDER + 'redux/v2_4_3/drpall-v2_4_3.fits'
FITS_FILENAME = 'DRP-master_file_vflag_BB_smooth1p85_mapFit_N2O2_HIdr2_morph_v6.txt'

#NSA_FILENAME = '/home/kelly/Documents/Data/NSA/nsa_v1_0_1.fits'
################################################################################



################################################################################
# Open the fits file
#-------------------------------------------------------------------------------
fits = Table.read(FITS_FILENAME, format='ascii.commented_header')

fits_index = {}

for i in range(len(fits)):
    gal_ID = str(fits['MaNGA_plate'][i]) + '-' + str(fits['MaNGA_IFU'][i])
    
    fits_index[gal_ID] = i
################################################################################


"""
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
"""


################################################################################
# Create a list of galaxy IDs for which to extract a rotation curve.
#-------------------------------------------------------------------------------
# Only fit those which have good fits
#-------------------------------------------------------------------------------
# Calculate the velocity at R90
#-------------------------------------------------------------------------------
# Convert r from arcsec to kpc
#-------------------------------------------------------------------------------
H0 = 100 # Hubble's constant in units of h km/s/Mpc

dist_to_galaxy_Mpc = (fits['NSA_redshift']*const.c.to('km/s')/H0).value
dist_to_galaxy_kpc = dist_to_galaxy_Mpc*1000

fits['R90_kpc'] = dist_to_galaxy_kpc*np.tan(fits['NSA_elpetro_th90']*(1./60)*(1./60)*(np.pi/180))
#-------------------------------------------------------------------------------

fits['V90_kms'] = rot_fit_BB(fits['R90_kpc'], 
                             [fits['Vmax_map'], 
                              fits['Rturn_map'], 
                              fits['alpha_map']])
#-------------------------------------------------------------------------------
# Calculate the mass ratio
#-------------------------------------------------------------------------------
fits['M90_Mdisk_ratio'] = 10**(fits['M90_map'] - fits['M90_disk_map'])
#-------------------------------------------------------------------------------

bad_boolean = np.logical_or.reduce([np.isnan(fits['M90_map']), 
                                    np.isnan(fits['M90_disk_map']), 
                                    fits['alpha_map'] > 99, 
                                    fits['ba_map'] > 0.998, 
                                    fits['V90_kms']/fits['Vmax_map'] < 0.9, 
                                    (fits['Tidal'] & (fits['DL_merge'] > 0.97)), 
                                    fits['map_frac_unmasked'] < 0.05, 
                                    (fits['map_frac_unmasked'] > 0.13) & (fits['DRP_map_smoothness'] > 1.96), 
                                    (fits['map_frac_unmasked'] > 0.07) & (fits['DRP_map_smoothness'] > 2.9), 
                                    (fits['map_frac_unmasked'] > -0.0638*fits['DRP_map_smoothness'] + 0.255) & (fits['DRP_map_smoothness'] > 1.96), 
                                    fits['M90_Mdisk_ratio'] > 1050])

N_files = len(fits) - np.sum(bad_boolean)

FILE_IDS = []

for i in range(len(fits)):
    
    if not bad_boolean[i]:
    
        FILE_IDS.append(str(fits['MaNGA_plate'][i]) + '-' + str(fits['MaNGA_IFU'][i]))

fits = add_star_columns(fits, vel_function)
################################################################################



################################################################################
# Fit the rotation curve for the H-alpha velocity map for all of the galaxies in 
# the 'files' array.
#-------------------------------------------------------------------------------
#num_masked_gal = 0 # Number of completely masked galaxies

num_masked_gal = Value(c_long)
num_missing_photo = Value(c_long)

with num_masked_gal.get_lock():
    num_masked_gal.value = 0
    
with num_missing_photo.get_lock():
    num_missing_photo.value = 0


job_queue = Queue()
return_queue = Queue()

num_tasks = len(FILE_IDS)

# Load jobs into queue
for i,gal_ID in enumerate(FILE_IDS):
        
    job_queue.put(gal_ID)


print('Starting processes', datetime.datetime.now(), flush=True)

processes = []

for i in range(12):

    p = Process(target=process_1_galaxy, args=(job_queue, i, 
                                               return_queue, 
                                               num_masked_gal, 
                                               num_missing_photo,
                                               DRP_FOLDER, 
                                               PIPE3D_FOLDER, 
                                               IMAGE_DIR, 
                                               IMAGE_FORMAT, 
                                               fits_index, 
                                               fits, 
                                               vel_function))
    
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
    param_outputs, mass_outputs, i_fits = return_tuple
    
    #print('Writing', i_DRP, flush=True)

    if param_outputs is not None:
        fits = fillin_output_table(fits, 
                                   param_outputs, 
                                   i_fits, 
                                   col_suffix='_star')
        fits = fillin_output_table(fits, 
                                   mass_outputs, 
                                   i_fits, 
                                   col_suffix='_star')
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
fits.write('Pipe3D_starVel_map_results_' + vel_function + '.fits', 
           format='fits', overwrite=True)
################################################################################


################################################################################
# Print number of galaxies that were completely masked
#-------------------------------------------------------------------------------
print('There were', num_masked_gal.value, 'galaxies that were completely masked.', flush=True)
print('There were', num_missing_photo.value, 'galaxies missing photometry.', flush=True)
################################################################################


################################################################################
# Clock the program's run time to check performance.
#-------------------------------------------------------------------------------
FINISH = datetime.datetime.now()
print("Runtime:", FINISH - START, flush=True)
################################################################################

