# IMPORT MODULES
#-------------------------------------------------------------------------------
import datetime
START = datetime.datetime.now()

import os.path, warnings

import numpy as np

from astropy.table import Table
import astropy.units as u

from file_io import add_columns, fillin_output_table

from DRP_rotation_curve import extract_data

from DRP_vel_map import fit_vel_map, estimate_total_mass

import sys
#sys.path.insert(1, '/Users/kellydouglass/Documents/Research/Rotation_curves/RotationCurves/')
sys.path.insert(1, '/Users/nityaravi/Documents/GitHub/RotationCurves/')
#sys.path.insert(1, '/home/kelly/Documents/RotationCurves/')
from mapSmoothness_functions import how_smooth

warnings.simplefilter('ignore', np.RankWarning)
warnings.simplefilter('ignore', RuntimeWarning)
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
# map_smoothness_max = 1.85
map_smoothness_max = 2.0 # changed for DR17


# Velocity function to use (options are 'BB' or 'tanh')
#vel_function = 'BB'
#vel_function = 'tanh'
#vel_function = 'tail' 

# Velocity map to use
V_type = 'Ha'
#V_type = 'star'

#nominal disk thickness
q0 = 0.2

# require HI_vel
HI_vel_req = False
################################################################################



################################################################################
# List of files (in "[MaNGA_plate]-[MaNGA_IFU]" format) to be ran through the
# individual galaxy version of this script.
#
# If RUN_ALL_GALAXIES is set to True, then code will ignore what is in FILE_IDS.
#-------------------------------------------------------------------------------
'''
fixed = ['11949-12702',
        '10845-6101',
        '11009-1902',
        '8950-12705',
        '11009-3703',
        '11939-3701',
        '8949-12703',
        '9037-9102',
        ]
'''

'''

FILE_IDS = ['10838-12705', # something isnt masked
'11759-1902', # bad ba
'11835-6104', # bad ba
'11867-9101', # bad ba
'12495-12704', # bad ba
'12651-3701', #bad ba
'8138-12702', # bad ba
'8255-1901', # no map?? - FLOP
'8565-12705', # bad ba
'8626-12702', #bad ba
'8719-9102', # bad ba (maybe phi)
'8942-3704', # bad ba
'8987-3704', # bad ba
'9042-6102', # bad ba
'9046-3704', # bad ba
'9512-12701', # bad ba
'11939-3701', # kelly's w HI
'8949-12703', #kelly's w HI
'11009-1902'] #kelly's w HI'''

FILE_IDS = ['8997-9102']


RUN_ALL_GALAXIES = False

TEXT_OUT = True # true for text file output of fits
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
    IMAGE_DIR = LOCAL_PATH + 'Images/DRP/'

    # Create directory if it does not already exist
    if not os.path.isdir( IMAGE_DIR):
        os.makedirs( IMAGE_DIR)
else:
    #IMAGE_DIR = None
    #IMAGE_DIR = LOCAL_PATH + 'Images/DRP/'
    IMAGE_DIR = '/Users/nityaravi/Documents/Research/RotationCurves/data/manga/PAPER_PLOTS/'


# for bluehive
# MANGA_FOLDER = '/scratch/kdougla7/data/SDSS/dr17/manga/spectro/'
# NSA_FILENAME = '/scratch/kdougla7/data/NSA/nsa_v1_0_1.fits'
# VEL_MAP_FOLDER = MANGA_FOLDER + 'analysis/v3_1_1/3.1.0/HYB10-MILESHC-MASTARSSP/'
# DRP_FILENAME = MANGA_FOLDER + 'redux/v3_1_1/v3_1_1.fits'

# old
# MANGA_FOLDER = '/Users/kellydouglass/Documents/Research/data/SDSS/dr16/manga/spectro/'
# NSA_FILENAME = '/Users/kellydouglass/Documents/Drexel/Research/Data/NSA/nsa_v1_0_1.fits'


# nitya's local machine
MANGA_FOLDER = '/Users/nityaravi/Documents/Research/RotationCurves/data/manga/'
NSA_FILENAME = '/Users/nityaravi/Documents/Research/RotationCurves/data/nsa_v1_0_1.fits'
VEL_MAP_FOLDER = MANGA_FOLDER + 'DR17/'
#DRP_FILENAME = MANGA_FOLDER + 'DR17/' + 'drpall-v3_1_1.fits'
#DRP_FILENAME = MANGA_FOLDER + '/output_files/DR17/disk_masses_HIdr3_err_morph_v2.fits'
#DRP_FILENAME = MANGA_FOLDER + '/output_files/DR17/CURRENT_MASTER_TABLE/refit.fits'
DRP_FILENAME = MANGA_FOLDER + '/output_files/DR17/CURRENT_MASTER_TABLE/H_alpha_HIvel_BB_extinction_H2_MxCG_R90_v3p5_Z_SFR_Portsmouthflux_Zglob.fits'


# old
# MANGA_FOLDER = '/home/kelly/Documents/Data/SDSS/dr16/manga/spectro/'
# NSA_FILENAME = '/home/kelly/Documents/Data/NSA/nsa_v1_0_1.fits'
# VEL_MAP_FOLDER = MANGA_FOLDER + 'analysis/v2_4_3/2.2.1/HYB10-GAU-MILESHC/'
# DRP_FILENAME = MANGA_FOLDER + 'redux/v2_4_3/drpall-v2_4_3.fits'



# for dr15:
# VEL_MAP_FOLDER = MANGA_FOLDER + 'DR15/'
# DRP_FILENAME = MANGA_FOLDER + 'drpall-v2_4_3.fits'





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
if RUN_ALL_GALAXIES:

    N_files = len(DRP_table)

    FILE_IDS = list(DRP_index.keys())

    DRP_table = add_columns(DRP_table)

else:

    N_files = len(FILE_IDS)
################################################################################



################################################################################
# Fit the rotation curve for the H-alpha velocity map for all of the galaxies in
# the 'files' array.
#-------------------------------------------------------------------------------
num_masked_gal = 0 # Number of completely masked galaxies
num_not_smooth = 0 # Number of galaxies which do not have smooth velocity maps

for gal_ID in FILE_IDS:

    i_DRP = DRP_index[gal_ID]

    #if DRP_table['mngtarg1'][i_DRP] > 0 or DRP_table['mngtarg3'][i_DRP] > 0:
    if DRP_table['mngtarg1'][i_DRP] > 0:

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
            continue

        if maps is None:
            print('\n')
            continue

        print( gal_ID, "extracted")
        ########################################################################


        ########################################################################
        # Calculate degree of smoothness of velocity map
        #-----------------------------------------------------------------------
        can_fit = True


        if V_type == 'Ha':
            map_smoothness = how_smooth(maps['Ha_vel'], maps['Ha_vel_mask'])
            map_smoothness_5sigma = how_smooth(maps['Ha_vel'],
                                               np.logical_or(maps['Ha_vel_mask'] > 0,
                                                             np.abs(maps['Ha_flux']*np.sqrt(maps['Ha_flux_ivar'])) < 5))
            if (map_smoothness > map_smoothness_max) and (map_smoothness_5sigma > map_smoothness_max):
                can_fit = False

                if can_fit == False:
                    print("Galaxy's map is not smooth enough to fit.")
                    num_not_smooth += 1
        ########################################################################



        ########################################################################
        # Check if galaxy is late type
        #-----------------------------------------------------------------------
        if DRP_table['DL_ttype'][i_DRP] <= 0:
            can_fit = False

            print('Not a late type galaxy.')
            print('T-Type: ' , DRP_table['DL_ttype'][i_DRP])
        ########################################################################
        
        
        
        ########################################################################
        # Check if galaxy has HI velocity
        #-----------------------------------------------------------------------
        HI_vel = DRP_table['WF50'][i_DRP]
        HI_vel_err = DRP_table['WF50_err'][i_DRP]

        if HI_vel < 0:
            HI_vel = None
            HI_vel_err = None

        if HI_vel_req == True:

            if HI_vel == None:
                can_fit = False
                print('Galaxy does not have HI velocity data.')

        HI_vel = None
        HI_vel_err = None
        ########################################################################



        if can_fit:
            ####################################################################
            # Extract the necessary data from the DRP table.
            #-------------------------------------------------------------------
            axis_ratio = DRP_table['nsa_elpetro_ba'][i_DRP]
            phi_EofN_deg = DRP_table['nsa_elpetro_phi'][i_DRP]

            z = DRP_table['nsa_z'][i_DRP]
            NSA_ID = DRP_table['nsa_nsaid'][i_DRP]

            ####################################################################


            ####################################################################
            # If the smoothness score exceeds the max, use 5sigma mask as the 
            # minimum mask
            #-------------------------------------------------------------------
            #mask_5sigma = False
            #if map_smoothness > map_smoothness_max:
            #    mask_5sigma = True


            if axis_ratio > -9999:
                ################################################################
                # Extract rotation curve data for the .fits file in question and
                # create an astropy Table containing said data.
                #---------------------------------------------------------------
                start = datetime.datetime.now()

                ################################################################
                # Set fit function using axis ratio
                #---------------------------------------------------------------
                cosi2 = (axis_ratio**2 - q0**2)/(1 - q0**2)
                
                if cosi2 < (np.cos(85*np.pi/180))**2:
                    vel_function = 'BB'
                else:
                    vel_function = 'tail'
                
                vel_function = 'BB'

                print('Fit function: ', vel_function)



                ################################################################
                # Extract the necessary data from the NSA table.
                #---------------------------------------------------------------

                

                i_NSA = NSA_index[NSA_ID] 
                R90 = NSA_table['ELPETRO_TH90_R'][i_NSA]
                ################################################################

                param_outputs, num_masked_gal, fit_flag = fit_vel_map(maps[V_type + '_vel'],
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
                                                                      HI_vel,
                                                                      HI_vel_err,
                                                                      R90,
                                                                      #mask_5sigma,
                                                                      gal_ID,
                                                                      vel_function,
                                                                      V_type=V_type,
                                                                      IMAGE_DIR=IMAGE_DIR,
                                                                      IMAGE_FORMAT=IMAGE_FORMAT,
                                                                      num_masked_gal=num_masked_gal)


                fit_time = datetime.datetime.now() - start
                print(gal_ID, "velocity map fit", fit_time)
                ################################################################

                
                if param_outputs is not None:
                    ############################################################
                    # Estimate the total mass within the galaxy
                    #-----------------------------------------------------------

                    if vel_function == 'BB':
                        M_R90, M_R90_err = estimate_total_mass([param_outputs['v_max'],
                                                        param_outputs['r_turn'],
                                                        param_outputs['alpha']],
                                                       R90, 
                                                       z,
                                                       vel_function,
                                                       gal_ID)
                    
                        M, M_err = estimate_total_mass([param_outputs['v_max'],
                                                        param_outputs['r_turn'],
                                                        param_outputs['alpha']],
                                                       3.5*R90, 
                                                       z,
                                                       vel_function,
                                                       gal_ID)

                    else:
                        M_R90, M_R90_err = estimate_total_mass([param_outputs['v_max'],
                                                        param_outputs['r_turn'],
                                                        param_outputs['alpha'],
                                                        param_outputs['b']],
                                                       R90, 
                                                       z,
                                                       vel_function,
                                                       gal_ID)
                    
                        M, M_err = estimate_total_mass([param_outputs['v_max'],
                                                        param_outputs['r_turn'],
                                                        param_outputs['alpha'],
                                                        param_outputs['b']],
                                                       3.5*R90, 
                                                       z,
                                                       vel_function,
                                                       gal_ID)
                    ############################################################


                if RUN_ALL_GALAXIES or TEXT_OUT:
                    ############################################################
                    # Write the best-fit values and calculated parameters to a
                    # text file in ascii format.
                    #-----------------------------------------------------------
                    if V_type == 'Ha':
                        DRP_table = fillin_output_table(DRP_table,
                                                        map_smoothness,
                                                        i_DRP,
                                                        col_name='smoothness_score')

                    DRP_table = fillin_output_table(DRP_table,
                                                    R90,
                                                    i_DRP,
                                                    col_name='nsa_elpetro_th90')
                    

                    if param_outputs is not None:
                        DRP_table = fillin_output_table(DRP_table,
                                                        param_outputs,
                                                        i_DRP)
                        DRP_table = fillin_output_table(DRP_table,
                                                        M_R90,
                                                        i_DRP,
                                                        col_name='M_R90')
                        DRP_table = fillin_output_table(DRP_table,
                                                        M_R90_err,
                                                        i_DRP,
                                                        col_name='M_R90_err')
                        DRP_table = fillin_output_table(DRP_table,
                                                        M,
                                                        i_DRP,
                                                        col_name='M')
                        DRP_table = fillin_output_table(DRP_table,
                                                        M_err,
                                                        i_DRP,
                                                        col_name='M_err')                                
                        DRP_table = fillin_output_table(DRP_table,
                                                        fit_flag,
                                                        i_DRP,
                                                        col_name='fit_flag')
                    
                    if vel_function == 'BB':
                        fit_type=0
                    if vel_function == 'tail':
                        fit_type=1
                        

                    DRP_table = fillin_output_table(DRP_table,
                                                    fit_type,
                                                    i_DRP,
                                                    col_name='fit_function')
                    print(gal_ID, "written")
                    ############################################################

                #else:
                    ############################################################
                    # Print output to terminal if not analyzing all galaxies
                    #-----------------------------------------------------------
                    print(DRP_table[['plateifu','nsa_z','nsa_elpetro_ba','nsa_elpetro_phi']][i_DRP])

                    if V_type == 'Ha':
                        print('Smoothness score:', map_smoothness)

                    print(param_outputs)
                    print('M90: ', M_R90, ' M_R90_err: ', M_R90_err)
                    print('M: ', M, ' M_err: ', M_err)
                    #print(mass_outputs)
                    print('Fit flag:', fit_flag)
                    ############################################################

            else:
                print(gal_ID, 'is missing photometric measurements.')

        else:
 
            if RUN_ALL_GALAXIES or TEXT_OUT:
                DRP_table = fillin_output_table(DRP_table,
                                                map_smoothness,
                                                i_DRP,
                                                col_name='smoothness_score')
            elif V_type == 'Ha':
                print('Smoothness score:', map_smoothness)
                print('5-sigma smoothness score:', map_smoothness_5sigma)

    else:
        print(gal_ID, 'is not a galaxy target.')

    print("\n")
################################################################################


#DRP_table.write('sn_test.fits', format='fits', overwrite=True)


################################################################################
# Save the output_table
#-------------------------------------------------------------------------------
if RUN_ALL_GALAXIES or TEXT_OUT:

    if V_type == 'Ha':
        out_filename = 'DRP_HaVel_map_results_' + vel_function + '_smooth_lt_' + str(map_smoothness_max) + '.txt'
    else:
        out_filename = 'DRP_starVel_map_resutls_' + vel_function + '.txt'

    #DRP_table.write(DRP_FILENAME, format = 'fits', overwrite=True)
    #DRP_table.write(out_filename,
    #                format='ascii.commented_header',
    #                overwrite=True)
################################################################################'''



################################################################################
# Print number of galaxies that were completely masked
#-------------------------------------------------------------------------------
if RUN_ALL_GALAXIES or TEXT_OUT:
    print('There were', num_masked_gal, 'galaxies that were completely masked.')
    print('There were', num_not_smooth, 'galaxies without smooth velocity maps.')
################################################################################


################################################################################
# Clock the program's run time to check performance.
#-------------------------------------------------------------------------------
FINISH = datetime.datetime.now()
print("Runtime:", FINISH - START)
################################################################################
