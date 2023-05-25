# IMPORT MODULES
#-------------------------------------------------------------------------------
import datetime
START = datetime.datetime.now()

import os.path, warnings

import numpy as np

from astropy.table import Table
import astropy.units as u
import astropy.constants as const

from file_io import add_columns, fillin_output_table

from DRP_rotation_curve import extract_data

from Pipe3D_vel_map import extract_Pipe3D_data

from DRP_vel_map import fit_vel_map, estimate_total_mass

import sys
#sys.path.insert(1, '/Users/kellydouglass/Documents/Research/Rotation_curves/RotationCurves/')
#sys.path.insert(1, '/home/kelly/Documents/RotationCurves/')
sys.path.insert(1, '/Users/nityaravi/Documents/GitHub/RotationCurves/')
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
# Maximum allowed smoothness score (H-alpha velocity maps only)
map_smoothness_max = 2.27

# Velocity function to use (options are 'BB' or 'tanh')
vel_function = 'BB'

# Velocity map to use
#V_type = 'Ha'
V_type = 'star'
################################################################################



################################################################################
# List of files (in "[MaNGA_plate]-[MaNGA_IFU]" format) to be ran through the
# individual galaxy version of this script.
# 
# If RUN_ALL_GALAXIES is set to True, then code will ignore what is in FILE_IDS.
#-------------------------------------------------------------------------------
#FILE_IDS = ['10001-3702']

'''
worked = [ '7992-6104', '8077-12704', '8077-6102',
            '8082-9102', '8150-9102', '8153-1901',
            '8158-1901',
        '8255-12704', '8257-9102',
        '8262-3702', '8262-9102', '8311-3703', 
        '8318-12702', '8318-9101', '8320-9101', '8322-1901',
        '8322-3701', '8329-12701', '8330-12703', '8332-3702',
        '8332-9102', '8335-12705', '8338-12701', '8440-6104',
        '8450-6102', '8453-3704', '8455-3701', '8459-1901', '8462-9101',
        '8465-9102', '8466-12702', '8483-6101', '8547-6102',
        '8548-12704', '8551-12705', '8552-12701',
        '8588-12705', '8588-6101', '8592-6101', ]
        '8588-12705', '8588-6101', '8592-6101', '8603-6104',
        '8604-12702', '8604-9102', '8624-12702', '8624-12703',
        '8626-12701', '8626-12702', '8626-3702', '8712-6101', '8713-6104',
        '8932-9102', '8935-6104', '8945-12701', '8950-12705',
        '8952-6104', '8978-3701', '8979-6102', '8987-3701',
        '8989-9102','8996-3703', '9024-1902', '9027-12701', '9029-12702', '9029-6102',
        '9036-9102', '9042-12703', '9047-6104', '9049-6104', '9050-3704',
        '9050-9101', '9085-12703', '9095-1901', '9095-9102',
        '9485-12705', '9485-3701', '9486-12701',
        '9486-12702', '9500-1901', '9508-12705',
        '9508-6101', '9508-6104', '9865-9102', '9871-6101',
        '9881-12705', '9881-3702', '9888-12704', '8092-3701', '8095-1902',
        '7977-3702', '7977-3703', '12085-6103', '12090-1901',
        '10218-12703', '9863-1902',
        '8089-12701', '9495-6101', '9506-1901', '10844-3703', '12769-6104', 
        '10513-1901', '12700-12702', '9494-3701', '10219-1901', '9863-3701', '10514-3701'
         ]

failed = [
        '7990-6104', #SVD did not converge
        '8080-12702', # ran but no fit params output
        '8082-12702', # all nan slice - SVD did not converge
        '8146-1901', # singular matrix
        '8155-12701', nan sl - svd did not converge
        '8452-12705',
        '8595-3703',
        '8595-6104'# singular matrix
        '8600-1901',  # singular matrix
        '8600-3704', # singular matrix'
        '8603-6103', # math domain error
          '8626-3703', nan sl - svd
          '8727-3701',
          '8993-6104', svd dnc
          '8997-12704',
          '9041-12701',  svd dnc
          '9044-6101', '9196-6103',
          '9888-12705','9513-3702',
          '11744-6103',
          '11974-3701',
          '7977-3704',
          '12085-9102',
          '9493-6103',
          '8087-9102',
          '10214-6101', # singular
          '8934-3701'

]

'''


FILE_IDS =  [ ,
           
         
         
         
         
           
         
        
          
          ]

RUN_ALL_GALAXIES = False

TEXT_OUT = False
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
    #IMAGE_DIR = LOCAL_PATH + 'Images/Pipe3D_star/'
    IMAGE_DIR = '/Users/nityaravi/Documents/Research/RotationCurves/data/manga/Images/Pipe3D_star/'

    # Create directory if it does not already exist
    if not os.path.isdir( IMAGE_DIR):
        os.makedirs( IMAGE_DIR)
else:
    #IMAGE_DIR = None
    IMAGE_DIR = '/Users/nityaravi/Documents/Research/RotationCurves/data/manga/Images/Pipe3D_star/'


#MANGA_FOLDER = '/Users/kellydouglass/Documents/Research/data/SDSS/'
#NSA_FILENAME = '/Users/kellydouglass/Documents/Drexel/Research/Data/NSA/nsa_v1_0_1.fits'

MANGA_FOLDER = '/Users/nityaravi/Documents/RotationCurves/Research/data/manga/'
NSA_FILENAME = '/Users/nityaravi/Documents/RotationCurves/Research/data/nsa_v1_0_1.fits'

'''
MANGA_FOLDER = '/home/kelly/Documents/Data/SDSS/dr16/manga/spectro/'
NSA_FILENAME = '/home/kelly/Documents/Data/NSA/nsa_v1_0_1.fits'
'''
#PIPE3D_FOLDER = MANGA_FOLDER + 'dr15/manga/spectro/pipe3d/v2_4_3/2.4.3/'
#DRP_FOLDER = MANGA_FOLDER + 'dr16/manga/spectro/analysis/v2_4_3/2.2.1/HYB10-GAU-MILESHC/'
#DRP_FILENAME = MANGA_FOLDER + 'dr16/manga/spectro/redux/v2_4_3/drpall-v2_4_3.fits'


# for DR17
PIPE3D_FOLDER = MANGA_FOLDER + 'Pipe3D/'
DRP_FOLDER = MANGA_FOLDER + 'DR17/'
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
# Fit the rotation curve for the derived stellar velocity map for all of the 
# galaxies in the 'files' array.
#-------------------------------------------------------------------------------
num_masked_gal = 0 # Number of completely masked galaxies
num_not_smooth = 0 # Number of galaxies which do not have smooth velocity maps

for gal_ID in FILE_IDS:

    i_DRP = DRP_index[gal_ID]

    if DRP_table['mngtarg1'][i_DRP] > 0:
    
        ########################################################################
        # Extract the necessary data from the .fits files.
        #-----------------------------------------------------------------------
        DRPmaps = extract_data(DRP_FOLDER, 
                               gal_ID, 
                               ['r_band', 'Ha_flux', 'Ha_sigma'])

        P3Dmaps = extract_Pipe3D_data(PIPE3D_FOLDER, 
                                      gal_ID, 
                                      ['star_vel', 'Ha_vel'])

        print( gal_ID, "extracted")
        ########################################################################


        ########################################################################
        # Construct mask for the velocity map
        #
        # Following Aquino-Ortiz et al. (2020), we require a maximum error of 
        # 25 km/s in the derived stellar velocities.
        #-----------------------------------------------------------------------
        if V_type == 'star':
            vel_mask = (P3Dmaps['star_vel_err'] > 25) | np.isnan(P3Dmaps['star_vel']) | (P3Dmaps['star_vel'] == 0)
        elif V_type == 'Ha':
            vel_mask = np.isnan(P3Dmaps['Ha_vel']) | (P3Dmaps['Ha_vel'] == 0)
        ########################################################################


        ########################################################################
        # Calculate degree of smoothness of velocity map
        #-----------------------------------------------------------------------
        can_fit = True

        if V_type == 'Ha':
            map_smoothness = how_smooth(P3Dmaps['Ha_vel'], vel_mask)

            map_smoothness_5sigma = how_smooth(P3Dmaps['Ha_vel'], 
                                               np.logical_or(vel_mask > 0, 
                                                             np.abs(DRPmaps['Ha_flux']*np.sqrt(DRPmaps['Ha_flux_ivar'])) < 5))

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

            NSA_ID = DRP_table['nsa_nsaid'][i_DRP]
            ####################################################################
            
            
            if axis_ratio > -9999:
                ################################################################
                # Subtract systemic velocity from velocity map
                #---------------------------------------------------------------
                vel_map = P3Dmaps[V_type + '_vel'] - (z*const.c.to('km/s')).value
                ################################################################


                ################################################################
                # Extract rotation curve data for the .fits file in question and 
                # create an astropy Table containing said data.
                #---------------------------------------------------------------
                start = datetime.datetime.now()
                
                param_outputs, num_masked_gal, fit_flag = fit_vel_map(vel_map, 
                                                                      1/P3Dmaps[V_type + '_vel_err']**2, 
                                                                      vel_mask, 
                                                                      DRPmaps['Ha_sigma'], 
                                                                      DRPmaps['Ha_sigma_ivar'], 
                                                                      DRPmaps['Ha_sigma_mask'], 
                                                                      DRPmaps['Ha_flux'], 
                                                                      DRPmaps['Ha_flux_ivar'], 
                                                                      DRPmaps['Ha_flux_mask'], 
                                                                      DRPmaps['r_band'], 
                                                                      DRPmaps['r_band_ivar'], 
                                                                      axis_ratio, 
                                                                      phi_EofN_deg, 
                                                                      z, 
                                                                      gal_ID, 
                                                                      vel_function, 
                                                                      V_type=V_type, 
                                                                      IMAGE_DIR=IMAGE_DIR, 
                                                                      IMAGE_FORMAT=IMAGE_FORMAT, 
                                                                      num_masked_gal=num_masked_gal)
                                                             
                fit_time = datetime.datetime.now() - start
                
                print(gal_ID, "velocity map fit", fit_time)
                ################################################################

                ################################################################
                # Extract the necessary data from the NSA table.
                #---------------------------------------------------------------
                i_NSA = NSA_index[NSA_ID]

                R90 = NSA_table['ELPETRO_TH90_R'][i_NSA]
                ################################################################

                if param_outputs is not None:
                    ############################################################
                    # Estimate the total mass within the galaxy
                    #-----------------------------------------------------------
                    mass_outputs = estimate_total_mass([param_outputs['v_max'], 
                                                        param_outputs['r_turn'], 
                                                        param_outputs['alpha']], 
                                                       R90, 
                                                       z, 
                                                       vel_function, 
                                                       gal_ID)
                    ############################################################
                else:
                    mass_outputs = None


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
                                                        mass_outputs, 
                                                        i_DRP)
                        DRP_table = fillin_output_table(DRP_table, 
                                                        fit_flag, 
                                                        i_DRP, 
                                                        col_name='fit_flag')

                    print(gal_ID, "written")
                    ################################################################

                else:
                    ################################################################
                    # Print output to terminal if not analyzing all galaxies
                    #---------------------------------------------------------------
                    print(DRP_table[['plateifu','nsa_z','nsa_elpetro_ba','nsa_elpetro_phi']][i_DRP])

                    if V_type == 'Ha':
                        print('Smoothness score:', map_smoothness)

                    print(param_outputs)
                    print(mass_outputs)
                    print('Fit flag:', fit_flag)
                    ################################################################
            
            else:
                print(gal_ID, 'is missing photometric measurements.')

        else:
            print("Galaxy's map is not smooth enough to fit.")

            num_not_smooth += 1

            if RUN_ALL_GALAXIES or TEXT_OUT:
                DRP_table = fillin_output_table(DRP_table, 
                                                map_smoothness, 
                                                i_DRP, 
                                                col_name='smoothness_score')
            else:
                print('Smoothness score:', map_smoothness)
                print('5-sigma smoothness score:', map_smoothness_5sigma)

    else:
        print(gal_ID, 'is not a galaxy target.')

    print("\n")
################################################################################



################################################################################
# Save the output_table
#-------------------------------------------------------------------------------
if RUN_ALL_GALAXIES or TEXT_OUT:

    if V_type == 'Ha':
        out_filename = 'Pipe3D_HaVel_map_results_' + vel_function + '_smooth_lt_' + str(map_smoothness_max) + '.txt'
    else:
        out_filename = 'Pipe3D_starVel_map_results_' + vel_function + '.txt'

    DRP_table.write('out.fits', format = 'fits', overwrite=True)
    '''DRP_table.write(out_filename, 
                    format='ascii.commented_header', 
                    overwrite=True)'''
################################################################################



################################################################################
# Print number of galaxies that were completely masked
#-------------------------------------------------------------------------------
if RUN_ALL_GALAXIES or TEXT_OUT:
    print('There were', num_masked_gal, 'galaxies that were completely masked.')
################################################################################


################################################################################
# Clock the program's run time to check performance.
#-------------------------------------------------------------------------------
FINISH = datetime.datetime.now()
print("Runtime:", FINISH - START)
################################################################################

