import datetime
START = datetime.datetime.now()

import os.path, warnings
import sys

import numpy as np

from astropy.table import Table
import astropy.units as u

from IO_data import extract_data, add_cols, extract_Pipe3d_data
from elliptical_stellar_mass import *



FILE_IDS = ['11011-1901','10223-9101','9877-12704','11760-9101','11953-1902',
            '9490-9101','11830-9101','8319-6101','8592-1901','8325-3704',
            '9878-6103','8716-1902','8143-3704','12514-6101','10837-1901',
            '9487-12705','8133-6103','9887-1901','12067-3704','8274-6103',
            '8134-3702','8085-6101','9089-1902','9488-6104','11953-6102',
            '11836-3702','12678-6104','11865-9101','12620-1901','8319-1902',
            '9039-3703','12485-6104','9090-6103','8551-6103','11950-12703',
            '11826-6104','11836-3703','8625-3702','8614-12703','12067-3701',
            '8093-9101','8330-3703','11828-3701','11969-9101','8248-1902',
            '8937-3702','10840-6102','8481-9101','9862-6101','8149-12704',
            '8488-6101','11946-3704','9512-6104','11969-9102','8330-12705',
            '9868-6104','8566-6102','8241-3701','8612-6103','10837-6101',
            '12073-3703','9502-12701','9186-1902','11004-9102','11011-3701',
            '9506-6104','11024-12702','10501-3701','8158-3702','8482-12704',
            '10511-6104','8710-1901','8655-3703','11757-6102','9875-12705',
            '10513-1902','8598-6101','7975-3703','8939-3701','8550-1901',
            '7959-6104','8456-6103','11828-6104','10220-1902','8725-1902',
            '9863-3704','8710-1901','8259-1901','9002-9101','11868-12702',
            '11748-6102','8454-12705','8310-3701','11017-1902','9502-6103',
            '11948-12705','10495-3703','11836-12703','10498-3703','8262-6102']

# FILE_IDS = ['11011-1901']

IMAGE_FORMAT = 'png'

################################################################################
# Paths for Nitya Macbook
################################################################################

MANGA_FOLDER = '/Users/nityaravi/Documents/Research/RotationCurves/data/manga/'
IMAGE_DIR = MANGA_FOLDER + 'Ellipticals_Images/'
MAP_FOLDER = MANGA_FOLDER + 'DR17/'
PIPE3D_FOLDER = MANGA_FOLDER +'Pipe3D/'
DRP_FILENAME = MANGA_FOLDER + 'output_files/DR17/CURRENT_MASTER_TABLE/Elliptical_StelVelDispDAPMeanSigma_Mvir_smoothness_lt_2_dipole.fits'
OUT_FILENAME = MANGA_FOLDER + 'output_files/DR17/CURRENT_MASTER_TABLE/Elliptical_StelVelDispDAPMeanSigma_Mvir_smoothness_lt_2_dipole_Mstar.fits'
COV_DIR = MANGA_FOLDER + 'elliptical_stellar_mass_cov/'

################################################################################
# Paths for Bluehive
################################################################################

# MANGA_FOLDER = '/scratch/kdougla7/data/SDSS/dr17/manga/spectro/'
# IMAGE_DIR = '/scratch/nravi3/ellipticals/'
# MAP_FOLDER = MANGA_FOLDER + 'analysis/v3_1_1/3.1.0/HYB10-MILESHC-MASTARSSP/'
# PIPE3D_FOLDER = MANGA_FOLDER + 'pipe3d/'
# #update
# DRP_FILENAME = '/scratch/nravi3/disk_masses_HIdr3_err_morph_v2.fits'

################################################################################
################################################################################

CLEAR_COLS = True  # zeros out columns in table
TEXT_OUT = True # print info

START = datetime.datetime.now()

################################################################################
# Open the DRPall file
#-------------------------------------------------------------------------------
DRP_table = Table.read(DRP_FILENAME, format='fits')

DRP_index = {}

for i in range(len(DRP_table)):
    gal_ID = DRP_table['plateifu'][i]

    DRP_index[gal_ID] = i
################################################################################

if CLEAR_COLS:
    DRP_table = add_cols(DRP_table, ['rho_c','rho_c_err',
                                     'R_scale', 'R_scale_err',
                                     'M_star_esph', 'M_star_esph_err',
                                     'chi2_M_star_esph'])
    

for gal_ID in FILE_IDS:

    i_DRP = DRP_index[gal_ID]

    # running only on galaxies that virial masses

    if DRP_table['Mvir'][i_DRP] > 0:

        ########################################################################
        # Extract necessary fits files
        #-----------------------------------------------------------------------
        pipe3d_maps = extract_Pipe3d_data(PIPE3D_FOLDER, gal_ID, ['sMass'])
        if pipe3d_maps is None:
            print('No Pipe3D data for ', gal_ID)
            continue

        maps = extract_data(MAP_FOLDER, gal_ID, ['Ha', 'flux'])
        if maps is None:
            print('No data for ', gal_ID)
            continue


        ########################################################################
        # get parameters from data table
        #-----------------------------------------------------------------------

        ba = DRP_table['nsa_elpetro_ba'][i_DRP]
        phi = DRP_table['nsa_elpetro_phi'][i_DRP]
        z = DRP_table['nsa_z'][i_DRP]

        ########################################################################
        # generate data table of enclosed stellar mass
        #-----------------------------------------------------------------------

        data_table = calc_mass_curve(pipe3d_maps['sMass_density'],
                                     pipe3d_maps['sMass_density_err'],
                                     maps['mflux'],
                                     ba,
                                     phi,
                                     z)
        
        ########################################################################
        # fit mass curve
        #-----------------------------------------------------------------------
        if data_table is not None:
        
            best_fit_params = fit_mass_curve(data_table, 
                                         gal_ID, 
                                         COV_DIR=COV_DIR, 
                                         IMAGE_DIR=IMAGE_DIR, 
                                         IMAGE_FORMAT='png')
            if TEXT_OUT:
                print(best_fit_params)
        else:
            best_fit_params = None
            print('failed')
        
        if best_fit_params is not None:
            ####################################################################
            # calculate stellar mass
            #-------------------------------------------------------------------

            M, M_err = calc_tot_stellar_mass(gal_ID, 
                                            best_fit_params['rho_c'], 
                                            best_fit_params['R_scale'], 
                                            COV_DIR=COV_DIR)
            if TEXT_OUT:
                print('M_star: ', M)
                print('M_star_err: ', M_err)
            
            ####################################################################
            # populate data table
            #-------------------------------------------------------------------

            DRP_table['rho_c'][i_DRP] = best_fit_params['rho_c']
            DRP_table['rho_c_err'][i_DRP] = best_fit_params['rho_c_err']
            DRP_table['R_scale'][i_DRP] = best_fit_params['R_scale']
            DRP_table['R_scale_err'][i_DRP] = best_fit_params['R_scale_err']
            DRP_table['chi2_M_star_esph'][i_DRP] = best_fit_params['chi2_M_star']


            DRP_table['M_star_esph'][i_DRP] = M
            DRP_table['M_star_esph_err'][i_DRP] = M_err

            DRP_table.write(OUT_FILENAME, format='fits', overwrite=True)

    else:
        print(gal_ID, ' not in sample')

print('Runtime: ', datetime.datetime.now() - START)
DRP_table.write(OUT_FILENAME, format='fits', overwrite=True)