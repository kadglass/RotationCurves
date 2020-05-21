'''
Generate plots of the fitted rotation curves for all galaxies.

This is an implementation of Rot_curve_plots.ipynb for all galaxies (instead of 
just one galaxy).
'''


################################################################################
# Import modules
#-------------------------------------------------------------------------------
import numpy as np

from astropy.table import QTable
import astropy.units as u

import matplotlib.pyplot as plt

from Pipe3D_rotation_curve import extract_data, calc_rot_curve
from dark_matter_mass_plottingFunctions import plot_fitted_rot_curve, plot_fitted_rot_curve_mass
################################################################################



################################################################################
# Directories and file names
#-------------------------------------------------------------------------------
IMAGE_DIR = 'Images/Pipe3D/'

FILE_DIR = 'Pipe3D-rot_curve_data_files/'

MANGA_FOLDER = '../data/MaNGA/MaNGA_DR15/pipe3d/'

master_table_filename = 'Pipe3D-master_file_vflag_10.txt'
################################################################################



################################################################################
# Import data
#-------------------------------------------------------------------------------
master_table = QTable.read(master_table_filename, format='ascii.ecsv')
################################################################################



################################################################################
# Generate fitted rotation curve plots
#-------------------------------------------------------------------------------
for i in range(len(master_table)):

    ############################################################################
    # Construct galaxy ID
    #---------------------------------------------------------------------------
    gal_ID = str(master_table['MaNGA_plate'][i]) + '-' + str(master_table['MaNGA_IFU'][i])
    #---------------------------------------------------------------------------


    ############################################################################
    # Plot fitted rotation curve
    #---------------------------------------------------------------------------
    plot_fitted_rot_curve( gal_ID, master_table[i], FILE_DIR, IMAGE_DIR=IMAGE_DIR)
    #---------------------------------------------------------------------------

    '''
    ############################################################################
    # Construct galaxy file name
    #---------------------------------------------------------------------------
    filename = MANGA_FOLDER + str(master_table['MaNGA_plate'][i]) + '/manga-' \
               + gal_ID + '.Pipe3D.cube.fits.gz'
    #---------------------------------------------------------------------------


    ############################################################################
    # Read data from fits file
    #---------------------------------------------------------------------------
    Ha_vel, Ha_vel_err, v_band, v_band_err, sMass_density,_,_,_,_ = extract_data( filename)
    #---------------------------------------------------------------------------


    ############################################################################
    # Extract rotation curve information
    #---------------------------------------------------------------------------
    rot_data_table,_,_ = calc_rot_curve( Ha_vel, Ha_vel_err, v_band, v_band_err, 
                                         sMass_density, master_table['NSA_ba'][i],
                                         master_table['NSA_phi'][i], 
                                         master_table['NSA_redshift'][i], gal_ID)
    #---------------------------------------------------------------------------


    ############################################################################
    # Plot fitted rotation curve with DM, M* curves
    #---------------------------------------------------------------------------
    plot_fitted_rot_curve_mass( gal_ID, master_table[i], rot_data_table, 
                                FILE_DIR, IMAGE_DIR=IMAGE_DIR)
    #---------------------------------------------------------------------------
    '''
################################################################################



