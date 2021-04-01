
from astropy.table import Table, QTable
import astropy.units as u

import numpy as np



################################################################################
# File names
#-------------------------------------------------------------------------------
#master_filename = '/Users/kellydouglass/Desktop/Pipe3D-master_file_vflag_10_smooth2p27_N2O2_noWords.txt'
master_filename = 'Pipe3D-master_file_vflag_BB_minimize_chi10_smooth2p27_mapFit_N2O2_noWords.txt'

map_fit_filename = 'DRP_vel_map_results_BB_smooth_lt_1p85_v2_diskFit.fits'
################################################################################



################################################################################
# Data files
#-------------------------------------------------------------------------------
#master_table = QTable.read(master_filename, format='ascii.ecsv')
master_table = Table.read(master_filename, format='ascii.commented_header')

map_fit_table = Table.read(map_fit_filename, format='fits')
################################################################################



################################################################################
# Create look-up dictionary for map_fit_table
#-------------------------------------------------------------------------------
map_fit_index = {}

for i in range(len(map_fit_table)):
    gal_ID = map_fit_table['plateifu'][i]

    map_fit_index[gal_ID] = i
################################################################################



################################################################################
# Columns to add
#-------------------------------------------------------------------------------
master_colnames = ['ba_map', 'ba_err_map', 
                   'phi_map', 'phi_err_map', 
                   'Vsys_map', 'Vsys_err_map', 
                   'x0_map', 'x0_err_map', 
                   'y0_map', 'y0_err_map', 
                   'Vmax_map', 'Vmax_err_map', 
                   'alpha_map', 'alpha_err_map', 
                   'Rturn_map', 'Rturn_err_map', 
                   'M90_map', 'M90_err_map', 
                   'Sigma_disk_map', 'Sigma_disk_err_map', 
                   'Rdisk_map', 'Rdisk_err_map', 
                   'M90_disk_map', 'M90_disk_err_map', 
                   'DRP_map_smoothness', 
                   'NSA_elpetro_th90', 
                   'chi2_map', 'chi2_disk_map']
map_fit_colnames = ['ba', 'ba_err', 
                    'phi', 'phi_err', 
                    'v_sys', 'v_sys_err', 
                    'x0', 'x0_err', 
                    'y0', 'y0_err', 
                    'v_max', 'v_max_err', 
                    'alpha', 'alpha_err', 
                    'r_turn', 'r_turn_err', 
                    'M90', 'M90_err', 
                    'Sigma_disk', 'Sigma_disk_err', 
                    'R_disk', 'R_disk_err', 
                    'M90_disk', 'M90_disk_err', 
                    'smoothness_score', 
                    'nsa_elpetro_th90', 
                    'chi2', 'chi2_disk']
'''
col_units = [None, None, 
             u.deg, u.deg, 
             u.km/u.s, u.km/u.s, 
             None, None, 
             None, None, 
             u.km/u.s, u.km/u.s, 
             None, None, 
             u.kpc, u.kpc, 
             u.dex(u.Msun), u.dex(u.Msun), 
             u.Msun/u.pc**2, u.Msun/u.pc**2, 
             u.kpc, u.kpc, 
             u.dex(u.Msun), u.dex(u.Msun), 
             None, 
             u.arcsec, 
             None, None]
'''
################################################################################




################################################################################
# Initialize columns in master table
#-------------------------------------------------------------------------------
Ngal = len(master_table)

for i,name in enumerate(master_colnames):
    '''
    if col_units[i] is not None:
        master_table[name] = np.nan*np.ones(Ngal)*col_units[i]
    else:
        master_table[name] = np.nan*np.ones(Ngal)
    '''
    #master_table[name] = np.nan*np.ones(Ngal)
    master_table[name] = -99*np.ones(Ngal, dtype=float)
################################################################################




################################################################################
# Fill in new columns
#-------------------------------------------------------------------------------
for i in range(Ngal):

    gal_ID = str(master_table['MaNGA_plate'][i]) + '-' + str(master_table['MaNGA_IFU'][i])

    # Find galaxy in map fit table
    i_fit = map_fit_index[gal_ID]

    for j in range(len(map_fit_colnames)):
        '''
        if col_units[j] is not None:
            master_table[master_colnames[j]][i] = map_fit_table[map_fit_colnames[j]][i_fit]*col_units[j]
        else:
            master_table[master_colnames[j]][i] = map_fit_table[map_fit_colnames[j]][i_fit]
        '''
        master_table[master_colnames[j]][i] = map_fit_table[map_fit_colnames[j]][i_fit]
################################################################################



################################################################################
# Save updated master table
#-------------------------------------------------------------------------------
master_table.write(master_filename[:-4] + '_v2.txt', 
                   #master_filename[:-4] + '_mapFit.txt', 
                   format='ascii.commented_header', #'ascii.ecsv', 
                   overwrite=True)
################################################################################




