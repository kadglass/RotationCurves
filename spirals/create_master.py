'''
Convert the output of the rotation curve fitting code to the "master" file 
(smaller and more usable).
'''


################################################################################
# Import modules
#-------------------------------------------------------------------------------
from astropy.table import Table
################################################################################




################################################################################
# File to convert
#-------------------------------------------------------------------------------
filename = 'DRP_vel_map_results_BB_smooth_lt_2_AJLaBarca.fits'

old_file = Table.read(filename, format='fits')
################################################################################




################################################################################
# Create a new table with only a subset of the old columns
#-------------------------------------------------------------------------------
columns_to_keep = ['plate', 
                   'ifudsgn', 
                   'objra', 
                   'objdec', 
                   'z', 
                   'nsa_nsaid', 
                   'nsa_elpetro_mass', 
                   'nsa_elpetro_ba', 
                   'nsa_elpetro_phi', 
                   'nsa_elpetro_th50_r', 
                   'v_sys', 
                   'v_sys_err', 
                   'ba', 
                   'ba_err', 
                   'x0', 
                   'x0_err', 
                   'y0', 
                   'y0_err', 
                   'phi', 
                   'phi_err', 
                   'r_turn', 
                   'r_turn_err', 
                   'v_max', 
                   'v_max_err', 
                   'chi2', 
                   'alpha', 
                   'alpha_err', 
                   'nsa_elpetro_th90', 
                   'fit_flag', 
                   'smoothness_score', 
                   'Rmax',
                   'M', 
                   'M_err']

new_file = old_file[columns_to_keep]

new_file['rabsmag'] = old_file['nsa_elpetro_absmag'][:,4]
################################################################################




################################################################################
# Rename columns to match old column names
#
# So that old code can be used
#-------------------------------------------------------------------------------
new_names = ['MaNGA_plate', 
             'MaNGA_IFU', 
             'NSA_RA', 
             'NSA_DEC', 
             'NSA_redshift', 
             'NSAID', 
             'NSA_Mstar', 
             'NSA_ba', 
             'NSA_phi', 
             'NSA_elpetro_th50', 
             'Vsys_map', 
             'Vsys_err_map', 
             'ba_map', 
             'ba_err_map', 
             'x0_map', 
             'x0_err_map', 
             'y0_map', 
             'y0_err_map', 
             'phi_map', 
             'phi_err_map', 
             'Rturn_map', 
             'Rturn_err_map',
             'Vmax_map', 
             'Vmax_err_map', 
             'chi2_map', 
             'alpha_map', 
             'alpha_err_map', 
             'NSA_elpetro_th90', 
             'map_fit_flag', 
             'DRP_map_smoothness', 
             'Rmax_map', 
             'M90_map', 
             'M90_err_map']

for i in range(len(columns_to_keep)):
    new_file[columns_to_keep[i]].name = new_names[i]
################################################################################




################################################################################
# Add MaNGA targeting flag column
#
# https://sdss-marvin.readthedocs.io/en/latest/tutorials/sample-selection.html
#-------------------------------------------------------------------------------
new_file['MaNGA_sample'] = -1

new_file['MaNGA_sample'][(old_file['mngtarg1'] & 2**10) > 0] = 1
new_file['MaNGA_sample'][(old_file['mngtarg1'] & 2**11) > 0] = 2
new_file['MaNGA_sample'][(old_file['mngtarg1'] & 2**12) > 0] = 3
################################################################################




################################################################################
# Save new file
#-------------------------------------------------------------------------------
new_filename = 'DRP-dr17_vflag_BB_smooth2_mapFit_AJLaBarca.txt'

new_file.write(new_filename, format='ascii.commented_header', overwrite=True)
################################################################################




