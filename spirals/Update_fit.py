'''
Read in one of the output files produced from a mass run of DRP_vel_map_main.py 
and replace the best-fit values for a single galaxy with that produced from 
running the same file on a single galaxy.
'''


################################################################################
# Import modules
#-------------------------------------------------------------------------------
from astropy.table import Table
################################################################################




################################################################################
# Read in data file to be updated
#-------------------------------------------------------------------------------
file_directory = 'dr15_files/'

filename = 'DRP-master_file_vflag_BB_smooth1p85_mapFit_N2O2_HIdr2_morph_SK_H2_noWords_v8.txt'

bestfits = Table.read(file_directory + filename, 
                      format='ascii.commented_header')
################################################################################




################################################################################
# New best-fit values
#
# This is copied from the terminal output from running DRP_vel_map_main.py on a 
# single galaxy.
#-------------------------------------------------------------------------------
galaxy = '9035-1902'

new_fit = {'v_sys': 47.856749767450644, 'v_sys_err': 44.55196307925782, 'ba': 0.6416687537318244, 'ba_err': 6.668439976206128e-06, 'x0': 18.086892384023226, 'x0_err': 3.1386594715769527, 'y0': 22.895493476938658, 'y0_err': 1.5065215658505249, 'phi': 341.0090536254158, 'phi_err': 3.635611415022312e-07, 'v_max': 1483.3780929670424, 'v_max_err': 89.70014762516467, 'r_turn': 23.312924313419522, 'r_turn_err': 2.121106483377199, 'chi2': 52.86798311962578, 'alpha': 71.34626453248849, 'alpha_err': 999023.3782573426, 'Rmax': 6.920848670270989}

mass_new = {'M': 9.521389189526086, 'M_err': 10.138898071107707}

fit_flag = -4
################################################################################




################################################################################
# Replace values in output file with those given above
#-------------------------------------------------------------------------------
# 1) Find the row in the table corresponding to the galaxy
#-------------------------------------------------------------------------------
plate, ifu = galaxy.split('-')

row_boolean = (bestfits['MaNGA_plate'] == int(plate)) & (bestfits['MaNGA_IFU'] == int(ifu))
#-------------------------------------------------------------------------------
# 2) Replace each of the specified properties
#-------------------------------------------------------------------------------
bestfits['Vsys_map'][row_boolean] = new_fit['v_sys']
bestfits['Vsys_err_map'][row_boolean] = new_fit['v_sys_err']

bestfits['ba_map'][row_boolean] = new_fit['ba']
bestfits['ba_err_map'][row_boolean] = new_fit['ba_err']

bestfits['x0_map'][row_boolean] = new_fit['x0']
bestfits['x0_err_map'][row_boolean] = new_fit['x0_err']

bestfits['y0_map'][row_boolean] = new_fit['y0']
bestfits['y0_err_map'][row_boolean] = new_fit['y0_err']

bestfits['phi_map'][row_boolean] = new_fit['phi']
bestfits['phi_err_map'][row_boolean] = new_fit['phi']

bestfits['Vmax_map'][row_boolean] = new_fit['v_max']
bestfits['Vmax_err_map'][row_boolean] = new_fit['v_max_err']

bestfits['Rturn_map'][row_boolean] = new_fit['r_turn']
bestfits['Rturn_err_map'][row_boolean] = new_fit['r_turn_err']

bestfits['chi2_map'][row_boolean] = new_fit['chi2']

bestfits['alpha_map'][row_boolean] = new_fit['alpha']
bestfits['alpha_err_map'][row_boolean] = new_fit['alpha_err']

bestfits['Rmax_map'][row_boolean] = new_fit['Rmax']

bestfits['M90_map'][row_boolean] = mass_new['M']
bestfits['M90_err_map'][row_boolean] = mass_new['M_err']

bestfits['map_fit_flag'][row_boolean] = fit_flag
#-------------------------------------------------------------------------------
################################################################################




################################################################################
# Write updated file over original
#-------------------------------------------------------------------------------
bestfits.write(file_directory + filename, 
               format='ascii.commented_header', 
               overwrite=True)
################################################################################



