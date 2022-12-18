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
file_directory = ''

filename = 'DRP-master_file_vflag_BB_smooth1p85_mapFit_N2O2_HIdr2_morph_SK_H2_noWords_v7.txt'

bestfits = Table.read(file_directory + filename, 
                      format='ascii.commented_header')
################################################################################




################################################################################
# New best-fit values
#
# This is copied from the terminal output from running DRP_vel_map_main.py on a 
# single galaxy.
#-------------------------------------------------------------------------------
galaxy = '9876-6101'

new_fit = {'v_sys': -10.516612392916022, 'v_sys_err': 24.192473943415685, 'ba': 0.7229720952623375, 'ba_err': 0.31289998514833794, 'x0': 26.10157344194446, 'x0_err': 7.293701680809073, 'y0': 32.48742129313883, 'y0_err': 8.146629110036697, 'phi': 92.62073451106335, 'phi_err': 2.7347459637415263, 'v_max': 91.46913771379435, 'v_max_err': 37.1721957249918, 'r_turn': 4.239668535289252, 'r_turn_err': 2.6894744839285423, 'chi2': 91.671286001927, 'alpha': 53.868728233105884, 'alpha_err': 724.4277776731709, 'Rmax': 6.445871033708616}

mass_new = {'M': 9.989988053582147, 'M_err': 9.882863446313143}

fit_flag = -1
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



