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
galaxy = '8555-12703'

new_fit = {'v_sys': -11.7915157879904, 'v_sys_err': 0.15179609332033261, 'ba': 0.3451567854528824, 'ba_err': 0.00042755585370431024, 'x0': 36.91848716322662, 'x0_err': 0.01532483005174314, 'y0': 36.27603198447388, 'y0_err': 0.012743543287219971, 'phi': 323.5159996195613, 'phi_err': 0.032811910319898996, 'v_max': 265.2916256570105, 'v_max_err': 0.6207670891661553, 'r_turn': 4.58334337593553, 'r_turn_err': 0.024066624164666522, 'chi2': 9.036286641175959, 'alpha': 1.7546423562137674, 'alpha_err': 0.016744408647898883, 'Rmax': 17.50798202428108}

mass_new = {'M': 11.348787424069096, 'M_err': 10.920815602546794}

fit_flag = -3
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



