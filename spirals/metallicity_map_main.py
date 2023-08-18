import time
#START = datetime.datetime.now()


from astropy.table import Table
import os
from astropy.io import fits
from metallicity_map import *
from metallicity_map_broadband import *

'''IMAGE_FORMAT = 'eps'

FILE_IDS = ['8082-12702']

IMAGE_DIR = '/Users/nityaravi/Documents/Research/RotationCurves/data/manga/Images/metallicity_maps_test/'

MANGA_FOLDER = '/Users/nityaravi/Documents/Research/RotationCurves/data/manga/'
VEL_MAP_FOLDER = MANGA_FOLDER + 'DR17/'
DRP_FILENAME = MANGA_FOLDER + 'DR17/' + 'drpall-v3_1_1.fits'

DRP_table = Table.read( DRP_FILENAME, format='fits')


DRP_index = {}

for i in range(len(DRP_table)):
    gal_ID = DRP_table['plateifu'][i]

    DRP_index[gal_ID] = i

for gal_ID in FILE_IDS:

    i_DRP = DRP_index[gal_ID]

    maps = extract_metallicity_data(VEL_MAP_FOLDER, gal_ID)
    print( gal_ID, "extracted")'''

MANGA_FOLDER = '/Users/nityaravi/Documents/Research/RotationCurves/data/manga/'
DRP_FOLDER = MANGA_FOLDER + 'DR17/'
IMAGE_DIR = '/Users/nityaravi/Documents/Research/RotationCurves/data/manga/metallicity_maps/'

DRP_TABLE_FN = MANGA_FOLDER + 'output_files/DR17/CURRENT_MASTER_TABLE/H_alpha_HIvel_BB_extinction_H2_MxCG_R90_v3p5.fits'

corr_law = 'CCM89'

gal_ID = '7443-6103'

DRP_table = Table.read(DRP_TABLE_FN, format='fits')
i_DRP = np.where(DRP_table['plateifu'] == gal_ID)[0][0]

# check if target is a galaxy
# want to choose center_coord, phi, inclination from fit if available
# else choose nsa values (conversion necessary for phi, inclination)

center_coord = (DRP_table['x0'][i_DRP], DRP_table['y0'][i_DRP])
phi = DRP_table['phi'][i_DRP]
ba = DRP_table['ba'][i_DRP]
z = DRP_table['nsa_z'][i_DRP]
A_g = DRP_table['A_g'][i_DRP]
A_r = DRP_table['A_r'][i_DRP]
r50 = DRP_table['nsa_elpetro_th50_r'][i_DRP]
log_M_HI = DRP_table['logHI'][i_DRP] 
log_M_H2 = DRP_table['logH2'][i_DRP]
log_M_star = DRP_table['M_disk'][i_DRP]





metallicity_param_outputs, r_kpc, scale, d_kpc, metallicity_mask = fit_metallicity_gradient(MANGA_FOLDER,
                                                    DRP_FOLDER, 
                                                    IMAGE_DIR, 
                                                    corr_law, 
                                                    gal_ID,
                                                    center_coord,
                                                    phi, 
                                                    ba,
                                                    z)
if metallicity_param_outputs is not None:

    print(metallicity_param_outputs)

    # add values to table

    surface_brightness_param_outputs = fit_surface_brightness_profile(DRP_FOLDER,
                                                                        IMAGE_DIR,
                                                                        gal_ID,
                                                                        A_g,
                                                                        A_r,
                                                                        #center_coord,
                                                                        #phi, 
                                                                        #ba,
                                                                        #z,
                                                                        r50,
                                                                        r_kpc,
                                                                        scale,
                                                                        d_kpc,
                                                                        metallicity_mask)


    R25_pc = surface_brightness_param_outputs['R25_pc']
    grad = metallicity_param_outputs['grad']
    Z0 = metallicity_param_outputs['12logOH_0']

    Z = find_global_metallicity(R25_pc, grad, Z0)

    M_HI = 10**(log_M_HI)
    M_H2 = None
    M_star = None

    if log_M_H2 > 0:
        M_H2 = 10**(log_M_H2)

    else:
        M_star = 10**(log_M_star)

    Mz, Md = calculate_metal_mass(Z, M_HI, M_H2, M_star)

    print('Mz: ', Mz)
    print('Md: ', Md)
                                                                
    
#    if surface_brightness_param_outputs is not None:

#        R25, R25_err = surface_brightness_param_outputs['R25'], surface_brightness_param_outputs['R25_err'] 

#        Z_12logOH, Z_12logOH_err = linear_metallicity_gradient_sigma(R25, R25_err, MOREPARAMS)

