import matplotlib.pyplot as plt
import numpy as np
from astropy.table import Table
from astropy.io import fits
import numpy.ma as ma
from scipy.optimize import curve_fit

# import sys
# sys.path.insert(1, '/Users/nityaravi/Documents/GitHub/RotationCurves/spirals/')

# from DRP_vel_map_functions import deproject_spaxel
# from metallicity_map_functions import linear_metallicity_gradient
from metallicity_map import fit_stellar_metallicity_map_gradient

# from DRP_rotation_curve import extract_data


# MANGA_FOLDER = '/Users/nityaravi/Documents/Research/RotationCurves/data/manga/'
MANGA_FOLDER = '/global/cfs/cdirs/sdss/data/sdss/dr17/manga/'


DRP_FOLDER = '/scratch/kdougla7/data/SDSS/dr17/manga/spectro/analysis/v3_1_1/3.1.0/HYB10-MILESHC-MASTARSSP/'
P3D_FOLDER =    '/scratch/kdougla7/data/SDSS/dr17/manga/spectro/pipe3d/'



IMAGE_DIR = '/scratch/nravi3/' + 'stellar_metallicity_gradient_median/'

DRP_TABLE_FN = '/scratch/nravi3/Elliptical_sphdisk_refitspirals_BPT_illustris_v11_gradZ'


FILE_IDS = []

RUN_ALL_GALAXIES = True

DRP_table = Table.read(DRP_TABLE_FN + '.fits', format='fits')
DRP_index = {}

for i in range(len(DRP_table)):
    gal_ID = DRP_table['plateifu'][i]

    DRP_index[gal_ID] = i

if RUN_ALL_GALAXIES:
    N_files = len(DRP_table)

    FILE_IDS = list(DRP_index.keys())

    # add columns to table
    DRP_table['stellar_grad_Z'] = np.ones(len(DRP_table))*np.nan
    DRP_table['stellar_grad_Z_err'] = np.ones(len(DRP_table))*np.nan
    DRP_table['stellar_Z_0'] = np.ones(len(DRP_table))*np.nan
    DRP_table['stellar_Z_0_err'] = np.ones(len(DRP_table))*np.nan

    for i_DRP in range(len(FILE_IDS)):
        # for i_DRP in range(0, 20):
    
        gal_ID = FILE_IDS[i_DRP]

        print('processing: ', gal_ID)

        if DRP_table['mngtarg1'][i_DRP] > 0:



            
            center_coord = (DRP_table['x0'][i_DRP], DRP_table['y0'][i_DRP])
            
            if ma.is_masked(center_coord[0]):

                center_coord = (None, None)
                phi = DRP_table['nsa_elpetro_phi'][i_DRP]
                ba = DRP_table['nsa_elpetro_ba'][i_DRP]

            else:
                phi = DRP_table['phi'][i_DRP]
                ba = DRP_table['ba'][i_DRP]

            z = DRP_table['nsa_z'][i_DRP]

            metallicity_param_outputs = fit_stellar_metallicity_map_gradient(P3D_FOLDER, 
                                                                             DRP_FOLDER, 
                                                                             IMAGE_DIR,
                                                                             gal_ID,
                                                                             ba, 
                                                                             z, 
                                                                             center_coord, 
                                                                             phi )


            # metallicity_param_outputs, r_kpc, scale, d_kpc, metallicity_mask = fit_metallicity_gradient(MANGA_FOLDER,
            #                                                     DRP_FOLDER, 
            #                                                     IMAGE_DIR, 
            #                                                     corr_law, 
            #                                                     gal_ID,
            #                                                     center_coord,
            #                                                     phi, 
            #                                                     ba,
            #                                                     z)
            if metallicity_param_outputs is not None:

                grad = metallicity_param_outputs['grad']
                grad_err = metallicity_param_outputs['grad_err']
                Z0 = metallicity_param_outputs['Z0']
                Z0_err = metallicity_param_outputs['Z0_err']


                DRP_table['stellar_grad_Z'][i_DRP] = grad
                DRP_table['stellar_grad_Z_err'][i_DRP] = grad_err
                DRP_table['stellar_Z_0'][i_DRP] = Z0
                DRP_table['stellar_Z_0_err'][i_DRP] = Z0_err
        
        else:
            print('Not a galaxy')    
                    
    
                
    
    
        DRP_table.write(DRP_TABLE_FN + '_stelZ.fits', format='fits', overwrite=True)