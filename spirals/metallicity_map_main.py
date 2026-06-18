import time
#START = datetime.datetime.now()


from astropy.table import Table
import os
from astropy.io import fits
from metallicity_map import *
# from metallicity_map_broadband import *


# MANGA_FOLDER = '/Users/nityaravi/Documents/Research/RotationCurves/data/manga/'
MANGA_FOLDER = '/global/cfs/cdirs/sdss/data/sdss/dr17/manga/'


DRP_FOLDER = MANGA_FOLDER + 'spectro/analysis/v3_1_1/3.1.0/HYB10-MILESHC-MASTARSSP/'


IMAGE_DIR = '/pscratch/sd/n/nravi/metallicity_maps/'

DRP_TABLE_FN = '/pscratch/sd/n/nravi/BTFR/' + 'Elliptical_sphdisk_refitspirals_BPT_illustris_v11_gradZ'

corr_law = 'CCM89'

method = 'map'
#method = 'global'


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


if method == 'map':

    # add columns to table]
    # DRP_table['grad_Z'] = np.ones(len(DRP_table))*np.nan
    # DRP_table['grad_Z_err'] = np.ones(len(DRP_table))*np.nan
    # DRP_table['Z_0'] = np.ones(len(DRP_table))*np.nan
    # DRP_table['Z_0_err'] = np.ones(len(DRP_table))*np.nan


    for i_DRP in range(1989,len(FILE_IDS)):
    # for i_DRP in range(0, 20):

        gal_ID = FILE_IDS[i_DRP]

        print('processing: ', gal_ID)

        if DRP_table['mngtarg1'][i_DRP] > 0:


            i_DRP = np.where(DRP_table['plateifu'] == gal_ID)[0][0]

            
            center_coord = (DRP_table['x0'][i_DRP], DRP_table['y0'][i_DRP])

            phi = DRP_table['phi'][i_DRP]
            ba = DRP_table['ba'][i_DRP]
            z = DRP_table['nsa_z'][i_DRP]



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

                grad = metallicity_param_outputs['grad']
                grad_err = metallicity_param_outputs['grad_err']
                Z0 = metallicity_param_outputs['12logOH_0']
                Z0_err = metallicity_param_outputs['12logOH_0_err']


                DRP_table['grad_Z'][i_DRP] = grad
                DRP_table['grad_Z_err'][i_DRP] = grad_err
                DRP_table['Z_0'][i_DRP] = Z0
                DRP_table['Z_0_err'][i_DRP] = Z0_err
        
        else:
            print('Not a galaxy')    
                

            


    DRP_table.write(DRP_TABLE_FN + '.fits', format='fits', overwrite=True)

# elif method == 'global':

#     #for i in range(0, len(DRP_table)):
#     for i in range(0, len(DRP_table)):


#         gal_ID = DRP_table['plateifu'][i]

#         print('Processing ', gal_ID)


#         fluxes = {'OII': DRP_table['Flux_OII_3726'][i],
#                     'OII_err': DRP_table['Flux_OII_3726_Err'][i],
#                     'OII2': DRP_table['Flux_OII_3728'][i],
#                     'OII2_err': DRP_table['Flux_OII_3728_Err'][i], 
#                     'NII': DRP_table['Flux_NII_6547'][i],
#                     'NII_err': DRP_table['Flux_NII_6547_Err'][i],
#                     'NII2': DRP_table['Flux_NII_6583'][i],
#                     'NII2_err': DRP_table['Flux_NII_6583_Err'][i],
#                     'OIII': DRP_table['Flux_OIII_4958'][i],
#                     'OIII_err': DRP_table['Flux_OIII_4958_Err'][i],
#                     'OIII2': DRP_table['Flux_OIII_5006'][i],
#                     'OIII2_err': DRP_table['Flux_OIII_5006_Err'][i],
#                     'Ha' : DRP_table['Flux_Ha_6562'][i],
#                     'Ha_err': DRP_table['Flux_Ha_6562_Err'][i],
#                     'Hb': DRP_table['Flux_Hb_4861'][i],
#                     'Hb_err': DRP_table['Flux_Hb_4861_Err'][i]
#                     }

#         Z, Z_err = calculate_global_metallicity(fluxes)
#         print(Z, Z_err)

#         DRP_table['Z'][i] = Z
#         DRP_table['Z_err'][i] = Z_err 

#         if i % 100 == 0:
#             DRP_table.write(MANGA_FOLDER + 'output_files/DR17/CURRENT_MASTER_TABLE/' + 'H_alpha_HIvel_BB_extinction_H2_MxCG_R90_v3p5_Z_SFR_Portsmouthflux_Zglob.fits',
#                     format='fits',
#                     overwrite=True)



#         if Z > 0 and DRP_table['logHI'][i] > 0 and DRP_table['param_H2'][i] > 0:

#             M_HI = 10**DRP_table['logHI'][i]
#             M_HI_err = 0
#             M_H2 = 10**DRP_table['param_H2'][i]
#             M_H2_err = 10**DRP_table['param_H2_err'][i]



            
#             Mztot, Mztot_err = calculate_metal_mass(Z, Z_err, M_HI, M_HI_err, M_H2, M_H2_err, None, None)
#             print('Mztot ', Mztot, ' Mztot_err ', Mztot_err)

#             DRP_table['M_Z'][i] = Mztot
#             DRP_table['M_Z_err'][i] = Mztot_err

#             DRP_table.write(MANGA_FOLDER + 'output_files/DR17/CURRENT_MASTER_TABLE/' + 'H_alpha_HIvel_BB_extinction_H2_MxCG_R90_v3p5_Z_SFR_Portsmouthflux_Zglob.fits',
#                         format='fits',
#                         overwrite=True)


#     else:

#         print('missing masses')





else:
    print('Invalid metallicity method')


DRP_table.write(DRP_TABLE_FN + '.fits', format='fits', overwrite=True)

                                                                    
        
