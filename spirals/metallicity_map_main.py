import time
#START = datetime.datetime.now()


from astropy.table import Table
import os
from astropy.io import fits
from metallicity_map import *
from metallicity_map_broadband import *


MANGA_FOLDER = '/Users/nityaravi/Documents/Research/RotationCurves/data/manga/'
DRP_FOLDER = MANGA_FOLDER + 'DR17/'
IMAGE_DIR = '/Users/nityaravi/Documents/Research/RotationCurves/data/manga/metallicity_maps/'

DRP_TABLE_FN = MANGA_FOLDER + 'output_files/DR17/CURRENT_MASTER_TABLE/' + 'H_alpha_HIvel_BB_extinction_H2_MxCG_R90_v3p5_Z_SFR_Portsmouthflux_Zglob.fits'

corr_law = 'CCM89'

method = 'map'
#method = 'global'

#gal_ID = '7443-6103'

'''
done = ['10214-6101','11743-12705', '11744-6103', '11958-6103','12085-6101',
        '12085-9102', '12088-12705','12488-9101', '12700-12702','7977-3704',
        '7990-3703','7990-6104','8077-12704', '8082-12702', '8084-12703', '8155-12701',
        '8249-3704','8257-12701','8257-9102','8262-9102', '8313-12702','8322-3701','8330-12703',
        '8332-9102','8335-12705','8450-6102','8452-12705','8483-6101','8548-12704',
        '8551-12705','8588-12705','8588-6101','8600-1901','8601-12702','8603-12704','8603-6104',
        '8604-12702', '8604-9102','8626-3703','8712-6101','8713-6104','8713-9102','8717-3704',
        '8727-12705','8932-9102','8945-12701','8952-6104','8996-3703','9027-12701','9029-12702',
        '9041-12701','9050-3704','9195-6104','9196-6103','9485-12705','9491-6101','9501-12701','9508-6101',
        '9508-12705','9514-12704','9881-12705','9888-12704','9888-12705','8087-9102','8329-12701',
         '8466-12702','9050-9101', '8095-1902','8600-3704','8624-12703','8157-12701','8993-6104',
         '9024-1902','12769-6104','7977-3703','8552-12701','12087-12705','10842-9101','12495-6101',
         '8080-12702', '8147-12705','8262-3702','8320-9101','8455-3701','8624-12702','8979-6102',
         '8985-3701','8987-3701','9029-6102','9044-6101', '9486-12701','9494-9102','9501-12705','9871-6101',
         '9095-1901','9487-3703','8252-9101','8318-12702','8318-9101','8547-6102',
         '8592-6101','8978-3701','9486-12702','9487-9102','9493-6103','9865-9102',
         '9878-3702','8313-6102','7992-6104','8082-9102','8250-6101','8338-12701','8989-9102']


flop = ['8255-12704',]

second_pass = [,,,]

failed = [ ]

FILE_IDS = []


no_vel_fit_dont_care = ['7993-1902', '8146-1901','8311-3703','8727-3701','9036-9102','9509-3702','8982-9101']

'''
#need_to_refit = []

DRP_table = Table.read(DRP_TABLE_FN, format='fits')


if method == 'map':

    FILE_IDS = ['12088-12705',
 '12488-9101',
 '12495-6101',
'12700-12702',
 '12769-6104',
  '7977-3703',
  '7977-3704',
  '8087-9102',
  '8095-1902',
 '8147-12705',
 '8157-12701',
  '8985-3701',
  '9195-6104',
  '9493-6103',
  '9494-9102',
 '9501-12701',
 '9501-12705',
 '9514-12704',
  '9878-3702'
]

    

    for i in range(0,len(FILE_IDS)):

        gal_ID = FILE_IDS[i]

        print('processing: ', gal_ID)


        i_DRP = np.where(DRP_table['plateifu'] == gal_ID)[0][0]

        


        center_coord = (DRP_table['x0'][i_DRP], DRP_table['y0'][i_DRP])
        #center_coord = (35,35)
        phi = DRP_table['nsa_elpetro_phi'][i_DRP] 
        print('phi:', phi)
        ba = DRP_table['nsa_elpetro_ba'][i_DRP]
        #phi = DRP_table['phi'][i_DRP]
        #ba = DRP_table['ba'][i_DRP]
        z = DRP_table['nsa_z'][i_DRP]
        A_g = DRP_table['A_g'][i_DRP]
        A_r = DRP_table['A_r'][i_DRP]
        r50 = DRP_table['nsa_elpetro_th50_r'][i_DRP]


        log_M_HI = DRP_table['logHI'][i_DRP] 
        log_M_HI_err = None


        # pick mascot first - set err to none
        # if no mascot val use xcg with err
        # if none set all to none


        log_M_H2 = None
        log_M_H2_err = None

        if DRP_table['logH2_M'][i_DRP] is not None:
            log_M_H2 = DRP_table['logH2'][i_DRP]

        elif DRP_table['logH2_CG'][i_DRP] is not None:
            log_M_H2 = DRP_table['logH2_CG'][i_DRP]
            log_M_H2_err = DRP_table['logH2_CG_err'][i_DRP]

        log_M_star = DRP_table['M_disk'][i_DRP]
        log_M_star_err = DRP_table['M_disk_err'][i_DRP]





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
            grad_err = metallicity_param_outputs['grad_err']
            Z0 = metallicity_param_outputs['12logOH_0']
            Z0_err = metallicity_param_outputs['12logOH_0_err']


            DRP_table['grad_Z'][i_DRP] = grad
            DRP_table['grad_Z_err'][i_DRP] = grad_err
            DRP_table['Z_0'][i_DRP] = Z0
            DRP_table['Z_0_err'][i_DRP] = Z0_err

            

            Z, Z_err = find_global_metallicity(R25_pc, grad, grad_err, Z0, Z0_err)
            print('Z: ', Z, ' Z_err ', Z_err)
            DRP_table['Z_map'][i_DRP] = Z
            DRP_table['Z_err_map'][i_DRP] = Z_err

            

            M_HI = 10**(log_M_HI)
            M_HI_err = 0
            M_H2 = 10**log_M_H2

            if log_M_H2_err == None:
                M_H2_err = 0
            else:
                M_H2_err = 10**log_M_H2_err
            M_star = 10**log_M_star
            M_star_err = 10**log_M_star_err

            

            #if M_HI > 0 and M_H2 > 0:

        Z = DRP_table['Z_map'][i_DRP]
        Z_err= DRP_table['Z_err_map'][i_DRP]
        M_HI = 10**(DRP_table['logHI'][i_DRP])
        M_HI_err = None
        M_H2 = 10**(DRP_table['logH2_CG'][i_DRP])
        M_H2_err = 10**(DRP_table['logH2_CG_err'][i_DRP])


        Mz, Mz_err = calculate_metal_mass(Z, Z_err, M_HI, M_HI_err, M_H2, M_H2_err, None, None)

        print('Mz: ', Mz)
        print('Mz_err: ', Mz_err)
        DRP_table['M_Z_map'][i_DRP] = Mz
        DRP_table['M_Z_err_map'][i_DRP] = Mz_err


        DRP_table.write(DRP_TABLE_FN, format='fits', overwrite=True)

elif method == 'global':

    #for i in range(0, len(DRP_table)):
    for i in range(0, len(DRP_table)):


        gal_ID = DRP_table['plateifu'][i]

        print('Processing ', gal_ID)


        fluxes = {'OII': DRP_table['Flux_OII_3726'][i],
                    'OII_err': DRP_table['Flux_OII_3726_Err'][i],
                    'OII2': DRP_table['Flux_OII_3728'][i],
                    'OII2_err': DRP_table['Flux_OII_3728_Err'][i], 
                    'NII': DRP_table['Flux_NII_6547'][i],
                    'NII_err': DRP_table['Flux_NII_6547_Err'][i],
                    'NII2': DRP_table['Flux_NII_6583'][i],
                    'NII2_err': DRP_table['Flux_NII_6583_Err'][i],
                    'OIII': DRP_table['Flux_OIII_4958'][i],
                    'OIII_err': DRP_table['Flux_OIII_4958_Err'][i],
                    'OIII2': DRP_table['Flux_OIII_5006'][i],
                    'OIII2_err': DRP_table['Flux_OIII_5006_Err'][i],
                    'Ha' : DRP_table['Flux_Ha_6562'][i],
                    'Ha_err': DRP_table['Flux_Ha_6562_Err'][i],
                    'Hb': DRP_table['Flux_Hb_4861'][i],
                    'Hb_err': DRP_table['Flux_Hb_4861_Err'][i]
                    }

        Z, Z_err = calculate_global_metallicity(fluxes)
        print(Z, Z_err)

        DRP_table['Z'][i] = Z
        DRP_table['Z_err'][i] = Z_err 

        if i % 100 == 0:
            DRP_table.write(MANGA_FOLDER + 'output_files/DR17/CURRENT_MASTER_TABLE/' + 'H_alpha_HIvel_BB_extinction_H2_MxCG_R90_v3p5_Z_SFR_Portsmouthflux_Zglob.fits',
                    format='fits',
                    overwrite=True)



        if Z > 0 and DRP_table['logHI'][i] > 0 and DRP_table['param_H2'][i] > 0:

            M_HI = 10**DRP_table['logHI'][i]
            M_HI_err = 0
            M_H2 = 10**DRP_table['param_H2'][i]
            M_H2_err = 10**DRP_table['param_H2_err'][i]



            
            Mztot, Mztot_err = calculate_metal_mass(Z, Z_err, M_HI, M_HI_err, M_H2, M_H2_err, None, None)
            print('Mztot ', Mztot, ' Mztot_err ', Mztot_err)

            DRP_table['M_Z'][i] = Mztot
            DRP_table['M_Z_err'][i] = Mztot_err

            DRP_table.write(MANGA_FOLDER + 'output_files/DR17/CURRENT_MASTER_TABLE/' + 'H_alpha_HIvel_BB_extinction_H2_MxCG_R90_v3p5_Z_SFR_Portsmouthflux_Zglob.fits',
                        format='fits',
                        overwrite=True)


    else:

        print('missing masses')





else:
    print('Invalid metallicity method')

DRP_table.write(MANGA_FOLDER + 'output_files/DR17/CURRENT_MASTER_TABLE/' + 'H_alpha_HIvel_BB_extinction_H2_MxCG_R90_v3p5_Z_SFR_Portsmouthflux_Zglob.fits',
                format='fits',
                overwrite=True)


                                                                    
        
