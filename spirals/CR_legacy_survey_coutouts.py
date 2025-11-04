
from astropy.table import Table
import os

from CR_plotting_functions import  plot_legacy_survey_VI

import sys
sys.path.insert(1, '/Users/nityaravi/Documents/GitHub/RotationCurves/')

from GenerateCutout import get_cutout




CACHE_DIR = '/Users/nityaravi/Documents/Research/RotationCurves/data/manga/legacy_survey/'
drp_table = Table.read('/Users/nityaravi/Documents/Research/RotationCurves/data/manga/DR17/drpall-v3_1_1.fits', format='fits', hdu=1)
nsa = Table.read('/Users/nityaravi/Documents/Research/RotationCurves/data/nsa_v1_0_1.fits', format='fits')

drp_table = drp_table[1283:]
RUN_ALL_GALAXIES = True

FILE_IDS = []

drp_dict = {}
for i in range(len(drp_table)):

    drp_dict[drp_table['plateifu'][i]] = i


nsa_dict = {}
for i in range(len(nsa)):

    nsa_dict[nsa['IAUNAME'][i]] = i 

if RUN_ALL_GALAXIES:
    FILE_IDS = drp_table['plateifu']



count = 0

for gal_ID in FILE_IDS:

    i_DRP = drp_dict[gal_ID]
    count += 1

    if count % 10 == 0:
        print(count)

    # if object is a galaxy target

    if drp_table['mngtarg1'][i_DRP] > 0:

        try:
            ra = drp_table['objra'][i_DRP]
            dec = drp_table['objdec'][i_DRP]
            # r90 = drp_table['nsa_elpetro_th90'][i_DRP]
            phi = drp_table['nsa_elpetro_phi'][i_DRP]
            ba = drp_table['nsa_elpetro_ba'][i_DRP]

            nsa_iauname = drp_table['nsa_iauname'][i_DRP]
            i_nsa = nsa_dict[nsa_iauname]
            r90 = nsa['ELPETRO_TH90_R'][i_nsa]

        
            # get and save cutouts
            _, w = get_cutout(gal_ID, ra, dec, r90, cache_dir=CACHE_DIR, 
                    layers=['default', 'model', 'resid'])
            
            # create and save figures


            plot_legacy_survey_VI(gal_ID, ra, dec, phi, r90, ba,
                                w, CACHE_DIR)
            
            os.remove(CACHE_DIR + '/default/' + gal_ID + '.jpg')
            os.remove(CACHE_DIR + '/model/' + gal_ID + '.jpg')
            os.remove(CACHE_DIR + '/resid/' + gal_ID + '.jpg')

        except:
            print(gal_ID, ' failed')
            continue


    