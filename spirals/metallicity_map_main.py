import time
#START = datetime.datetime.now()


from astropy.table import Table
import os
from astropy.io import fits
from metallicity_map import *

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


DRP_FOLDER = '/Users/nityaravi/Documents/Research/RotationCurves/data/manga/DR17/'
IMAGE_DIR = '/Users/nityaravi/Documents/Research/RotationCurves/data/manga/metallicity_maps/'
corr_law = 'K76'

get_metallicity_map(DRP_FOLDER, IMAGE_DIR, corr_law, '8082-9102')