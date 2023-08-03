import time
#START = datetime.datetime.now()


from astropy.table import Table
import os
from astropy.io import fits
from metallicity_map import *
from metallicity_map_broadband_functions import *

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
corr_law = 'CCM89'

# check if target is a galaxy
# want to choose center_coord, phi, inclination from fit if available
# else choose nsa values (conversion necessary for phi, inclination)

center_coord = (26.348064449203378, 24.578296983961433)
phi = 201.61168900325225
ba = 0.5187071826939094
z = 0.0183422
A_g = 0.082
A_r = 0.057
gal_ID = '7443-6103'


#master_table=Table.read(master_fn, format='fits')

#i_master = np.where(master_table['plateifu'] == gal_ID)[0][0]

#A_g = master_table['A_g'][i_master]
#A_r = master_table['A_r'][i_master]

metallicity_param_outputs = fit_metallicity_gradient(DRP_FOLDER, 
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
                                                                        center_coord,
                                                                        phi, 
                                                                        ba,
                                                                        z)

                                                                
    
#    if surface_brightness_param_outputs is not None:

#        R25, R25_err = surface_brightness_param_outputs['R25'], surface_brightness_param_outputs['R25_err'] 

#        Z_12logOH, Z_12logOH_err = linear_metallicity_gradient_sigma(R25, R25_err, MOREPARAMS)

