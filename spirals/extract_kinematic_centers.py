import numpy as np
from astropy.table import Table
from astropy.io import fits




DATA_DIRECTORY = '/scratch/kdougla7/data/SDSS/dr17/manga/spectro/analysis/v3_1_1/3.1.0/HYB10-MILESHC-MASTARSSP/'
drp_fn = '/scratch/nravi3/Elliptical_sphdisk_refitspirals_BPT_illustris_v10.fits'

SAVE_FN = '/scratch/nravi3/manga_kinematic_centers.fits'

drp_table = Table.read(drp_fn)
spirals = drp_table[drp_table['spiral_mask'] == 1]
spirals['kin_center_ra'] = np.ones(len(spirals))*np.nan
spirals['kin_center_dec'] = np.ones(len(spirals))*np.nan

for i in range(len(spirals)):

    gal_ID = spirals['plateifu'][i]

    # get maps

    plate, ifu = gal_ID.split('-')

    cube_fn = DATA_DIRECTORY + plate + '/' + ifu + '/manga-' + gal_ID + '-MAPS-HYB10-MILESHC-MASTARSSP.fits.gz'
    cube = fits.open(cube_fn)
    skyco_x = cube['SPX_SKYCOO'].data[0]
    skyco_y = cube['SPX_SKYCOO'].data[1]
    cube.close()

    # get phot gal center

    objra = spirals['objra'][i]
    objdec = spirals['objdec'][i]

    # get kin center from fit

    x0 = spirals['x0'][i]
    y0 = spirals['y0'][i]

    # get kin center offset

    delta_ra = skyco_x[int(x0)][int(y0)]
    delta_dec = skyco_y[int(x0)][int(y0)]

    # get kin ra dec in degrees
    kin_ra = objra + delta_ra / 3600
    kin_dec = objdec + delta_dec/ 3600  

    spirals['kin_center_ra'][i] = kin_ra
    spirals['kin_center_dec'][i] = kin_dec

    if i % 100 == 0:
        print(i)

spirals = spirals['plateifu', 'objra', 'objdec', 'kin_center_ra', 'kin_center_dec']
spirals.write(SAVE_FN, format='fits', overwrite=True)