import numpy as np
from astropy.table import Table

from metallicity_map import fit_stellar_metallicity_map_gradient


MANGA_FOLDER = '/global/cfs/cdirs/sdss/data/sdss/dr17/manga/'


DRP_FOLDER = '/scratch/kdougla7/data/SDSS/dr17/manga/spectro/analysis/v3_1_1/3.1.0/HYB10-MILESHC-MASTARSSP/'
P3D_FOLDER =    '/scratch/kdougla7/data/SDSS/dr17/manga/spectro/pipe3d/'


DRP_TABLE_FN = '/scratch/nravi3/Elliptical_sphdisk_refitspirals_BPT_illustris_v11_gradZ_stelZ'


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
    DRP_table['stellar_Z0_1kpc'] = np.ones(len(DRP_table))*np.nan
    DRP_table['stellar_Z0_0p5kpc'] = np.ones(len(DRP_table))*np.nan

    DRP_table['gas_Z0_1kpc'] = np.ones(len(DRP_table))*np.nan
    DRP_table['gas_Z0_0p5kpc'] = np.ones(len(DRP_table))*np.nan

