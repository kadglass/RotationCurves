# modified from jupyter notebook from Nate Brunacini, nbrunaci@u.rochester.edu
# original repo at https://github.com/kadglass/Metallicity_gradients

#import marvin
#from marvin.tools.maps import Maps
#from marvin.tools import Image
#from marvin import config
#config.setRelease('DR17')

import numpy as np
import numpy.ma as ma
from numpy import log10, pi

import matplotlib.pyplot as plt

from astropy.io import fits
import astropy.constants as const

import math

import pyneb as pn


################################################################################
################################################################################
################################################################################


def extract_metallicity_data(DRP_FOLDER, gal_ID):
    '''
    Open the MaNGA .fits file and extract data.

    PARAMETERS
    ==========

    DRP_FOLDER : string
        Address to location of DRP data on computer system

    gal_ID : string
        '[PLATE]-[IFUID]' of the galaxy


    RETURNS
    =======

    maps : dictionary
        Dictionary of maps.  includes:
          - Ha_flux, _ivar: H-alpha flux [1e-17 erg/s/cm^2/ang/spaxel]
          - Hb_flux, _ivar: H-beta flux [1e-17 erg/s/cm^2/ang/spaxel]
          - OII_flux, _ivar: [OII] 3728 flux [1e-17 erg/s/cm^2/ang/spaxel]
          - OII2_flux, _ivar: [OII] 3729 flux [1e-17 erg/s/cm^2/ang/spaxel]
          - OIII_flux, _ivar: [OIII] 4960 flux [1e-17 erg/s/cm^2/ang/spaxel]
          - OIII2_flux, _ivar: [OIII] 5008 flux [1e-17 erg/s/cm^2/ang/spaxel]
          - NII_flux, _ivar: [NII] 6549 flux [1e-17 erg/s/cm^2/ang/spaxel]
          - NII2_flux, _ivar: [NII] 6585 flux [1e-17 erg/s/cm^2/ang/spaxel]

    '''
    
    

    [plate, IFU] = gal_ID.split('-')

    # for dr17:
    file_name = DRP_FOLDER + '/manga-' + gal_ID + '-MAPS-HYB10-MILESHC-MASTARSSP.fits.gz'

    if not os.path.isfile(file_name):
        print(gal_ID, 'data file does not exist.')
        return None

    cube = fits.open(file_name)

    maps = {}

    # H-alpha maps

    maps['Ha_flux'] = cube['EMLINE_GFLUX'].data[23]
    maps['Ha_flux_ivar'] = cube['EMLINE_GFLUX_IVAR'].data[23]
    maps['Ha_flux_mask'] = cube['EMLINE_GFLUX_MASK'].data[23]

    # H-beta maps

    maps['Hb_flux'] = cube['EMLINE_GFLUX'].data[14] 
    maps['Hb_flux_ivar'] = cube['EMLINE_GFLUX_IVAR'].data[14]
    maps['Hb_flux_mask'] = cube['EMLINE_GFLUX_MASK'].data[14]

    # [OII] maps

    maps['OII_flux'] = cube['EMLINE_GFLUX'].data[0] 
    maps['OII_flux_ivar'] = cube['EMLINE_GFLUX_IVAR'].data[0]
    maps['OII_flux_mask'] = cube['EMLINE_GFLUX_MASK'].data[0]

    # [OII]2 maps

    maps['OII2_flux'] = cube['EMLINE_GFLUX'].data[1] 
    maps['OII2_flux_ivar'] = cube['EMLINE_GFLUX_IVAR'].data[1]
    maps['OII2_flux_mask'] = cube['EMLINE_GFLUX_MASK'].data[1]

    # [OIII] maps

    maps['OIII_flux'] = cube['EMLINE_GFLUX'].data[15] 
    maps['OIII_flux_ivar'] = cube['EMLINE_GFLUX_IVAR'].data[15]
    maps['OIII_flux_mask'] = cube['EMLINE_GFLUX_MASK'].data[15]

    # [OIII]2 maps

    maps['OIII2_flux'] = cube['EMLINE_GFLUX'].data[16] 
    maps['OIII2_flux_ivar'] = cube['EMLINE_GFLUX_IVAR'].data[16]
    maps['OIII2_flux_mask'] = cube['EMLINE_GFLUX_MASK'].data[16]

    # [NII] maps

    maps['NII_flux'] = cube['EMLINE_GFLUX'].data[22] 
    maps['NII_flux_ivar'] = cube['EMLINE_GFLUX_IVAR'].data[22]
    maps['NII_flux_mask'] = cube['EMLINE_GFLUX_MASK'].data[22]

    # [NII]2 maps

    maps['NII2_flux'] = cube['EMLINE_GFLUX'].data[24] 
    maps['NII2_flux_ivar'] = cube['EMLINE_GFLUX_IVAR'].data[24]
    maps['NII2_flux_mask'] = cube['EMLINE_GFLUX_MASK'].data[24]


    '''
    # Observed emission lines
    HbF_map = maps["emline_gflux_hb_4862"]
    OII_map = maps["emline_gflux_oii_3727"]
    OIII_map = maps["emline_gflux_oiii_5008"]
    OIII2_map = maps["emline_gflux_oiii_4960"]
    NII_map = maps["emline_gflux_nii_6585"]
    NII2_map = maps["emline_gflux_nii_6549"]
    HaF_map = maps["emline_gflux_ha_6564"]
    OII2_map = maps["emline_gflux_oii_3729"]
    observed = [HbF_map,OII_map,OII2_map,OIII_map,OIII2_map,NII_map,NII2_map,HaF_map,]# Array of observed emission lines
    names = ['HbF','OII','OII2','OIII','OIII2','NII','NII2','HaF']# Array of names of observed emission lines
    
    # Array of masked flux arrays for observed emission lines
    observed_m = []
    for line in observed:
        observed_m.append(line.masked)
    
    # Array of masked inverse variance arrays for observed emission lines
    observed_ivar_m = []
    for line in observed:
        observed_ivar_m.append(ma.array(line.ivar,mask=line.mask))
    '''

    '''
    maps = {
        names[0]: {'flux': observed_m[0], 'ivar': observed_ivar_m[0]},
        names[1]: {'flux': observed_m[1], 'ivar': observed_ivar_m[1]},
        names[2]: {'flux': observed_m[2], 'ivar': observed_ivar_m[2]},
        names[3]: {'flux': observed_m[3], 'ivar': observed_ivar_m[3]},
        names[4]: {'flux': observed_m[4], 'ivar': observed_ivar_m[4]},
        names[5]: {'flux': observed_m[5], 'ivar': observed_ivar_m[5]},
        names[6]: {'flux': observed_m[6], 'ivar': observed_ivar_m[6]},
        names[7]: {'flux': observed_m[7], 'ivar': observed_ivar_m[7]},
        'wavelength': [4862,3727,5008,4960,6585,6549,6564,3729]# Wavelength of each line
    }
    '''

    cube.close()

    return maps

################################################################################
################################################################################
################################################################################


def calc_metallicity(R2, N2, R3):

    '''
    Takes R2, N2, and R3 ratio maps and calculates a metallicity map


    PARAMETERS
    ==========
    
    R2 : array
        [OII] doublet / H beta flux mao
    
    N2 : array
        [NII] doublet / H beta flux map

    R3 : array
        [OIII] doublet / H beta flux map

    RETURNS
    =======

    z : array
        metallicity map

    '''

    
    z = np.ones((len(N2),len(N2)[0]))*np.nan

    for i in range(0, len(z)):
        for j in range(0, len(z)[0]):

        # upper branch
        if ma.log10(N2[i][j]) >= -0.6:
            z[i][j] = 8.589 + 0.022*ma.log10(R3[i][j]/R2[i][j]) + 0.399*ma.log10(N2[i][j]) \ 
                + (-0.137 + 0.164*ma.log10(R3[i][j]/R2[i][j]) + 0.589*ma.log10(N2[i][j]))*ma.log10(N2[i][j])
    
        # lower branch
        elif ma.log10(N2[i][j]) < -0.6:
            z[i][j] = 7.932 + 0.944*ma.log10(R3[i][j]/R2[i][j]) + 0.695*ma.log10(N2[i][j]) \
                + (0.970 - 0.291*ma.log10(R3[i][j]/R2[i][j]) - 0.019*ma.log10(N2[i][j]))*ma.log(R2[i][j])

    return z


################################################################################
################################################################################
################################################################################


def mask_AGN(OIII2, Hb, Ha, NII2):
    
    '''

    Takes [OIII] 5007, [NII] 6584, H-beta, H-alpha maps and creates mask for AGN-like regions
    using eq. 1 from Kauffmann et al. 2003


    PARAMETERS
    ==========

    OIII2 : array
        [OIII] 5007 flux map
    
    NII2 : array
        [NII] 6584 flux map
    
    Hb : array
        H-beta flux map

    Ha : array
        H-alpha flux map

    


    RETURNS
    =======

    mask : array
        AGN spaxel mask

    '''



    mask = ma.log10(OIII2/Hb) > 0.61 / (ma.log10(NII2/Ha) - 0.05) + 1.3

    return mask

################################################################################
################################################################################
################################################################################


def dustCorrection(maps):

    '''

    Takes flux map and applies default mask and dust correction

    PARAMETERS
    ==========

    map : array
        flux map

    RETURNS
    =======

    corrected_map : array
        dust corrected flux map and inverse variance map

    '''

    rc = pn.RedCorr(law='CCM89')

    H_ratio = maps['mHa'] / maps['mHb']

    rc.setCorr(H_ratio / 2.86, 6563, 4861)  # is H-a 6563 or 6564





