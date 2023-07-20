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
import os


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
          - OII_flux, _ivar: [OII] 3727 flux [1e-17 erg/s/cm^2/ang/spaxel]
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

    wavelengths=np.array([6564, 4862, 3727, 3729, 4960, 5008, 6549, 6585])



    cube.close()

    return maps, wavelengths

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

    
    z = np.ones((len(N2),len(N2[0])))*np.nan

    for i in range(0, len(z)):
        for j in range(0, len(z[0])):

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


def mask_AGN(dmaps):
    
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

    maps : array
        AGN spaxel mask applied to maps

    '''

    OIII2 = dmaps['dmOIII2_flux']
    Hb = dmaps['dmHb_flux']
    NII2 = dmaps['dmNII2_flux']
    Ha = dmaps['dmHa_flux']

    AGN_mask = ma.log10(OIII2/Hb) > 0.61 / (ma.log10(NII2/Ha) - 0.05) + 1.3

    #for m in enum


    for m in dmaps:
        dmaps[m] = ma.array(dmaps[m], mask=dmaps[m].mask + AGN_mask)

    return dmaps

################################################################################
################################################################################
################################################################################


def dust_correction(maps, wavelengths, corr_law='CCM89'):

    '''

    Takes flux maps and applies default mask and dust correction

    PARAMETERS
    ==========

    maps : array
        flux maps

    wavelengths : array
        list of wavelengths
    
    corr_law : string
        correction law to be used, CCM89 by default

    RETURNS
    =======

    maps : array
        maps dictionary with new dust corrected maps

    '''

    # set correction law

    rc = pn.RedCorr(law=corr_law)

    # create masked flux and ivar arrays. since Ha, Hb are used for correction, 
    # minimum mask is Ha + Hb mask

    maps['mHa_flux'] = ma.array(maps['Ha_flux'], mask=maps['Ha_flux_mask'] + maps['Hb_flux_mask'])
    maps['mHa_flux_ivar'] = ma.array(maps['Ha_flux_ivar'], mask=maps['Ha_flux_mask'] + maps['Hb_flux_mask'])

    maps['mHb_flux'] = ma.array(maps['Hb_flux'], mask=maps['Ha_flux_mask'] + maps['Hb_flux_mask'])
    maps['mHb_flux_ivar'] = ma.array(maps['Hb_flux_ivar'], mask=maps['Ha_flux_mask'] + maps['Hb_flux_mask'])

    maps['mOII_flux'] = ma.array(maps['OII_flux'], mask=maps['OII_flux_mask'] + maps['Ha_flux_mask'] + maps['Hb_flux_mask'])
    maps['mOII_flux_ivar'] = ma.array(maps['OII_flux_ivar'], mask=maps['OII_flux_mask'] + maps['Ha_flux_mask'] + maps['Hb_flux_mask'] )

    maps['mOII2_flux'] = ma.array(maps['OII2_flux'], mask=maps['OII2_flux_mask'] + maps['Ha_flux_mask'] + maps['Hb_flux_mask'])
    maps['mOII2_flux_ivar'] = ma.array(maps['OII2_flux_ivar'], mask=maps['OII2_flux_mask'] + maps['Ha_flux_mask'] + maps['Hb_flux_mask'])

    maps['mOIII_flux'] = ma.array(maps['OIII_flux'], mask=maps['OIII_flux_mask'] + maps['Ha_flux_mask'] + maps['Hb_flux_mask'])
    maps['mOIII_flux_ivar'] = ma.array(maps['OIII_flux_ivar'], mask=maps['OIII_flux_mask'] + maps['Ha_flux_mask'] + maps['Hb_flux_mask'])

    maps['mOIII2_flux'] = ma.array(maps['OIII2_flux'], mask=maps['OIII2_flux_mask'] + maps['Ha_flux_mask'] + maps['Hb_flux_mask'])
    maps['mOIII2_flux_ivar'] = ma.array(maps['OIII2_flux_ivar'], mask=maps['OIII2_flux_mask'] + maps['Ha_flux_mask'] + maps['Hb_flux_mask'])

    maps['mNII_flux'] = ma.array(maps['NII_flux'], mask=maps['NII_flux_mask'] + maps['Ha_flux_mask'] + maps['Hb_flux_mask'])
    maps['mNII_flux_ivar'] = ma.array(maps['NII_flux_ivar'], mask=maps['NII_flux_mask'] + maps['Ha_flux_mask'] + maps['Hb_flux_mask'])

    maps['mNII2_flux'] = ma.array(maps['NII2_flux'], mask=maps['NII2_flux_mask'] + maps['Ha_flux_mask'] + maps['Hb_flux_mask'])
    maps['mNII2_flux_ivar'] = ma.array(maps['NII2_flux_ivar'], mask=maps['NII2_flux_mask'] + maps['Ha_flux_mask'] + maps['Hb_flux_mask'])

    # set the correction coefficient based on H-alpha and H-beta maps

    H_ratio = ma.array(maps['mHa_flux'] / maps['mHb_flux'], mask=maps['Ha_flux_mask'] + maps['Hb_flux_mask'] + (maps['mHa_flux'] / maps['mHb_flux'] ==0))

    H_ratio[H_ratio.mask] = np.nan

    rc.setCorr(H_ratio / 2.86, 6564, 4862)

    

    # deredden

    dmaps = {}

    #corr = np.ones(len(wavelengths))
    #for i in range(0, len(wavelengths)):
    #    corr[i] = rc.getCorrHb(wavelengths[i])

    wavelengths=np.array([6564, 4862, 3727, 3729, 4960, 5008, 6549, 6585])

    corr = rc.getCorrHb(wavelengths)



    dmaps['dmHa_flux'] = maps['mHa_flux'] * corr[0]
    dmaps['dmHa_flux_ivar'] = maps['mHa_flux_ivar'] / corr[0]**2

    dmaps['dmHb_flux'] = maps['mHb_flux'] * corr[1]
    dmaps['dmHb_flux_ivar'] = maps['mHb_flux_ivar'] / corr[1]**2

    dmaps['dmOII_flux'] = maps['mOII_flux'] * corr[2]
    dmaps['dmOII_flux_ivar'] = maps['mOII_flux_ivar'] / corr[2]**2

    dmaps['dmOII2_flux'] = maps['mOII2_flux'] * corr[3]
    dmaps['dmOII2_flux_ivar'] = maps['mOII2_flux_ivar'] / corr[3]**2

    dmaps['dmOIII_flux'] = maps['mOIII_flux'] * corr[4]
    dmaps['dmOIII_flux_ivar'] = maps['mOIII_flux_ivar'] / corr[4]**2

    dmaps['dmOIII2_flux'] = maps['mOIII_flux'] * corr[5]
    dmaps['dmOIII2_flux_ivar'] = maps['mOIII_flux_ivar'] / corr[5]**2

    dmaps['dmNII_flux'] = maps['mNII_flux'] * corr[6]
    dmaps['dmNII_flux_ivar'] = maps['mNII_flux_ivar'] / corr[6]**2

    dmaps['dmNII2_flux'] = maps['mNII2_flux'] * corr[7]
    dmaps['dmNII2_flux_ivar'] = maps['mNII2_flux_ivar'] / corr[7]**2

    return dmaps






################################################################################
################################################################################
################################################################################

def calc_metallicity_ratios(maps):
    return

################################################################################
################################################################################
################################################################################

def plot_metallicity_map(IMAGE_DIR, metallicity_map):

    plt.imshow(metallicity_map, vmin=8,vmax=9)
    plt.gca().invert_yaxis()
    plt.title(gal_ID)
    plt.xlabel('spaxel')
    plt.ylabel('spaxel')
    plt.colorbar(label='12+log(O/H) (dex)')
    plt.savefig(IMAGE_DIR + gal_ID + '_metallicity_map.eps')
    plt.close()


################################################################################
################################################################################
################################################################################


def get_metallicity_map(DRP_FOLDER, IMAGE_DIR, corr_law, gal_ID):

    # extract metallicity maps
    maps, wavelengths = extract_metallicity_data(DRP_FOLDER, gal_ID)

    # apply default mask + dust correction
    dmaps = dust_correction(maps, wavelengths, corr_law='CCM89')

    # mask AGN
    AGN_masked_maps = mask_AGN(dmaps)
    #AGN_masked_maps = dmaps


    # calculate metallicity ratios

    R2 = (AGN_masked_maps['dmOII_flux'] + AGN_masked_maps['dmOII2_flux']) / AGN_masked_maps['dmHb_flux']
    N2 = (AGN_masked_maps['dmNII_flux'] + AGN_masked_maps['dmNII2_flux']) / AGN_masked_maps['dmHb_flux']
    R3 = (AGN_masked_maps['dmOIII_flux'] + AGN_masked_maps['dmOIII2_flux']) / AGN_masked_maps['dmHb_flux']

    # calculate metallicity map

    metallicity_map = calc_metallicity(R2,N2,R3)

    # plot maps and save figures

    plt.imshow(metallicity_map, vmin=8,vmax=9)
    plt.gca().invert_yaxis()
    plt.title(gal_ID)
    plt.xlabel('spaxel')
    plt.ylabel('spaxel')
    plt.colorbar(label='12+log(O/H) (dex)')
    plt.savefig(gal_ID + '_metallicity_map.eps')
    plt.close()

    #return metallicity_map, metallicity_sigma_map