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

from scipy.optimize import curve_fit

from DRP_vel_map_functions import deproject_spaxel
from metallicity_map_plottingFunctions import plot_metallicity_map, plot_metallicity_gradient
from metallicity_map_functions import linear_metallicity_gradient



################################################################################
################################################################################
################################################################################



q0 = 0.2 #nominal disk thickness
ln10 = ma.log(10)
H_0 = 100      # Hubble's Constant in units of h km/s/Mpc
c = 299792.458 # Speed of light in units of km/s
MANGA_SPAXEL_SIZE = 0.5*(1/60)*(1/60)*(np.pi/180)  # spaxel size (0.5") in radians


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
    file_name = DRP_FOLDER +  '/' + plate + '/' + IFU + '/manga-' + gal_ID + '-MAPS-HYB10-MILESHC-MASTARSSP.fits.gz'

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


def calc_metallicity(R2, R2_ivar, N2, N2_ivar, R3, R3_ivar):

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
    dydR2 = np.ones((len(N2),len(N2[0])))*np.nan
    dydN2 = np.ones((len(N2),len(N2[0])))*np.nan
    dydR3 = np.ones((len(N2),len(N2[0])))*np.nan




    for i in range(0, len(z)):
        for j in range(0, len(z[0])):

        # upper branch
            if ma.log10(N2[i][j]) >= -0.6:
                z[i][j] = 8.589 + 0.022*ma.log10(R3[i][j]/R2[i][j]) + 0.399*ma.log10(N2[i][j]) \
                    + (-0.137 + 0.164*ma.log10(R3[i][j]/R2[i][j]) + 0.589*ma.log10(N2[i][j]))*ma.log10(N2[i][j])

                dydR2[i][j] = -0.022/(R2[i][j] * ln10) - 0.164*R2[i][j]*ma.log10(R2[i][j])/ln10 \
                    + (-0.137 + 0.164*ma.log10(R3[i][j]/R2[i][j]) + 0.589*ma.log10(N2[i][j]))*(1/(R2[i][j] * ln10))

                dydR3[i][j] = 0.022/(R3[i][j]*ln10) + 0.164*ma.log10(R3[i][j])/(R3[i][j]*ln10)

                dydN2[i][j] = 0.399/(N2[i][j]*ln10) + 0.589*ma.log10(R2[i][j])/(N2[i][j]*ln10)
                

    
        # lower branch
            elif ma.log10(N2[i][j]) < -0.6:
                z[i][j] = 7.932 + 0.944*ma.log10(R3[i][j]/R2[i][j]) + 0.695*ma.log10(N2[i][j]) \
                    + (0.970 - 0.291*ma.log10(R3[i][j]/R2[i][j]) - 0.019*ma.log10(N2[i][j]))*ma.log(R2[i][j])

                dydR2[i][j] = -0.944/(R2[i][j] * ln10) +0.291*R2[i][j]*ma.log10(R2[i][j])/ln10 \
                    + (0.970 - 0.291*ma.log10(R3[i][j]/R2[i][j]) - 0.019*ma.log10(N2[i][j]))*(1/(R2[i][j] * ln10))

                dydR3[i][j] = 0.944/(R3[i][j]*ln10) - 0.291*ma.log10(R3[i][j])/(R3[i][j]*ln10)

                dydN2[i][j] = 0.695/(N2[i][j]*ln10) - 0.019*ma.log10(R2[i][j])/(N2[i][j]*ln10)

    z_ivar = 1 / (dydR2**2/R2_ivar + dydN2**2 / N2_ivar + dydR3**2 / R3_ivar)


  

    return z,z_ivar


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

def calc_intensity_ratio(maps, line):

    '''
    Calculates the doublet line intensity ratio with H-beta and their uncertainties

    PARAMETERS
    ==========

    maps : dictionary
        maps dictionary containing flux maps

    line : string
        emission lines to calculate the ratio with H-beta

    RETURNS
    =======

    y : array
        map of line intensity ratio with H-beta

    y_ivar : array
        inverse variance of y




    '''

    x1 = maps['dm' + line + '_flux']
    x1_ivar = maps['dm' + line + '_flux_ivar']

    x2 = maps['dm' + line + '2_flux']
    x2_ivar = maps['dm' + line + '2_flux_ivar']

    x3 = maps['dmHb_flux']
    x3_ivar = maps['dmHb_flux_ivar']

    y = (x1 + x2) / x3
    
    y_ivar = x3**2 / ((1 / x1_ivar) + (1 / x2_ivar) + (y**2 / x3_ivar))

    return y, y_ivar



################################################################################
################################################################################
################################################################################


def get_metallicity_map(DRP_FOLDER, IMAGE_DIR, corr_law, gal_ID):

    # extract metallicity maps
    maps, wavelengths = extract_metallicity_data(DRP_FOLDER, gal_ID)

    # apply default mask + dust correction
    dmaps = dust_correction(maps, wavelengths, corr_law)

    # mask AGN
    AGN_masked_maps = mask_AGN(dmaps)


    # calculate metallicity ratios

    R2, R2_ivar = calc_intensity_ratio(AGN_masked_maps, 'OII')
    N2, N2_ivar = calc_intensity_ratio(AGN_masked_maps, 'NII')
    R3, R3_ivar = calc_intensity_ratio(AGN_masked_maps, 'OIII')

    # calculate metallicity map

    metallicity_map, metallicity_map_ivar = calc_metallicity(R2, 
                                                            R2_ivar, 
                                                            N2, 
                                                            N2_ivar, 
                                                            R3, 
                                                            R3_ivar)

    # plot maps and save figures

    plot_metallicity_map(IMAGE_DIR, metallicity_map, metallicity_map_ivar, gal_ID)

    return metallicity_map, metallicity_map_ivar

    





################################################################################
################################################################################
################################################################################

def fit_metallicity_gradient(   DRP_FOLDER, 
                                IMAGE_DIR, 
                                corr_law, 
                                gal_ID,
                                center_coord,
                                phi,
                                ba,
                                z):

    

    # scaling constants

    dist_to_galaxy_Mpc = c*z/H_0
    dist_to_galaxy_kpc = dist_to_galaxy_Mpc*1000
    pix_scale_factor = dist_to_galaxy_kpc*np.tan(MANGA_SPAXEL_SIZE)
    
    # get metallicity map and uncertainty


    metallicity_map, metallicity_map_ivar = get_metallicity_map(DRP_FOLDER, 
                                                                IMAGE_DIR, 
                                                                corr_law,
                                                                gal_ID)


    # deproject maps

    cosi2 = (ba**2 - q0**2)/(1 - q0**2)
    i_angle = np.arccos(np.sqrt(cosi2))

    r_kpc = np.zeros((len(metallicity_map), len(metallicity_map[0])))

    for i in range(len(metallicity_map)):
        for j in range(len(metallicity_map[0])):

            r_spax, _ = deproject_spaxel((i,j), center_coord, phi, i_angle)
            r_kpc[i][j] = r_spax*pix_scale_factor


    # apply linear fit

    nan_mask = np.isnan(metallicity_map)

    r_flat = ma.array(r_kpc, mask=nan_mask).compressed()
    m = ma.array(metallicity_map, mask=nan_mask).compressed()
    m_sigma = ma.array(1/ma.sqrt(metallicity_map_ivar), mask=nan_mask).compressed()


    popt, pcov = curve_fit(linear_metallicity_gradient, 
                                    r_flat, 
                                    m,
                                    sigma=m_sigma
                                    )

    
    np.save('metallicity_' + gal_ID + '_cov.npy', pcov) # for bluehive

    perr = np.sqrt(np.diag(pcov))

    best_fit_values = {'grad': popt[0], 
                        'grad_err': perr[0], 
                        '12logOH_0': popt[1], 
                        '12logOH_0_err': perr[1]}


    plot_metallicity_gradient(IMAGE_DIR, gal_ID, r_flat, m, m_sigma, popt)


    return best_fit_values