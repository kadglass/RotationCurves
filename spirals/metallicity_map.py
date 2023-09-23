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

import numdifftools as ndt


from astropy.io import fits
import astropy.constants as const

import math

import pyneb as pn
import os

from scipy.optimize import curve_fit, minimize

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
fHep = 0.2485 # primordial Helium fraction
grad_Hep_z = 1.41 


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


    np.seterr(divide='ignore')

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

    maps['mHa_flux'] = ma.array(maps['Ha_flux'], 
                                mask=np.logical_or(maps['Ha_flux_mask'], 
                                    np.abs(maps['Ha_flux']*np.sqrt(maps['Ha_flux_ivar']) < 3)))
    maps['mHa_flux_ivar'] = ma.array(maps['Ha_flux_ivar'], mask=maps['mHa_flux'].mask)

    maps['mHb_flux'] = ma.array(maps['Hb_flux'], 
                                mask = np.logical_or(maps['Hb_flux_mask'], 
                                    np.abs(maps['Hb_flux']*np.sqrt(maps['Hb_flux_ivar']) < 3)))
    maps['mHb_flux_ivar'] = ma.array(maps['Hb_flux_ivar'], mask=maps['mHb_flux'].mask)

    maps['mOII_flux'] = ma.array(maps['OII_flux'], 
                                mask= np.logical_or(maps['OII_flux_mask'], 
                                    np.abs(maps['OII_flux']*np.sqrt(maps['OII_flux_ivar']) < 3)))
    maps['mOII_flux_ivar'] = ma.array(maps['OII_flux_ivar'], mask=maps['mOII_flux'].mask)

    maps['mOII2_flux'] = ma.array(maps['OII2_flux'], 
                                mask= np.logical_or(maps['OII2_flux_mask'], 
                                    np.abs(maps['OII2_flux']*np.sqrt(maps['OII2_flux_ivar']) < 3)))
    maps['mOII2_flux_ivar'] = ma.array(maps['OII2_flux_ivar'], mask=maps['mOII2_flux'].mask)

    maps['mOIII_flux'] = ma.array(maps['OIII_flux'], 
                                mask= np.logical_or(maps['OIII_flux_mask'], 
                                    np.abs(maps['OIII_flux']*np.sqrt(maps['OIII_flux_ivar']) < 3)))
    maps['mOIII_flux_ivar'] = ma.array(maps['OIII_flux_ivar'], mask=maps['mOIII_flux'].mask)

    maps['mOIII2_flux'] = ma.array(maps['OIII2_flux'], 
                                    mask= np.logical_or(maps['OIII2_flux_mask'], 
                                    np.abs(maps['OIII2_flux']*np.sqrt(maps['OIII2_flux_ivar']) < 3)))
    maps['mOIII2_flux_ivar'] = ma.array(maps['OIII2_flux_ivar'], mask=maps['mOIII2_flux'].mask)

    maps['mNII_flux'] = ma.array(maps['NII_flux'], 
                                mask= np.logical_or(maps['NII_flux_mask'], 
                                    np.abs(maps['NII_flux']*np.sqrt(maps['NII_flux_ivar']) < 3)))
    maps['mNII_flux_ivar'] = ma.array(maps['NII_flux_ivar'], mask=maps['mNII_flux'].mask)

    maps['mNII2_flux'] = ma.array(maps['NII2_flux'],  
                                mask= np.logical_or(maps['NII2_flux_mask'], 
                                    np.abs(maps['NII2_flux']*np.sqrt(maps['NII2_flux_ivar']) < 3)))
    maps['mNII2_flux_ivar'] = ma.array(maps['NII2_flux_ivar'], mask=maps['mNII2_flux'].mask)
    

    
    # set the correction coefficient based on H-alpha and H-beta maps

    H_ratio = ma.array(maps['mHa_flux'] / maps['mHb_flux'], mask=maps['Ha_flux_mask'] + maps['Hb_flux_mask'] + (maps['mHa_flux'] / maps['mHb_flux'] ==0))

    H_ratio[H_ratio.mask] = np.nan

    rc.setCorr(H_ratio / 2.86, 6564, 4862)

    

    # deredden

    dmaps = {}


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

    # get super mask
    super_mask = ma.array(np.zeros((len(dmaps['dmHb_flux']), len(dmaps['dmHb_flux'][0]))))

    for m in dmaps:
        super_mask = ma.array(super_mask, mask = super_mask.mask + dmaps[m].mask)


    # mask AGN
    AGN_masked_maps = mask_AGN(dmaps)
    #AGN_masked_maps = dmaps
 

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

    return metallicity_map, metallicity_map_ivar, super_mask

    
################################################################################
################################################################################
################################################################################


def find_global_metallicity(R25_pc, gradient, gradient_err, met_0, met_0_err):

    '''

    Calculate the global metallicity as the metallicity at 0.4 x R25
    De Vis et al. (2019)

    PARAMETERS
    ==========

    R25_pc : float
        R25 B-band radius [pc]

    gradient : float
        metallicity gradient [dex/kpc]

    met_0 : float
        central metallicity [dex]

    RETURNS
    =======
    
    met_glob : float
        global metallicity [dex]

    '''

    met_glob = gradient * R25_pc / 1000 * 0.4 + met_0

    met_glob_err2 = (R25_pc*0.4/1000)**2 * gradient_err**2 + met_0_err**2
    met_glob_err = np.sqrt(met_glob_err2)



    return met_glob, met_glob_err

################################################################################
################################################################################
################################################################################

def calculate_gradient_chi2(params, r, m_med, m_sigma_asym):

    met_model = linear_metallicity_gradient(r, params[0], params[1])
    #m_sigma = np.ones(len(r))*np.nan
    '''
    for i in range(0, len(r)):
        if met_model[i] <= m_med[i]:
            m_sigma[i] = m_sigma_asym[i][0]

        if met_model[i] > m_med[i]:
            m_sigma[i] = m_sigma_asym[i][0]
    '''

    chi2 = np.sum((met_model-m_med)**2/m_sigma_asym**2)

    return chi2


################################################################################
################################################################################
################################################################################
def calculate_metal_mass(met_glob, met_glob_err, M_HI, M_HI_err, M_H2, M_H2_err, M_star, M_star_err):

    '''

    calculate metal mass (minus metals locked up in dust) and dust mass (including metals in dust)
    according to De Vis et al. (2019) for PG16R metallicity calibration 
    dust mass(incl metals) / (metal mass  + metal mass in dust) = 0.206


    PARAMETERS
    ==========

    met_glob : float
        global metallicity defined as metallicity at 0.4 R25

    M_HI : float
        HI mass

    M_H2 : float
        H2 mass
    
    M_star : float
        stellar mass

    RETURNS
    =======

    Mz : float
        heavy metal mass (not including metals in dust)

    Md : float 
        dust mass (includes metals in dust)

    '''

    Mz = None
    Md = None

    if M_HI_err == None:
        M_HI_err = 0

    if met_glob > 8.2:
        print('global metallicity', met_glob)

        if M_H2 == None:

            return 0,0

            '''
            print(M_HI)
            M_H2 = M_HI * 10**(-0.72*np.log10(M_HI/M_star) - 0.78)
            M_H2_err = np.sqrt(((0.28*M_HI**-0.72 * M_star**0.72*10**-0.78)**2 * M_HI_err**2 \
            + 0.72*M_HI**0.28 * M_star**-0.28*10**-0.78)**2 * M_H2_err**2)

            print('MH2: ', np.log10(M_H2))
            '''
        
        fz = 27.36 * 10**(met_glob - 12)
        xi = 1 / (1 - (fHep + fz * grad_Hep_z) - fz)
        Mg = xi * M_HI * (1 + M_H2/M_HI)

        Mz = fz*Mg

        ratio = np.random.normal(-0.69, 0.21, 1000)

        #Md = Mz / (1/0.206 - 1)
        Md = Mz / (1/10**ratio - 1)

        Mz_tot = np.mean(Md + Mz)
        Mz_std = np.std(Md+Mz)

        sigma_fz2 = (fz*np.log(10))**2 * met_glob_err**2
        sigma_xi2 = xi**2*(grad_Hep_z +1)**2*sigma_fz2

        sigma_Mg2 = sigma_xi2*(M_HI + M_H2) + ((10**M_HI_err)**2+(10**M_H2_err)**2)*xi**2

        sigma_Mz2 = Mg**2*sigma_fz2 + fz**2*sigma_Mg2

        sigma_Mztot = np.sqrt(Mz_std**2 + sigma_Mz2)


    else:
        print('global metallicity', met_glob)

        return None, None


    #return ma.log10(Mz), ma.log10(Md)

    return ma.log10(Mz_tot), ma.log10(sigma_Mztot)




################################################################################
################################################################################
################################################################################

def fit_metallicity_gradient(   MANGA_FOLDER,
                                DRP_FOLDER, 
                                IMAGE_DIR, 
                                corr_law, 
                                gal_ID,
                                center_coord,
                                phi,
                                ba,
                                z
                                ):


    
    ################################################################################
    # calculate scales for deprojection
    ################################################################################



    dist_to_galaxy_Mpc = c*z/H_0
    dist_to_galaxy_kpc = dist_to_galaxy_Mpc*1000
    pix_scale_factor = dist_to_galaxy_kpc*np.tan(MANGA_SPAXEL_SIZE)

    
    ################################################################################
    # get metallicity map and uncertainties
    ################################################################################


    metallicity_map, metallicity_map_ivar, super_mask = get_metallicity_map(DRP_FOLDER, 
                                                                IMAGE_DIR, 
                                                                corr_law,
                                                                gal_ID)


    ################################################################################
    # deproject metallicity maps, flatten and remove nans
    ################################################################################

    cosi2 = (ba**2 - q0**2)/(1 - q0**2)
    i_angle = np.arccos(np.sqrt(cosi2))

    print('cosi2', cosi2)
    print('i', i_angle)
    print('center', center_coord)
    print()

    r_kpc = np.zeros((len(metallicity_map), len(metallicity_map[0])))

    for i in range(len(metallicity_map)):
        for j in range(len(metallicity_map[0])):

            r_spax, _ = deproject_spaxel((i,j), center_coord, phi, i_angle)
            r_kpc[i][j] = r_spax*pix_scale_factor


    nan_mask = np.isnan(metallicity_map)

    print('r_kpc', r_kpc)

    r_flat = ma.array(r_kpc, mask=nan_mask).compressed()
    m = ma.array(metallicity_map, mask=nan_mask).compressed()
    m_sigma = ma.array(1/ma.sqrt(metallicity_map_ivar), mask=nan_mask).compressed()

   
    '''
    ################################################################################
    # bin metallicity by radius, use median uncertainty (kenney and keeping 1962)
    ################################################################################

    bin_edges = np.linspace(0, np.max(r_flat), 15)
    step_size = (bin_edges[0] + bin_edges[1]) / 2
    bin_centers = bin_edges[:-1] + step_size 

    m_median = np.zeros(len(bin_centers))
    m_sigma_med = np.zeros(len(bin_centers))

    for i in range(0, len(bin_centers)):

        if i == 0:
            vals = m[np.logical_and(r_flat >= bin_edges[i], r_flat <= bin_edges[i+1])]
            sigmas = m_sigma[np.logical_and(r_flat >= bin_edges[i], r_flat <= bin_edges[i+1])]
        
        else:
            vals = m[np.logical_and(r_flat > bin_edges[i], r_flat <= bin_edges[i+1])]
            sigmas = m_sigma[np.logical_and(r_flat > bin_edges[i], r_flat <= bin_edges[i+1])]


        m_median[i] = ma.median(vals)
        stddev = ma.std(vals)
        N = len(vals)

        if N != 1:
            m_sigma_med[i] = stddev*ma.sqrt(2*np.pi*N/(N-1))

        if N == 1:
            m_sigma_med[i] = sigmas[0]


    print('m_med', m_median)
    print('m_sigma_med', m_sigma_med)
    '''

    ################################################################################
    # fit metallicity map to linear metallicity gradient
    ################################################################################
    
    print('r_flat', r_flat)
    print('m', m)
    print('m_sigma', m_sigma)
    #median
    popt, pcov = curve_fit(linear_metallicity_gradient, 
                                    r_flat, 
                                    m,
                                    sigma=m_sigma
                                    )
    
    

    ################################################################################
    # unpack and plot results
    ################################################################################


    cov_dir = MANGA_FOLDER + 'metallicity_cov/'
    np.save(cov_dir + 'metallicity_' + gal_ID + '_cov.npy', pcov) 

    perr = np.sqrt(np.diag(pcov))

    
    best_fit_values = {'grad': popt[0], 
                        'grad_err': perr[0], 
                        '12logOH_0': popt[1], 
                        '12logOH_0_err': perr[1]}
    
    print(best_fit_values)
                        
    


    plot_metallicity_gradient(cov_dir, IMAGE_DIR, gal_ID, r_flat, m, m_sigma, popt)
    


    return best_fit_values, r_kpc, pix_scale_factor, dist_to_galaxy_kpc, super_mask


################################################################################
################################################################################
################################################################################


def calculate_global_metallicity(fluxes, corr_law='CCM89'):

    '''
    Metallicity method for global fluxes instead of map
    
    PARAMETERS
    ==========

    fluxes : dict
        dictionary of fluxes and uncertainties
        - Ha_flux, _ivar: H-alpha flux [1e-17 erg/s/cm^2/ang/spaxel]
        - Hb_flux, _ivar: H-beta flux [1e-17 erg/s/cm^2/ang/spaxel]
        - OII_flux, _ivar: [OII] 3727 flux [1e-17 erg/s/cm^2/ang/spaxel]
        - OII2_flux, _ivar: [OII] 3729 flux [1e-17 erg/s/cm^2/ang/spaxel]
        - OIII_flux, _ivar: [OIII] 4960 flux [1e-17 erg/s/cm^2/ang/spaxel]
        - OIII2_flux, _ivar: [OIII] 5008 flux [1e-17 erg/s/cm^2/ang/spaxel]
        - NII_flux, _ivar: [NII] 6549 flux [1e-17 erg/s/cm^2/ang/spaxel]
        - NII2_flux, _ivar: [NII] 6585 flux [1e-17 erg/s/cm^2/ang/spaxel]


    RETURNS
    =======

    Z : float
        metallicity [dex]
    Z_err : float
        uncertainty on metallicity [dex]



    '''


    ################################################################################
    # check that all lines have positive flux values
    ################################################################################
    
    for key in fluxes:

        if not fluxes[key] > 0:
            print('Missing fluxes')
            return 0, 0

    ################################################################################
    # dust correction using Balmer decrement
    ################################################################################

    rc = pn.RedCorr(law=corr_law)
    H_ratio = fluxes['Ha'] / fluxes['Hb']
    rc.setCorr(H_ratio/2.86, 6564, 4862)
    wavelengths=np.array([3727, 3729, 4960, 5008, 6549, 6585, 6564, 4862])
    corr = rc.getCorrHb(wavelengths)

    OII = fluxes['OII'] * corr[0]
    OII_err = fluxes['OII_err'] * corr[0]

    OII2 = fluxes['OII2'] * corr[1]
    OII2_err = fluxes['OII2_err'] * corr[1]

    OIII = fluxes['OIII'] * corr[2]
    OIII_err = fluxes['OIII_err'] * corr[2]

    OIII2 = fluxes['OIII2'] * corr[3]
    OIII2_err = fluxes['OIII2_err'] * corr[3]

    NII = fluxes['NII'] * corr[4]
    NII_err = fluxes['NII_err'] * corr[4]

    NII2 = fluxes['NII2'] * corr[5]
    NII2_err = fluxes['NII2_err'] * corr[5]

    Ha = fluxes['Ha'] * corr[6]
    Ha_err = fluxes['Ha_err'] * corr[6]

    Hb = fluxes['Hb'] * corr[7]
    Hb_err = fluxes['Hb_err'] * corr[7]

    ################################################################################
    # calculate metallicity ratios and uncertainties
    ################################################################################

    R2 = (OII + OII2)/Hb
    R2_err = 1/Hb * np.sqrt(OII_err**2 + OII2_err**2 + Hb_err**2*R2**2)

    N2 = (NII + NII2)/Hb
    N2_err = 1/Hb * np.sqrt(NII_err**2 + NII2_err**2 + Hb_err**2*N2**2)

    R3 = (OIII + OIII2)/Hb
    R3_err = 1/Hb * np.sqrt(OIII_err**2 + OIII2_err**2 + Hb_err**2*R3**2)

    ################################################################################
    # calculate metallicity and uncertainty
    ################################################################################

    # upper branch

    if np.log10(N2) >= -0.6:
        a1 = 8.589
        a2 = 0.022
        a3 = 0.399
        a4 = -0.137
        a5 = 0.164
        a6 = 0.589


    # lower branch

    elif np.log10(N2) < -0.6:
        a1 = 7.932
        a2 = 0.944
        a3 = 0.695
        a4 = 0.970
        a5 = -0.291
        a6 = -0.019


    else:
        return 0,0

    Z = a1 + a2*np.log10(R3/R2) + a3*np.log10(N2) \
        + (a4 + a5*np.log10(R3/R2) + a6*np.log10(N2))*np.log10(R2)

    
    dZdR3 = a2/ln10 * 1/R3 + a5/ln10**2* np.log(R2)/R3
    dZdR2 = -a2/ln10 *(1/R2) - a5/ln10**2 *2*np.log(R2)/R2
    dZdN2 = a3/ln10 *1/N2 + a6/ln10**2*np.log(R2)/N2

    Z_err = np.sqrt(dZdR3**2*R3_err**2 + dZdR2**2*R2_err**2 + dZdN2**2*N2_err**2)


    return Z, Z_err