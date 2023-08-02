import math 
import numpy.ma as ma 
import numpy as np

from astropy.io import fits

import matplotlib.pyplot as plt

from DRP_rotation_curve import extract_data

from astropy.table import Table

from metallicity_map_plottingFunctions import plot_broadband_image



################################################################################
################################################################################
################################################################################

def extract_broadband_images(FOLDER, gal_ID):
    '''
    Open the MaNGA .fits file and extract g and r images.

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
          - g_band: g-band image [nanomaggies/pixel]
          - r_band: r-band image [nanomaggies/pixel]

    '''

    file_name = FOLDER + 'manga-' + gal_ID + '-LOGCUBE.fits.gz'

    maps = {}

    cube = fits.open(file_name)

    maps['r_band'] = cube[13].data
    maps['g_band'] = cube[12].data

    cube.close()

    return maps


################################################################################
################################################################################
################################################################################

def B_band_map(IMAGE_DIR,
                gal_ID,
                 g,r, A_g, A_r, Ha_flux, Ha_flux_ivar, Ha_vel_mask):

    '''
    
    Takes g and r-band maps of galaxy, applies H-alpha velocity mask and SN < 3 flux mask,
    converts nMgy to magnitudes, applies dust correction, and converts to Vega B-band map
    using Blanton & Roweis (2007)

    PARAMETERS
    ==========

    g : array
        MaNGA g-band map

    r : array
        MaNGA r-band map

    A_g : float
        g-band extinction correction [mag]
    
    A_r : float
        r-band extinction correction [mag]

    Ha_flux : array
        MaNGA H-alpha flux map

    Ha_flux_ivar : array
        MaNGA H-alpha flux inverse variance map

    Ha_vel_mask : array
        MaNGA H-alpha velocity map nasj

    RETURNS
    =======
    Bvega : array
        B-band map in Vega photometric system


    '''

    # build SN < 3 and Ha_vel mask

    SN_3_mask = np.logical_or(Ha_vel_mask > 0, np.abs(Ha_flux*np.sqrt(Ha_flux_ivar)) < 3)

    g = ma.array(g, mask=SN_3_mask)
    r = ma.array(r, mask=SN_3_mask)


    # convert g-band and r-band nanomaggies to asinh mag

    g_ab = asinh_magnitude('g', g)
    r_ab = asinh_magnitude('r', r)



    # apply extinction correction and convert to B band magnitude


    Bab = (g_ab + A_g) + 0.2354 + 0.3915*(((g_ab+A_g)-(r_ab+ A_r)) - 0.6102)
    Bvega = Bab + 0.09

    plot_broadband_image(IMAGE_DIR, gal_ID, Bvega, 'B')

    return Bvega


################################################################################
################################################################################
################################################################################



def sersic_profile(params, r):

    '''
    
    Sersic profile for surface brightness of galaxy

    PARAMETERS
    ==========

    

    RETURNS
    =======
    Sigma : float
        brightness at radius r


    '''


    Sigma_e, Re, n, Sigma_0_in, h_in, Sigma_0_out, h_out, R_break = params

    bn = 0

    if r <= R_break:
        Sigma = Sigma_e * np.exp(-bn*((r/Re)**(1/n) - 1)) + Sigma_0_in*np.exp(-r/h_in)

    else:
        Sigma = Sigma_e*np.exp(-bn*((r/Re)**(1/n)-1)) + Sigma_0_out*np.exp(-r/h_out)


    return Sigma



################################################################################
################################################################################
################################################################################


def asinh_magnitude(filter, flux):

    '''
    Takes map in SDSS g-band or r-band and converts from flux in nanomaggies to 
    SDSS asinh magnitude

    PARAMETERS
    ==========

    filter : string
        g or r filter used for broadband image

    flux : array or float
        broadband image flux nanomaggies/pixel units

    RETURNS
    =======

    mag : array or float
        conversion to mag/pixel units


    '''

    if filter == 'r':
        b = 1.2e-10
    elif filter == 'g':
        b = 0.9e-10
    else:
        print('add conversion for ' + filter + ' filter')
        return

    f0 = 10e9 # [nanomaggies]

    mag = -2.5/ma.log(10) * (ma.arcsinh((flux/f0)/(2*b)) + ma.log(b))

    return mag


################################################################################
################################################################################
################################################################################


def fit_surface_brightness_profile(DRP_FOLDER, IMAGE_DIR, gal_ID, A_g, A_r):
    
    maps = extract_broadband_images(DRP_FOLDER, gal_ID)

    drp_maps = extract_data(DRP_FOLDER, gal_ID, ['Ha_vel', 'Ha_flux'])

    B_map = B_band_map(IMAGE_DIR,
                    gal_ID,
                    maps['g_band'], 
                    maps['r_band'], 
                    A_g, A_r, 
                    drp_maps['Ha_flux'], 
                    drp_maps['Ha_flux_ivar'],
                    drp_maps['Ha_vel_mask'])

    # deproject map

    # fit to sersic profile

    # plot sersic profile with best fit

    # return best fit values

    return None


