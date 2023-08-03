import math 
import numpy.ma as ma 
import numpy as np

from astropy.io import fits

import matplotlib

import matplotlib.pyplot as plt

from DRP_rotation_curve import extract_data

from DRP_vel_map_functions import deproject_spaxel

from astropy.table import Table

from metallicity_map_plottingFunctions import plot_broadband_image



################################################################################
################################################################################
################################################################################


B_sol = 5.25 # magnitude of the usun in B-band Vega photometric system
q0 = 0.2 #nominal disk thickness
ln10 = ma.log(10)
H_0 = 100      # Hubble's Constant in units of h km/s/Mpc
c = 299792.458 # Speed of light in units of km/s
MANGA_SPAXEL_SIZE = 0.5*(1/60)*(1/60)*(np.pi/180)  # spaxel size (0.5") in radians


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


    Bab = (g_ab - A_g) + 0.2354 + 0.3915*(((g_ab-A_g)-(r_ab- A_r)) - 0.6102)
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

    #mag = -2.5/ma.log(10) * (ma.arcsinh((flux/f0)/(2*b)) + ma.log(b))

    mag = 22.5 - 2.5*ma.log10(flux)


    return mag


################################################################################
################################################################################
################################################################################

def B_mag_to_lum(B_mag,d):

    '''
    Converts B-band magnitude to luminosity in solar units

    PARAMETERS
    ==========
    B_mag : float or array
        B-band magnitude map

    RETURNS
    =======
    L : float or array
        luminosity in solar units


    '''

    B_mag = B_mag-5*ma.log10(d*1000/10)

    L = 10**((B_sol - B_mag)/2.5)

    plt.imshow(L)
    plt.colorbar()
    plt.savefig('L_test2.png')
    plt.close()

    return L



################################################################################
################################################################################
################################################################################


def fit_surface_brightness_profile(DRP_FOLDER, 
                                    IMAGE_DIR, 
                                    gal_ID, 
                                    A_g, 
                                    A_r,
                                    center_coord, 
                                    phi, 
                                    ba,
                                    z):
    
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

    print('B map min:')
    print(np.min(B_map))

    dist_to_galaxy_Mpc = c*z/H_0
    dist_to_galaxy_kpc = dist_to_galaxy_Mpc*1000
    pix_scale_factor = dist_to_galaxy_kpc*np.tan(MANGA_SPAXEL_SIZE)

    print(pix_scale_factor)

    B_lum_map = B_mag_to_lum(B_map, dist_to_galaxy_kpc)

    # deproject map into lum/pc^2 from lum/px
    # change this section to just read in deprojected map from before

    # scaling constants

    

    # deproject maps

    cosi2 = (ba**2 - q0**2)/(1 - q0**2)
    i_angle = np.arccos(np.sqrt(cosi2))

    r_pc = np.zeros((len(B_lum_map), len(B_lum_map[0])))

    for i in range(len(B_lum_map)):
        for j in range(len(B_lum_map[0])):

            r_spax, _ = deproject_spaxel((i,j), center_coord, phi, i_angle)
            r_pc[i][j] = r_spax*pix_scale_factor*1000 

    B_lum_map_pc2 = B_lum_map / (pix_scale_factor*1000)**2

    plt.imshow(B_lum_map_pc2)
    plt.colorbar(norm=matplotlib.colors.LogNorm())
    plt.savefig('B_lum_Lpc22.png')
    plt.close()

    plt.scatter(r_pc.flatten()/5820, np.log10(B_lum_map_pc2.flatten()))
    #plt.yscale('log')
    plt.savefig('B_lum_Lpc22_flat.png')
    plt.close()

    # fit to sersic profile

    # plot sersic profile with best fit

    # return best fit values

    return None


