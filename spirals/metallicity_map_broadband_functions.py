import math 
import numpy.ma as ma 
import numpy as np

from astropy.io import fits

import matplotlib

import matplotlib.pyplot as plt

from scipy.optimize import minimize

from DRP_rotation_curve import extract_data

from DRP_vel_map_functions import deproject_spaxel

from astropy.table import Table

from metallicity_map_plottingFunctions import plot_broadband_image, plot_surface_brightness



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
                 g,r, A_g, A_r, Ha_flux, Ha_flux_ivar, Ha_vel_mask, metallicity_mask):

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
        MaNGA H-alpha velocity map mask

    RETURNS
    =======
    Bvega : array
        B-band map in Vega photometric system


    '''

    # build SN < 3 and Ha_vel mask

    SN_3_mask = np.logical_or(Ha_vel_mask > 0, np.abs(Ha_flux*np.sqrt(Ha_flux_ivar)) < 3)

    g = ma.array(g, mask=SN_3_mask + metallicity_mask)
    r = ma.array(r, mask=SN_3_mask + metallicity_mask)


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

    bn = 2*n - 1/3

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
    d : float
        distance to galaxy [kpc]

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


def calculate_sigma(params, surface_brightness, r_pc):


    # calc model values

    surface_brightness_model = np.zeros(len(surface_brightness.mask))

    for i in range(0,len(surface_brightness_model)):

            surface_brightness_model[i] = sersic_profile(params, r_pc[i])

    # flatten and remove nans

    msurface_brightness_model = ma.array(surface_brightness_model, mask = surface_brightness.mask)
    flat_surface_brightness_model = msurface_brightness_model.compressed()

    flat_surface_brightness = surface_brightness.compressed()

    sigma = np.sqrt(np.sum((flat_surface_brightness_model/flat_surface_brightness - 1)**2) / len(flat_surface_brightness))

    return sigma






################################################################################
################################################################################
################################################################################

def fit_surface_brightness_profile(DRP_FOLDER, 
                                    IMAGE_DIR, 
                                    gal_ID, 
                                    A_g, 
                                    A_r,
                                    r_kpc,
                                    scale,
                                    d_kpc,
                                    metallicity_mask):



    '''
    PARAMETERS
    ==========

    DRP_FOLDER : string
        directory containing maps

    IMAGE_DIR : string
        directory to save plots

    gal_ID : string
        PLATE-IFU for galaxy

    A_g : float
        g-band extinction correction

    A_r : float
        r-band extinction correction

    r_kpc : array
        deprojected distances of spaxels from center of galaxy spaxel [kpc]

    scale : float
        scale factor to convert spaxel distance to kpc [kpc/spax]

    d_kpc : float
        distance to galaxy [kpc]

    RETURNS
    =======




    '''
    
    # extract necessary maps

    maps = extract_broadband_images(DRP_FOLDER, gal_ID)

    drp_maps = extract_data(DRP_FOLDER, gal_ID, ['Ha_vel', 'Ha_flux'])

    # generate B-band map from g-band and r-band maps

    B_map = B_band_map(IMAGE_DIR,
                    gal_ID,
                    maps['g_band'], 
                    maps['r_band'], 
                    A_g, A_r, 
                    drp_maps['Ha_flux'], 
                    drp_maps['Ha_flux_ivar'],
                    drp_maps['Ha_vel_mask'],
                    metallicity_mask)

    print('B map min:')
    print(np.min(B_map))

    # convert apparent B-band magnitude to solar luminosity/pc^2

    B_lum_map = B_mag_to_lum(B_map, d_kpc)

    # convert deprojected 



    r_pc = ma.array(r_kpc.flatten() * 1000, mask = B_map.mask.flatten())

    surface_brightness = ma.array(B_lum_map.flatten() / (scale*1000)**2,mask = B_map.mask.flatten())
    '''
    plot_surface_brightness(IMAGE_DIR, gal_ID, surface_brightness, r_pc)
    '''

    # fit to sersic profile

    #break rad guess = 2000

    '''
    Sigma_e_guess = 10 # L/pc^2
    Re_guess = 100 # pc
    n_guess = 5
    Sigma_0_in_guess = 100 #L/pc^2
    h_in_guess = 50 #pc
    Sigma_0_out_guess = 5 # L/pc^2
    h_out_guess = 500 #pc
    R_break_guess = 2000 # pc

    guesses = [Sigma_e_guess, Re_guess, n_guess, Sigma_0_in_guess, 
                h_in_guess, Sigma_0_out_guess, h_out_guess, R_break_guess]


    print('fitting')

    result = minimize(calculate_sigma,
                        guesses,
                        method='Powell',
                        args = (surface_brightness,r_pc),
                        options={'disp':True})

                        

    print('fitting done!')
    print(result.x)
    '''

    # plot sersic profile with best fit

    # return best fit values

    # TEMP

    surface_brightness_model = np.zeros(len(r_pc))
    p = [7.82121654e+03, 1.04503516e+02,  1.38814393e+06, -4.27257651e+04, 
        6.36970124e+01,  1.61773647e+02,  1.19415002e+03,  1.99986707e+03]

    for i in range(0, len(r_pc)):
        surface_brightness_model[i] = sersic_profile(p, r_pc[i])

    good = surface_brightness_model[surface_brightness_model >= 0]
    good_r = r_pc[surface_brightness_model >= 0]

    plt.scatter(r_pc, np.log10(surface_brightness), marker='.', color='k')
    plt.scatter(good_r, np.log10(good))
    plt.xlabel('radius [pc]')
    plt.ylabel('$log\Sigma_L\ (L\odot/pc^2)$')
    plt.title(gal_ID)
    plt.savefig('TEST_SB.eps')
    plt.close()


    return None


