import math 
import numpy.ma as ma 
import numpy as np

import datetime

from astropy.io import fits

import matplotlib

import matplotlib.pyplot as plt

from scipy.optimize import minimize

from DRP_rotation_curve import extract_data

from DRP_vel_map_functions import deproject_spaxel

from astropy.table import Table

from metallicity_map_plottingFunctions import plot_broadband_image, plot_surface_brightness

from metallicity_map_broadband_functions import surface_brightness_profile


#from metallicity_map_broadband_functions_cython import*





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

    g_ab = pogson_magnitude('g', g)
    r_ab = pogson_magnitude('r', r)



    # apply extinction correction and convert to B band magnitude


    Bab = (g_ab - A_g) + 0.2354 + 0.3915*(((g_ab-A_g)-(r_ab- A_r)) - 0.6102)
    Bvega = Bab + 0.09

    plot_broadband_image(IMAGE_DIR, gal_ID, Bvega, 'B')

    return Bvega


################################################################################
################################################################################
################################################################################







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

    #mag = 22.5 - 2.5*ma.log10(flux)


    return mag

################################################################################
################################################################################
################################################################################


def pogson_magnitude(filter, flux):

    '''
    Takes flux map in nanomaggies and converts to Pogson magnitude

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

    return L



################################################################################
################################################################################
################################################################################


def calculate_sigma(params, surface_brightness, r_pc):


    # calc model values

    #surface_brightness_model = np.zeros(len(surface_brightness.mask))
    surface_brightness_model = np.zeros(len(surface_brightness))

    for i in range(0,len(surface_brightness_model)):

            surface_brightness_model[i] = surface_brightness_profile(params.tolist(), r_pc[i])

    # flatten and remove nans

    #msurface_brightness_model = ma.array(surface_brightness_model, mask = surface_brightness.mask)
    #flat_surface_brightness_model = msurface_brightness_model.compressed()

    #flat_surface_brightness = surface_brightness.compressed()

    #sigma = np.sqrt(np.sum((flat_surface_brightness_model/flat_surface_brightness - 1)**2) / len(flat_surface_brightness))
    sigma = np.sqrt(np.sum((surface_brightness_model/surface_brightness - 1)**2) / len(surface_brightness))

    return sigma


################################################################################
################################################################################
################################################################################


def L_difference(r, L_25, params):
    
    return np.abs(L_25 - surface_brightness_profile(params, r))


################################################################################
################################################################################
################################################################################

def calculate_chi2(params, r, sb_median, sigma_asym):

    sb_model = np.zeros(len(r))
    for i in range(len(r)):
        sb_model[i] = surface_brightness_profile(params, r[i])

    sigma = np.ones(len(r))*np.nan

    for i in range(0, len(r)):
        if sb_model[i] >= sb_median[i]:
            sigma[i] = sigma_asym[i][1]

        if sb_model[i] < sb_median[i]:
            sigma[i] = sigma_asym[i][0]

    chi2 = np.sum((sb_model - sb_median)**2/sigma**2)

    return chi2




################################################################################
################################################################################
################################################################################

def fit_surface_brightness_profile(DRP_FOLDER, 
                                    IMAGE_DIR, 
                                    gal_ID, 
                                    A_g, 
                                    A_r,
                                    r50,
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
    ################################################################################
    # extract necessary maps
    ################################################################################
    
    maps = extract_broadband_images(DRP_FOLDER, gal_ID)

    drp_maps = extract_data(DRP_FOLDER, gal_ID, ['Ha_vel', 'Ha_flux'])

    ################################################################################
    # generate B-band map from g-band and r-band maps
    ################################################################################

    B_map = B_band_map(IMAGE_DIR,
                    gal_ID,
                    maps['g_band'], 
                    maps['r_band'], 
                    A_g, A_r, 
                    drp_maps['Ha_flux'], 
                    drp_maps['Ha_flux_ivar'],
                    drp_maps['Ha_vel_mask'],
                    metallicity_mask)




    ################################################################################
    # convert apparent B-band magnitude to solar luminosity/pc^2 and deproject maps
    ################################################################################

    B_lum_map = B_mag_to_lum(B_map, d_kpc)

    r_pc = ma.array(r_kpc.flatten() * 1000, mask = B_map.mask.flatten())

    surface_brightness = ma.array(B_lum_map.flatten() / (scale*1000)**2,mask = B_map.mask.flatten())

    # remove the masked elements

    flat_r_pc = r_pc.compressed()
    flat_sb = surface_brightness.compressed()



    ################################################################################
    # convert nsa r50 to pc
    ################################################################################

    r50_spax = r50 / 0.5 
    r50_pc = r50_spax * scale * 1000
    print('r50_pc:', r50_pc)


    ################################################################################
    # pick initial guesses and bounds for surface brightness profile free params
    ################################################################################

    Sigma_e_guess = 100 # L/pc^2
    Sigma_e_bounds = (0.5, 1000)

    Re_guess = r50_pc # [pc] - use r50 from nsa as initial guess 
    Re_bounds = (100, 100000)

    n_guess = 5
    n_bounds = (1,10)  # from pilyugin breaks in disc galaxy abundance gradients

    #Sigma_0_in_guess = 10**2.2 #L/pc^2
    Sigma_0_in_guess = ma.max(surface_brightness)
    Sigma_0_in_bounds = (50, 10**4)

    h_in_guess = r50_pc/2 #pc
    h_in_bounds = (0.5, 5000)

    Sigma_0_out_guess = 1000 # L/pc^2
    Sigma_0_out_bounds = (10, 10000)

    h_out_guess = r50_pc/2 #pc
    h_out_bounds = (0.5, 5000)

    R_break_guess = r50_pc # pc
    R_break_bounds = (0.5, 4000)


    guesses = [Sigma_e_guess, Re_guess, n_guess, Sigma_0_in_guess, 
                h_in_guess, Sigma_0_out_guess, h_out_guess, R_break_guess]

    bounds = (Sigma_e_bounds, Re_bounds, n_bounds, Sigma_0_in_bounds,
                h_in_bounds, Sigma_0_out_bounds, h_out_bounds, R_break_bounds)


    ################################################################################
    # bin data by radius 
    ################################################################################

    #bin_edges = np.arange(0, np.max(r_pc), 700)
    bin_edges = np.linspace(0, np.max(r_pc), 40)
    step_size = (bin_edges[0] + bin_edges[1]) / 2
    bin_centers = bin_edges[:-1] + step_size 

    sb_median = np.zeros(len(bin_centers))
    sigma_asym = np.zeros((len(bin_centers), 2))

    for i in range(0, len(bin_centers)):

        if i == 0:
            vals = flat_sb[np.logical_and(flat_r_pc >= bin_edges[i], flat_r_pc <= bin_edges[i+1])]
        
        else:
            vals = flat_sb[np.logical_and(flat_r_pc > bin_edges[i], flat_r_pc <= bin_edges[i+1])]

        sb_median[i] = ma.median(vals)
        sigma_asym[i] = np.quantile(vals, [0.16,0.84])


    print('sb_med', sb_median)
    print('sigma_asym', sigma_asym)


    ################################################################################
    # fit data to bulge + exponential break profile
    ################################################################################

    plt.scatter(flat_r_pc, flat_sb)
    plt.savefig('test.png')
    plt.close()

    

    start = datetime.datetime.now()

    print('starting fit!')
    
    '''

    result = minimize(calculate_sigma,
                        guesses,
                        method='Powell',
                        args = (sb_median, bin_centers),
                        bounds = bounds,
                        options={'disp':True})

    '''

    result = minimize(calculate_chi2,
                        guesses,
                        method='Powell',
                        args=(bin_centers,sb_median,sigma_asym),
                        bounds=bounds,
                        options={'disp':True})


    print('fitting done!')

    end = datetime.datetime.now() - start
    print('fit time: ', end)

    ################################################################################
    # unpack result
    ################################################################################

    best_fit_vals = result.x.tolist()
    print(result)




    


    ################################################################################
    # now where is r25 ? 
    ################################################################################

    L_25 = B_mag_to_lum(25, d_kpc) / (scale*1000)**2

    r25_res = minimize(L_difference,
                        r50_pc,
                        method='Powell',
                        args=(L_25, best_fit_vals),
                        options={'disp':True})
    
    r25_pc = r25_res.x[0]
    print('R25 [pc] = ', r25_pc)

    ################################################################################
    # plot data and model
    ################################################################################

    plot_surface_brightness(IMAGE_DIR, gal_ID, sb_median, bin_centers, flat_r_pc, best_fit_vals, r25_pc)
    


    return {'R25_pc': r25_pc}



