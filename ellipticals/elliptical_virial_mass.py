import numpy as np
import numpy.ma as ma

from elliptical_plottingFunctions import *

################################################################################
# CONSTANTS
################################################################################

G = 4.3009 * 10**-6 # [kpc (km/s)^2 / M_sun]
H_0 = 100 # [km / s / Mpc]
c = 299792.458 # Speed of light in units of km/s

################################################################################


def sMass_weighted_star_sigma(gal_ID, star_sigma, star_sigma_ivar, 
                              star_sigma_corr, star_sigma_mask, sMass_density,
                              sMass_density_err, IMAGE_DIR=None, 
                              IMAGE_FORMAT='png'):
    
    '''

    Apply correction and mask to stellar velocity dispersion map and 
    calculate stellar mass weighted mean, uncertainty

    PARAMETERS
    ==========
    gal_ID : string
        galaxy plate-ifu

    star_sigma : array
        stellar velocity dispersion map [km/s]
    
    star_sigma_ivar : array
        stellar velocity dispersion inverse variance map

    star_sigma_corr : array
        stellar velocity dispersion correction map

    star_sigma_mask : array
        stellar velocity dispersion default mask

    sMass_density : array
        stellar mass density map [log(M_sun/spax^2)]

    sMass_density_err : array
        stellar mass density uncertainty map
    
    IMAGE_DIR : string
        directory to save figures, default is None

    IMAGE_FORMAT : string
        format to save figures, deafault is 'png'

    RETURNS
    =======
    x2 : float
        stellar mass weighted mean square velocity dispersion [(km/s)^2]

    x2_err2 : float
        square of the uncertainty on x2

    mstar_sigma : array
        masked stellar velocity dispersion map

    mstar_sigma_corrected : array
        masked and corrected stellar velocity dispersion map



    '''

    ############################################################################
    # mask all maps
    ############################################################################

    
    sn10_mask = np.logical_or(star_sigma * 
                               np.sqrt(star_sigma_ivar) < 10, 
                               star_sigma_mask)

    mstar_sigma = ma.array(star_sigma, mask=sn10_mask)
    mstar_sigma_ivar = ma.array(star_sigma_ivar, mask=sn10_mask)
    mstar_sigma_corr = ma.array(star_sigma_corr, mask=sn10_mask)

    msMass_density = ma.array(sMass_density, mask=sn10_mask)
    msMass_density_err = ma.array(sMass_density_err, mask=sn10_mask)

    ############################################################################
    # apply correction to stellar velocity dispersion map and plot
    ############################################################################

    mstar_sigma_corrected = np.sqrt(mstar_sigma**2 - mstar_sigma_corr**2)
    plot_sigma_map(gal_ID, mstar_sigma, IMAGE_DIR=IMAGE_DIR, IMAGE_FORMAT=IMAGE_FORMAT)
    plot_sigma_map(gal_ID, mstar_sigma_corrected, corr=True, IMAGE_DIR=IMAGE_DIR, 
                   IMAGE_FORMAT=IMAGE_FORMAT)
    
    ############################################################################
    # calculate stellar mass weighted mean velocity dispersion
    ############################################################################

    # sMass = np.ma.power(10,msMass_density) # stellar mass map in linear units
    # sMass_tot = ma.sum 
    # sigma2 = ma.sum(sMass*mstar_sigma_corrected**2)/ma.sum(sMass)
    # sigma = np.sqrt(sigma2)
    # return sigma, 0, mstar_sigma, mstar_sigma_corrected


    lin_sMass = np.ma.power(10, msMass_density) # stellar mass map in linear units
    lin_sMass_err = np.ma.power(10, msMass_density_err) # stellar mass error in linear units


    M = ma.sum(lin_sMass) # total stellar mass, normalization factor
    S = ma.sum(lin_sMass*mstar_sigma_corrected**2) # weighted sum of vel disp squared

    # mass weighted mean velocity disp squared and square of uncertainty on this quantity
    x2 = S / M 
    x2_err2 = (4 / M**2) * \
        ma.sum((lin_sMass*mstar_sigma_corrected)**2/mstar_sigma_ivar) +\
        (1 / M**4) * ma.sum(((M*mstar_sigma_corrected**2-S)*lin_sMass_err)**2)
    
    return x2, x2_err2, mstar_sigma, mstar_sigma_corrected




def median_star_sigma(gal_ID, star_sigma, star_sigma_ivar, 
                      star_sigma_corr, star_sigma_mask, IMAGE_DIR=None, IMAGE_FORMAT='png'):
    '''

    Apply correction and mask to stellar velocity dispersion map and find 
    median, uncertainty

    PARAMETERS
    ==========

    gal_ID : string
        MaNGA plate-IFU

    star_sigma : array
        stellar velocity dispersion map

    star_sigma_ivar : array
        stellar velocity dispersion inverse variance

    star_sigma_corr : array
        stellar velocity dispersion instrument correction

    star_sigma_mask : array
        stellar velocity dispersion default mask

    IMAGE_DIR : string
        directory to save plots

    IMAGE_FORMAT : string
        format to save plots. default is 'png'

    RETURNS
    =======
    star_sigma_med : float
        median stellar velocity dispersion calculated from masked map

    star_sigma_med_err : float
        error on star_sigma_med

    mstar_sigma : array
        masked stellar velocity dispersion map

    mstar_sigma_corr : array
        masked stellar velocity dispersion correction map
          mstar_sigma, mstar_sigma_corr




    '''
    ############################################################################
    # mask all maps
    ############################################################################

    
    # sn10_mask = np.logical_or(maps['star_sigma'] * 
    #                           np.sqrt(maps['star_sigma_ivar']) < 10, 
    #                           star_sigma_mask)

    mstar_sigma = ma.array(star_sigma, mask=star_sigma_mask)
    mstar_sigma_ivar = ma.array(star_sigma_ivar, mask=star_sigma_mask)
    mstar_sigma_corr = ma.array(star_sigma_corr, mask=star_sigma_mask)

    ############################################################################
    # apply correction to stellar velocity dispersion map and plot
    ############################################################################

    mstar_sigma_corrected = np.sqrt(mstar_sigma**2 - mstar_sigma_corr**2)
    plot_sigma_map(gal_ID, mstar_sigma, IMAGE_DIR=IMAGE_DIR, IMAGE_FORMAT=IMAGE_FORMAT)
    plot_sigma_map(gal_ID, mstar_sigma_corrected, corr=True, IMAGE_DIR=IMAGE_DIR, 
                   IMAGE_FORMAT=IMAGE_FORMAT)

    ############################################################################
    # find the median velocity dispersion and uncertainty
    ############################################################################

    N = mstar_sigma_corrected.size
    n = (N-1)/2

    star_sigma_med = ma.median(mstar_sigma_corrected)
    star_sigma_med_err = ma.std(mstar_sigma_corrected) * \
        np.sqrt(np.pi * (2*n+1)/(4*n))
    
    
    return star_sigma_med, star_sigma_med_err, mstar_sigma, mstar_sigma_corrected



def calculate_virial_mass(gal_ID, star_sigma, star_sigma_ivar, 
                          star_sigma_corr, star_sigma_mask, flux_map,
                          sMass_density, sMass_density_err,
                           Ha_vel, Ha_vel_mask, r50, z, 
                          IMAGE_DIR=None, IMAGE_FORMAT='png' ):
    '''
    calculate virial mass using the stellar mass weighted mean weighted velocity
    dispersion

    plot diagnostic plots for each galaxy 

    PARAMETERS
    ==========
    gal_ID : string
        galaxy plate-ifu

    star_sigma : array
        stellar velocity dispersion map [km/s]
    
    star_sigma_ivar : array
        stellar velocity dispersion inverse variance map

    star_sigma_corr : array
        stellar velocity dispersion correction map

    star_sigma_mask : array
        stellar velocity dispersion default mask

    flux_map : array
        g-band weighted mean flux map

    sMass_density : array
        stellar mass density map [log(M_sun/spax^2)]

    sMass_density_err : array
        stellar mass density uncertainty map

    Ha_vel : array
        H-alpha velocity map

    Ha_vel_mask : array
        H-alpha velocity map mask

    r50 : float
        Petrosian half-light radius [arcseconds]

    z : float
        nsa redshift
    
    IMAGE_DIR : string
        directory to save figures, default is None

    IMAGE_FORMAT : string
        format to save figures, deafault is 'png'

    RETURNS
    =======
    logMvir : float
        virial mass of galaxy [log(M_sun)]

    logMvir_err : float
        uncertainty on logMvir

    x : float
        stellar mass weighted mean velocity dispersion [km/s]

    x_err : float
        uncertainty on x

    
    '''

    ############################################################################
    # get median velocity dispersion and uncertainty
    ############################################################################

    # star_sigma_med, star_sigma_med_err, mstar_sigma, mstar_sigma_corrected = median_star_sigma(gal_ID,
    #                                                        star_sigma, 
    #                                                        star_sigma_ivar, 
    #                                                        star_sigma_corr, 
    #                                                        star_sigma_mask,
    #                                                        IMAGE_DIR,
    #                                                        IMAGE_FORMAT)
    
    x2, x2_err2, mstar_sigma, mstar_sigma_corrected = \
        sMass_weighted_star_sigma(gal_ID,
                                  star_sigma, 
                                  star_sigma_ivar, 
                                  star_sigma_corr, 
                                  star_sigma_mask,
                                  sMass_density,
                                  sMass_density_err,
                                  IMAGE_DIR,
                                  IMAGE_FORMAT)
    
    ############################################################################
    # plot photometric image
    ############################################################################

    plot_flux_map( gal_ID, flux_map, IMAGE_DIR=IMAGE_DIR, IMAGE_FORMAT=IMAGE_FORMAT)

    ############################################################################
    # plot photometric image
    ############################################################################
    mvel = ma.array(Ha_vel, mask=Ha_vel_mask)

    plot_vel(mvel, gal_ID, IMAGE_DIR=IMAGE_DIR, IMAGE_FORMAT=IMAGE_FORMAT)

    plot_diagnostic_panel(flux_map, mstar_sigma, mstar_sigma_corrected, mvel, gal_ID, 
                          IMAGE_DIR, IMAGE_FORMAT )


    ############################################################################
    # calculate r in kpc 
    ############################################################################

    dist_to_galaxy_Mpc = c*z/H_0
    dist_to_galaxy_kpc = dist_to_galaxy_Mpc*1000

    r50_kpc = dist_to_galaxy_kpc*np.tan(r50*(1./60)*(1./60)*(np.pi/180))

    ############################################################################
    # calculate mass and uncertainty
    ############################################################################

    # logMvir = np.log10(7.5 * r50_kpc * star_sigma_med**2 / G)
    # logMvir_err = 2 / (star_sigma_med * np.log(10)) * star_sigma_med_err
    
    x = np.sqrt(x2)
    x_err = np.sqrt(x2_err2/(4*x2))
    logMvir = np.log10(7.5 * r50_kpc * x2 / G)
    logMvir_err = 2 / (x * np.log(10)) * x_err


    return logMvir, logMvir_err, x, x_err


def calculate_dipole_moment(Ha_vel, Ha_vel_mask, Ha_flux, Ha_flux_ivar, flux):
    '''
    calculate the dipole moment of H-alpha velocity map: p = sum_i(v_i*(r_i-r))
    where r is a reference point chosen as a center guess (brightest point in 
    flux map)

    PARAMETERS
    ==========
    Ha_vel : array
        H-alpha velocity map

    Ha_vel_mask : array
        H-alpha velocity map mask

    Ha_flux : array
        H-alpha flux map

    Ha_flux_ivar : array
        H-alpha flux inverse variance
    
    flux : array
        g-band weighted mean flux map

    RETURNS
    =======
    p_mag : float
        magnitude of dipole moment
    
    '''
    
    # mask all maps
    
    mask = np.logical_or(Ha_vel_mask, Ha_flux*np.sqrt(Ha_flux_ivar) < 5)
    mHa_vel = ma.array(Ha_vel, mask=mask)
    mflux = ma.array(flux, mask=mask)
    
    # guess the center by brightest spaxel
    
    x0, y0 = np.unravel_index(ma.argmax(mflux), mflux.shape)
    
    # caculate dipole moment
    p = [0,0]
    shape = mHa_vel.shape
    for i in range(shape[0]):
        for j in range(shape[1]):
            
            if not ma.is_masked(mHa_vel[i][j]):
                p += mHa_vel[i][j] * np.array([i-x0, j-y0])

    p_mag = np.hypot(p[0], p[1])
                
    return p_mag