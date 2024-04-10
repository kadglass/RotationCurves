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




def median_star_sigma(gal_ID, star_sigma, star_sigma_ivar, 
                      star_sigma_corr, star_sigma_mask, IMAGE_DIR=None, IMAGE_FORMAT='png'):
    '''

    Apply correction and mask to stellar velocity dispersion map and find 
    median, uncertainty



    '''
    ############################################################################
    # mask all maps
    ############################################################################

    mstar_sigma = ma.array(star_sigma, mask=star_sigma_mask)
    mstar_sigma_ivar = ma.array(star_sigma_ivar, mask=star_sigma_mask)
    mstar_sigma_corr = ma.array(star_sigma_corr, mask=star_sigma_mask)

    ############################################################################
    # apply correction to stellar velocity dispersion map and plot
    ############################################################################

    mstar_sigma_corrected = np.sqrt(mstar_sigma**2 - mstar_sigma_corr**2)
    plot_sigma_map(gal_ID, mstar_sigma, IMAGE_DIR=IMAGE_DIR, IMAGE_FORMAT=IMAGE_FORMAT)
    plot_sigma_map(gal_ID, mstar_sigma_corr, corr=True, IMAGE_DIR=IMAGE_DIR, 
                   IMAGE_FORMAT=IMAGE_FORMAT)

    ############################################################################
    # find the median velocity dispersion and uncertainty
    ############################################################################

    N = mstar_sigma_corrected.size
    n = (N-1)/2

    star_sigma_med = ma.median(mstar_sigma_corrected)
    star_sigma_med_err = ma.std(mstar_sigma_corrected) * \
        np.sqrt(np.pi * (2*n+1)/(4*n))
    
    return star_sigma_med, star_sigma_med_err, mstar_sigma, mstar_sigma_corr



def calculate_virial_mass(gal_ID, star_sigma, star_sigma_ivar, 
                          star_sigma_corr, star_sigma_mask, flux_map,
                           Ha_vel, Ha_vel_mask, r50, z, 
                          IMAGE_DIR=None, IMAGE_FORMAT='png' ):
    '''
    calculate virial mass 

    PARAMETERS
    ==========

    '''

    ############################################################################
    # get median velocity dispersion and uncertainty
    ############################################################################

    star_sigma_med, star_sigma_med_err, mstar_sigma, mstar_sigma_corr = median_star_sigma(gal_ID,
                                                           star_sigma, 
                                                           star_sigma_ivar, 
                                                           star_sigma_corr, 
                                                           star_sigma_mask,
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

    plot_diagnostic_panel(flux_map, mstar_sigma, mstar_sigma_corr, mvel, gal_ID, 
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

    logMvir = np.log10(7.5 * r50_kpc * star_sigma_med**2 / G)
    logMvir_err = 2 / (star_sigma_med * np.log(10)) * star_sigma_med_err

    return logMvir, logMvir_err, star_sigma_med, star_sigma_med_err
