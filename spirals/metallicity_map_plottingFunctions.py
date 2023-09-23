import matplotlib.pyplot as plt
import numpy as np
import numpy.ma as ma

from metallicity_map_functions import linear_metallicity_gradient

from metallicity_map_broadband_functions import surface_brightness_profile



################################################################################
################################################################################
################################################################################

def plot_metallicity_map(IMAGE_DIR, metallicity_map, metallicity_map_ivar, gal_ID):

    #plt.imshow(metallicity_map, vmin=8,vmax=9)
    plt.imshow(metallicity_map)
    plt.gca().invert_yaxis()
    plt.title(gal_ID)
    plt.xlabel('spaxel')
    plt.ylabel('spaxel')
    plt.colorbar(label='12+log(O/H) (dex)')
    plt.savefig(IMAGE_DIR + 'metallicity/' + gal_ID + '_metallicity.eps')
    plt.close()

    plt.imshow(np.sqrt(1/metallicity_map_ivar), vmin=0,vmax=9)
    plt.gca().invert_yaxis()
    plt.title(gal_ID)
    plt.xlabel('spaxel')
    plt.ylabel('spaxel')
    plt.colorbar(label='$\sigma$ (dex)')
    plt.savefig(IMAGE_DIR + 'metallicity_sigma/' + gal_ID + '_metallicity_sigma.eps')
    plt.close()

################################################################################
################################################################################
################################################################################

def plot_metallicity_gradient(cov_dir, IMAGE_DIR, gal_ID, r, m, m_sigma, popt):

    
    grad, met_0 = popt

    r_depro = np.linspace(0, np.max(r), 1000)

    plt.scatter(r, m, zorder=0, color='k', alpha=0.2)
    #plt.scatter(bin_centers, m_median, color='b', zorder=1)
    plt.plot(r_depro, grad * r_depro + met_0, zorder=2, color='r')
    plt.ylim(np.min(m) - 0.05,np.max(m) + 0.05)
    plt.title(gal_ID)
    plt.xlabel('r [kpc]')
    plt.ylabel('12 + log(O/H) (dex)')
    plt.savefig(IMAGE_DIR + 'metallicity_gradient/' + gal_ID + '_metallicity_gradient.png')
    plt.close()



    '''

    cov = np.load(cov_dir + 'metallicity_' + gal_ID + '_cov.npy')

    N_samples = 1000

    random_sample = np.random.multivariate_normal(mean=[grad, 
                                                        met_0], 
                                                        cov=cov, 
                                                        size=N_samples)


    y_sample = np.zeros(( len(random_sample[:,0]), len(r_depro)))
    for i in range(len(random_sample[:,0])):
            # Calculate the values of the curve at this location
        y_sample[i] = linear_metallicity_gradient(r_depro, 
                                random_sample[:,0][i], 
                                random_sample[:,1][i])


    y = linear_metallicity_gradient(r_depro, grad, met_0)
    stdevs = np.nanstd(y_sample, axis=0)


    plt.plot(r_depro, y, 'c', zorder=2)
    #plt.fill_between(r_depro, y - stdevs, y + stdevs, facecolor='red', zorder=3)

    plt.errorbar(r, m, 
 #               yerr=m_sigma, 
                fmt='.', zorder=1, color='k')

    plt.text(2,9,'$12+log(O/H)_0 = $' + str(met_0) + '\n$grad = $' + str(grad), fontsize=10)



    plt.ylim(np.min(m) - 0.05,np.max(m) + 0.05)
    plt.title(gal_ID)
    plt.xlabel('r [kpc]')
    plt.ylabel('12 + log(O/H) (dex)')
    plt.axvline(0.4*3.35, color='r', label='$0.4\ R_{25}$')
    plt.legend()
    plt.savefig(IMAGE_DIR + 'metallicity_gradient/' + gal_ID + '_metallicity_gradient.eps')
    plt.close()
    '''


def plot_broadband_image(IMAGE_DIR, gal_ID, im_map, band):

    plt.imshow(im_map)
    plt.colorbar(label='[magnitude]')
    plt.gca().invert_yaxis()
    plt.xlabel('spaxel')
    plt.ylabel('spaxel')
    plt.title(gal_ID)
    plt.savefig(IMAGE_DIR + band + '_band/' + gal_ID + '_' + band + '_band.eps')
    plt.close()


def plot_surface_brightness(IMAGE_DIR, gal_ID, sb_mean, r_bins, r_pc, best_fit_vals, r25_pc, L_25):

    '''

    PARAMETERS
    ==========
    sb_mean : array
        binned surface brightness data

    r_bins : array
        binned radius

    r_pc : array
        radius data

    best_fit_vals : list
        best fit params for surface brightnes profile


    '''

    r = np.linspace(np.min(r_pc), np.max(r_pc), 1000)
    sb_model = np.zeros(len(r))

    for i in range(0, len(r)):
        sb_model[i] = surface_brightness_profile(best_fit_vals, r[i])

    plt.scatter(r_bins/1000, ma.log10(sb_mean), marker='.', color='k')
    plt.plot(r/1000, ma.log10(sb_model))
    plt.axvline(r25_pc/1000, color='r', label='$R_{25}$')
    plt.axhline(ma.log10(L_25), color='b', label='$L_{25}$')
    plt.legend()
    plt.xlabel('radius [kpc]')
    plt.ylabel('$log\Sigma_L\ (L\odot/pc^2)$')
    plt.title(gal_ID)
    plt.savefig(IMAGE_DIR + 'surface_brightness/' + gal_ID + '_surface_brightness.eps')
    plt.close()