import matplotlib.pyplot as plt
import numpy as np

from metallicity_map_functions import linear_metallicity_gradient



################################################################################
################################################################################
################################################################################

def plot_metallicity_map(IMAGE_DIR, metallicity_map, metallicity_map_ivar, gal_ID):

    plt.imshow(metallicity_map, vmin=8,vmax=9)
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


    plt.plot(r_depro, y, 'c', zorder=3)
    plt.fill_between(r_depro, y - stdevs, y + stdevs, facecolor='aliceblue', alpha=0.5, zorder=2)

    plt.errorbar(r, m, 
 #               yerr=m_sigma, 
                fmt='.', zorder=1, color='k')



    plt.ylim(np.min(m) - 0.5,np.max(m) + 0.5)
    plt.title(gal_ID)
    plt.xlabel('r [kpc]')
    plt.ylabel('12 + log(O/H) (dex)')
    plt.savefig(IMAGE_DIR + 'metallicity_gradient/' + gal_ID + '_metallicity_gradient.eps')
    plt.close()



def plot_broadband_image(IMAGE_DIR, gal_ID, im_map, band):

    plt.imshow(im_map)
    plt.colorbar(label='[magnitude]')
    plt.gca().invert_yaxis()
    plt.xlabel('spaxel')
    plt.ylabel('spaxel')
    plt.title(gal_ID)
    plt.savefig(IMAGE_DIR + band + '_band/' + gal_ID + '_' + band + '_band.eps')
    plt.close()


def plot_surface_brightness(IMAGE_DIR, gal_ID, sb, r_pc):

    plt.scatter(r_pc, np.log10(sb), marker='.', color='k')
    plt.xlabel('radius [pc]')
    plt.ylabel('$log\Sigma_L\ (L\odot/pc^2)$')
    plt.title(gal_ID)
    plt.savefig(IMAGE_DIR + 'surface_brightness/' + gal_ID + '_surface_brightness.eps')
    plt.close()