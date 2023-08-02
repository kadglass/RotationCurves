import matplotlib.pyplot as plt

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

def plot_metallicity_gradient(IMAGE_DIR, gal_ID):
    return