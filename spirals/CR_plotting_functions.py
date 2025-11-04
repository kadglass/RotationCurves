import os
import numpy as np
from astropy.wcs import WCS
from astropy import wcs
import matplotlib as mpl
import matplotlib.pyplot as plt
from astropy.coordinates import SkyCoord
import astropy.units as u



mpl.rcParams['figure.dpi'] = 100
mpl.rcParams['savefig.dpi'] = 300
mpl.rcParams['figure.titlesize'] = 20
mpl.rcParams['axes.labelsize'] = 20
mpl.rcParams['xtick.labelsize'] = 15


def plot_legacy_survey_VI(gal_ID, center_ra, center_dec, phi, r90, ba, w, img_dir):

    '''
    Take saved legacy survey cutouts and make row of plots 
    [image, model, residual] for VI

    PARAMETERS
    ----------
    gal_ID : string
        galaxy plate-ifu

    center_ra : float [deg]
        ra of center of galaxy

    center_dec : float [deg]
        dec of center of galaxy

    phi : float [deg]
        position angle of galaxy

    r90 : float [arcsec]
        r90 of galaxy

    ba : float
        axis ratio

    w : astropy.wcs.WCS
        World coordinate system for the image.    
        
    img_dir : string
        location of cutouts, images will be saved here

    
    
    '''


    centers = SkyCoord(center_ra*u.deg, center_dec*u.deg)

    phi = phi*u.deg
    r90 = r90*u.arcsec

    x = 1 # Determined during SV

    # Maximum distance along the semi-major axis from the center coordinate for our endpoints
    delta_a = x*r90
    delta_b = delta_a * ba

    # Target positions
    major_1 = centers.directional_offset_by(phi, x*delta_a)
    major_2 = centers.directional_offset_by(phi + 180*u.deg, x*delta_a)
    minor_1 = centers.directional_offset_by(phi-90*u.deg, x *delta_b)
    minor_2 = centers.directional_offset_by(phi +90*u.deg, x *delta_b)

    major_ra = [major_1.ra.value, major_2.ra.value]
    major_dec = [major_1.dec.value, major_2.dec.value]

    minor_ra = [minor_1.ra.value, minor_2.ra.value]
    minor_dec = [minor_1.dec.value, minor_2.dec.value]

    img = mpl.image.imread(img_dir + '/default/' + gal_ID + '.jpg')
    model =  mpl.image.imread(img_dir + '/model/' + gal_ID + '.jpg')
    resid =  mpl.image.imread(img_dir + '/resid/' + gal_ID + '.jpg')

    ############################################################################


    fig = plt.figure(figsize=(25,10))
    ax1 = fig.add_subplot(1,3,1, projection=w)

    ax1.imshow(np.flip(img, axis=0))
    ax1.set(xlabel='ra', ylabel='dec')
    ax1.plot(major_ra, major_dec, transform=ax1.get_transform('world'), color='r', linestyle='--', linewidth=2)
    ax1.plot(minor_ra, minor_dec, transform=ax1.get_transform('world'), color='tab:green', linestyle='-', linewidth=2)


    ax2 = fig.add_subplot(1,3,2, projection=w)
    ax2.set(xlabel='ra', ylabel=' ')
    ax2.imshow(np.flip(model, axis=0))

    ax3 = fig.add_subplot(1,3,3, projection=w)
    ax3.set(xlabel='ra', ylabel=' ')
    ax3.imshow(np.flip(resid, axis=0))

    fig.suptitle(gal_ID)
    fig.tight_layout()


    if not os.path.isdir(img_dir + '/VI'):
            os.makedirs(img_dir + '/VI')

    fig.savefig(img_dir + '/VI/' + gal_ID + '.png', bbox_inches='tight')
    plt.close(fig)