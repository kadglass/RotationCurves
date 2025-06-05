import os
from astropy.wcs import WCS
import requests


def get_cutout(plateifu, ra, dec, r90, cache_dir, layers=['default'], verbose=False):
    '''Take MaNGA galaxy ra, dec, r90 and generates a cutout from the legacy survey viewer
    with size 4*r90 by 4*r90. cutout is saved at cache_dir with file name plateifu.jpg.
    
    Parameters
    ----------
    plateifu : str
        plateifu for galaxy.
    ra : float
        Right ascension (degrees).
    dec : float
        Declination (degrees).
    r90 : float
        r90 in arcsec for galaxy, to be used for cutout size
    cache_dir : string
        cache location
    layers : list
        list of layers of legacy survey to generate cutouts from, by default
        generates cutout with default ls dr9 photometry
        'default' is dr9 phot
        'model' is dr9 model
        'resid' is dr9 residual
    verbose : bool
        Add some status messages if true.
        
    Returns
    -------
    img_name : str
        Name of JPG cutout file written after query.
    w : astropy.wcs.WCS
        World coordinate system for the image.
    '''

    layer_dict = {'default': 'ls-dr9',
                  'model': 'ls-dr9-model',
                  'resid': 'ls-dr9-resid'}
    

    for layer in layers:

        if not os.path.isdir(cache_dir + '/' + layer):
            os.makedirs(cache_dir + '/' + layer)


        img_name = cache_dir + '/' + layer + '/' + plateifu + '.jpg'
        
        size = int(4 * r90 / 0.262)

    
    
        if os.path.exists(img_name):
            if verbose:
                print('{} exists.'.format(img_name))


        else:
            img_url = 'https://www.legacysurvey.org/viewer/cutout.jpg?ra={}&dec={}&zoom=14&size={}&layer={}'.format(ra, dec, size, layer_dict[layer])
            if verbose:
                print('Get {}'.format(img_url))
                
            with open(img_name, 'wb') as handle: 
                response = requests.get(img_url, stream=True) 
                if not response.ok: 
                    print(response) 
                for block in response.iter_content(1024): 
                    if not block: 
                        break 
                    handle.write(block)
                
    # Set up the WCS.

    
    wcs_input_dict = {
        'CTYPE1': 'RA---TAN',
        'CUNIT1': 'deg',
        'CDELT1': -0.262/3600,
        'CRPIX1': size/2 + 0.5,
        'CRVAL1': ra,
        'NAXIS1': size,
        'CTYPE2': 'DEC--TAN',
        'CUNIT2': 'deg',
        'CDELT2': 0.262/3600,
        'CRPIX2': size/2 + 0.5,
        'CRVAL2': dec,
        'NAXIS2': size
    }
    w = WCS(wcs_input_dict)
    
    return img_name, w