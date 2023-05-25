import numpy as np

from astropy.io import fits

import astropy.units as u


################################################################################
################################################################################
################################################################################

def extract_Pipe3D_data(VEL_MAP_FOLDER, gal_ID, include_maps):
    '''
    Open the Pipe3D .fits file and extract the stellar velocity map data.


    PARAMETERS
    ==========

    VEL_MAP_FOLDER : string
        Path to the Pipe3D fits files

    gal_ID : string
        <MaNGA plate> - <MaNGA IFU>

    include_maps : list
        Map names to import 


    RETURNS
    =======

    maps : dictionary
        Each field contains the data maps requested.  Possiblities include
          - star_vel: derived stellar velocity map [km/s]
          - star_vel_err: derived error in the stellar velocity map [km/s]
    '''

    plate, IFU = gal_ID.split('-')

    #file_name = VEL_MAP_FOLDER + plate + '/manga-' + gal_ID + '.Pipe3D.cube.fits.gz'
    file_name = VEL_MAP_FOLDER + '/manga-' + gal_ID + '.Pipe3D.cube.fits.gz'

    ############################################################################
    # Open the fits file
    #---------------------------------------------------------------------------
    main_file = fits.open(file_name)

    ssp = main_file[1].data
    #flux_emlines = main_file[3].data
    flux_emlines = main_file[4].data

    main_file.close()
    ############################################################################


    maps = {}


    ############################################################################
    # Extract the stellar velocity maps
    #---------------------------------------------------------------------------
    if 'star_vel' in include_maps:
        maps['star_vel'] = ssp[13]     # km/s
        maps['star_vel_err'] = ssp[14] # km/s
    ############################################################################


    ############################################################################
    # Extract the V-band images
    #---------------------------------------------------------------------------
    if 'v_band' in include_maps:
        maps['v_band'] = ssp[0]     # erg/s/cm^2
        maps['v_band_err'] = ssp[4] # erg/s/cm^2
    ############################################################################


    ############################################################################
    # Extract the stellar mass density map
    #---------------------------------------------------------------------------
    if 'sMass_density' in include_maps:
        maps['sMass_density'] = ssp[19]*u.dex(u.M_sun) # log10(Msun/spaxel^2)
    ############################################################################


    ############################################################################
    # Extract the H-alpha velocity maps
    #---------------------------------------------------------------------------
    if 'Ha_vel' in include_maps:
        maps['Ha_vel'] = flux_emlines[0]     # km/s
        maps['Ha_vel_err'] = flux_emlines[1] # km/s
    ############################################################################


    return maps