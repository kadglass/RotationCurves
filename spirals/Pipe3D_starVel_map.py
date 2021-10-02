import numpy as np

from astropy.io import fits


################################################################################
################################################################################
################################################################################

def extract_Pipe3D_data(VEL_MAP_FOLDER, gal_ID):
    '''
    Open the Pipe3D .fits file and extract the stellar velocity map data.


    PARAMETERS
    ==========

    VEL_MAP_FOLDER : string
        Path to the Pipe3D fits files

    gal_ID : string
        <MaNGA plate> - <MaNGA IFU>


    RETURNS
    =======

    star_vel : ndarray of shape (n,n)
        Derived stellar velocity map in units of km/s

    star_vel_err : ndarray of shape (n,n)
        Derived error in the stellar velocity map in units of km/s

    '''

    plate, IFU = gal_ID.split('-')

    file_name = VEL_MAP_FOLDER + plate + '/manga-' + gal_ID + '.Pipe3D.cube.fits.gz'

    ############################################################################
    # Open the fits file
    #---------------------------------------------------------------------------
    main_file = fits.open(file_name)

    ssp = main_file[1].data

    main_file.close()
    ############################################################################


    ############################################################################
    # Extract the stellar velocity maps
    #---------------------------------------------------------------------------
    star_vel = ssp[13]     # km/s
    star_vel_err = ssp[14] # km/s
    ############################################################################


    return star_vel, star_vel_err