'''
Functions to read in data
'''


################################################################################
# IMPORT MODULES
#-------------------------------------------------------------------------------
from astropy.io import fits
from astropy.table import Table

from parse_data import build_galaxy_dict
################################################################################






################################################################################
#-------------------------------------------------------------------------------
def read_master_file(filename):
    '''
    Read in the master file containing all the galaxy meta-data


    PARAMETERS
    ==========

    filename : string
        File name of the master file


    RETURNS
    =======

    master_table : astropy table
        Astropy table of the master file
    '''

    master_table = Table.read(filename)

    return master_table
################################################################################






################################################################################
#-------------------------------------------------------------------------------
def construct_filename(galaxy_ID, data_directory):
    '''
    Build the file name for a galaxy's data cube.


    PARAMETERS
    ==========

    galaxy_ID : length-2 tuple
        (plate, fiberID) for individual MaNGA galaxy

    data_directory : string
        Location of data cubes on computer system


    RETURNS
    =======

    cube_filename : string
        File name of data cube for MaNGA galaxy identified by the given 
        plate-fiberID.
    '''

    plate, fiberID = galaxy_ID

    cube_filename = data_directory + plate + '/manga-' + plate + '-' + fiberID 
                    + '-MAPS-HYB10-GAU-MILESHC.fits.gz'

    return cube_filename
################################################################################






################################################################################
#-------------------------------------------------------------------------------
def write_masses(masses, IDs, master_filename):
    '''
    Add mass data to master table.


    PARAMETERS
    ==========

    masses : numpy array of shape (N,)
        Array of galaxy masses, mass ratios, and associated errors.

    IDs : list of length-2 tuples
        List of (plate, IFU) galaxy IDs.  Length is the same as N, such that 
        masses[i] is the mass of galaxy IDs[i].

    master_filename : string
        File name of the master file
    '''


    ############################################################################
    # Read in master file
    #---------------------------------------------------------------------------
    master_table = read_master_file(master_filename)
    ############################################################################


    ############################################################################
    # Match up elliptical galaxies to table
    #---------------------------------------------------------------------------
    galaxy_dict = build_galaxy_dict(master_table)

    for i, galaxy in enumerate(IDs):

        j = galaxy_dict[galaxy]

        ########################################################################
        # Write data to table
        #-----------------------------------------------------------------------
        master_table['Mtot'][j] = masses['Mtot'][i]
        master_table['Mtot_error'][j] = masses['Mtot_err'][i]
        master_table['Mdark'][j] = masses['Mdark'][i]
        master_table['Mdark_error'][j] = masses['Mdark_err'][i]
        master_table['Mdark_Mstar_ratio'][j] = masses['Mdark_Mstar_ratio'][i]
        master_table['Mdark_Mstar_ratio_error'][j] = masses['Mdark_Mstar_ratio_err'][i]
        ########################################################################
    ############################################################################


    ############################################################################
    # Write master table to file
    #---------------------------------------------------------------------------
    master_table.write(master_filename[:-4] + '_ellipticals.txt', 
                       format='ascii.commented_header', overwrite=True)
    ############################################################################
################################################################################






################################################################################
#-------------------------------------------------------------------------------
def open_map(cube_filename, map_name):
    '''
    Import a masked map from the data cube.


    PARAMETERS
    ==========

    cube_filename : string
        File name of data cube for a MaNGA galaxy.

    map_name : string
        Field name of the map to import


    RETURNS
    =======

    masked_map : 2d masked numpy array
        Masked numpy array of the requested map from the data cube.
    '''


    ############################################################################
    # Import data cube
    #---------------------------------------------------------------------------
    cube = fits.open(cube_filename)
    ############################################################################


    ############################################################################
    # Extract (and correct for instrumental resolution effects) stellar velocity 
    # dispersion map
    #
    # See https://www.sdss.org/dr16/manga/manga-data/working-with-manga-data/ 
    # for correction details
    #---------------------------------------------------------------------------
    if map_name is 'STELLAR_SIGMA':

        # Data
        star_sigma = np.sqrt(cube['STELLAR_SIGMA'].data**2 
                             - cube['STELLAR_SIGMACORR'].data**2)

        # Mask extension
        star_sigma_mask_extension = cube['STELLAR_SIGMA'].header['QUALDATA']

        # Mask
        mStar_sigma = ma.array(star_sigma, mask=cube[star_sigma_mask_extension].data>0)

        masked_map = ma.masked_invalid(mStar_sigma)