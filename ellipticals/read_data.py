'''
Functions to read in data
'''


################################################################################
#
# IMPORT MODULES
#
#-------------------------------------------------------------------------------

from astropy.table import Table

################################################################################



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




def construct_filename(galaxy_ID, data_directory):
    '''
    Build the file name for a galaxy's data cube.


    PARAMETERS
    ==========

    galaxy_ID : string
        plate-fiberID for individual MaNGA galaxy

    data_directory : string
        Location of data cubes on computer system


    RETURNS
    =======

    cube_filename : string
        File name of data cube for MaNGA galaxy identified by the given 
        plate-fiberID.
    '''

    [plate, fiberID] = galaxy_ID.split('-')

    cube_filename = data_directory + plate + '/manga-' + plate + '-' + fiberID 
                    + '-MAPS-HYB10-GAU-MILESHC.fits.gz'

    return cube_filename




    