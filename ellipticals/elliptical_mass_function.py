################################################################################
#
# IMPORT MODULES
#
#-------------------------------------------------------------------------------

from read_data import construct_filename

from parse_data import find_ellipticals

################################################################################



def elliptical_masses(master_filename, data_directory, galaxy_ID):
    '''
    Parse through the data files and calculate the mass of elliptical galaxies.


    PARAMETERS
    ==========

    master_filename : string
        File name of the master table.  This table is a list of all the MaNGA 
        galaxies, along with their associated NSA data and other parameters 
        previously calculated by us.

    data_directory : string
        Location of the data stored on the local computer.

    galaxy_ID : string
        Either the plate-IFU identification for a particular MaNGA galaxy to be 
        analyzed, or 'all'.  If 'all', then all galaxies should be analyzed.
    '''


    if galaxy_ID is 'all':
        # Analyze all the elliptical galaxies in MaNGA

        elliptical_IDs = find_ellipticals(master_filename)