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