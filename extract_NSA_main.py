'''
Extract various data values from the NSA catalog.
'''


################################################################################
#
#   IMPORT MODULES
#
################################################################################


from astropy.table import Table

import numpy as np


################################################################################
#
#   FILE NAMES
#
################################################################################


# File name of data to be matched
data_filename = 'master_table.txt'

# File name of NSA catalog
NSA_filename = '/Users/kellydouglass/Documents/Drexel/Research/Data/nsa_v0_1_2.fits'