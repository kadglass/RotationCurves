
import numpy as np

from astropy.table import Table

from extract_HI_functions import galaxies_dict

from extract_DRP_functions import match_DRP




def match_morph_visual(data):
    '''
    Locate the galaxy morphology from the MaNGA Visual Morphology Catalog and 
    add it to the given data table.


    PARAMETERS
    ==========

    data : astropy table (may or may not have quantities)
        Table of galaxies


    RETURNS
    =======

    data : astropy table (may or may not have quantities)
        Table of galaxies, with the added morphology data:
          - Hubble_type : Hubble classification
          - Bar : Degree of bar
              - 1 = conspicuous straight bar
              - 0.75 = clear conspicuous bar
              - 0.5 = clear bar in the inner regions of the galaxy
              - 0.25 = typically a roundish structure
              - 0 = no bar
              - -0.5 = difficult to distinguish
          - Tidal : Whether or not tidal debris is present (1 = present, 0 = not)
    '''

    ############################################################################
    # Initialize morphology columns in the data table
    #---------------------------------------------------------------------------
    data['Hubble_type'] = '      '
    #data['Bar'] = np.nan
    data['Tidal'] = -1
    ############################################################################


    ############################################################################
    # Load in morphology data
    #---------------------------------------------------------------------------
    data_directory = '/Users/kellydouglass/Documents/Research/data/SDSS/dr16/manga/morphology/manga_visual_morpho/1.0.1/'

    morph_filename = data_directory + 'manga_visual_morpho-1.0.1.fits'

    morph_data = Table.read(morph_filename, format='fits')

    #print(morph_data.colnames)
    ############################################################################


    ############################################################################
    # Build galaxy reference dictionary
    #---------------------------------------------------------------------------
    data_dict = galaxies_dict(data)
    ############################################################################


    ############################################################################
    # Insert morphology data into table
    #---------------------------------------------------------------------------
    for i in range(len(morph_data)):

        ########################################################################
        # Deconstruct galaxy ID
        #-----------------------------------------------------------------------
        plate, IFU = morph_data['plateifu'][i].split('-')
        ########################################################################


        if (int(plate), int(IFU)) in data_dict:
            ####################################################################
            # Find galaxy's row number in the data table
            #-------------------------------------------------------------------
            gal_i = data_dict[(int(plate), int(IFU))]
            ####################################################################


            ####################################################################
            # Insert morphology data into data table
            #-------------------------------------------------------------------
            data['Hubble_type'][gal_i] = morph_data['Type'][i]
            #data['Bar'][gal_i] = morph_data['BARS'][i]
            data['Tidal'][gal_i] = morph_data['tidal'][i]
            ####################################################################
    ############################################################################

    return data






def match_morph_gz(data):
    '''
    Locate the galaxy morphology from the MaNGA morphologies from Galaxy Zoo and 
    add it to the given data table.


    PARAMETERS
    ==========

    data : astropy table (may or may not have quantities)
        Table of galaxies


    RETURNS
    =======

    data : astropy table (may or may not have quantities)
        Table of galaxies, with the added morphology data:
          - GZ_edge_on : Likelihood that the galaxy is edge-on
          - GZ_bar : Likelihood of a bar
          - GZ_spiral : Likelihood that the galaxy is a spiral
    '''

    ############################################################################
    # Initialize morphology columns in the data table
    #---------------------------------------------------------------------------
    data['GZ_edge_on'] = np.nan
    data['GZ_bar'] = np.nan
    data['GZ_spiral'] = np.nan
    ############################################################################


    ############################################################################
    # Load in morphology data
    #---------------------------------------------------------------------------
    data_directory = '/Users/kellydouglass/Documents/Research/data/SDSS/dr15/manga/morphology/galaxyzoo/'

    morph_filename = data_directory + 'MaNGA_gz-v1_0_1.fits'

    morph_data = Table.read(morph_filename, format='fits')

    #print(morph_data.colnames)
    ############################################################################


    ############################################################################
    # Add the MaNGA ID number to the table (so that we can match to the Galaxy 
    # Zoo data table)
    #---------------------------------------------------------------------------
    data = match_DRP(data, ['mangaid'], ['S10'])
    ############################################################################


    ############################################################################
    # Build galaxy reference dictionary
    #---------------------------------------------------------------------------
    data_dict = {}

    for i in range(len(data)):

        galaxy_ID = data['mangaid'][i]

        data_dict[galaxy_ID] = i
    ############################################################################


    ############################################################################
    # Insert morphology data into table
    #---------------------------------------------------------------------------
    for i in range(len(morph_data)):

        if morph_data['MANGAID'][i] in data_dict:
            ####################################################################
            # Find galaxy's row number in the data table
            #-------------------------------------------------------------------
            gal_i = data_dict[morph_data['MANGAID'][i]]
            ####################################################################


            ####################################################################
            # Insert morphology data into data table
            #-------------------------------------------------------------------
            data['GZ_edge_on'][gal_i] = morph_data['t02_edgeon_a04_yes_weight_fraction'][i]
            data['GZ_bar'][gal_i] = morph_data['t03_bar_a06_bar_weight_fraction'][i]
            data['GZ_spiral'][gal_i] = morph_data['t04_spiral_a08_spiral_weight_fraction'][i]
            ####################################################################
    ############################################################################

    return data







def match_morph_dl(data):
    '''
    Locate the galaxy morphology from the MaNGA Deep Learning morphology catalog 
    and add it to the given data table.


    PARAMETERS
    ==========

    data : astropy table (may or may not have quantities)
        Table of galaxies


    RETURNS
    =======

    data : astropy table (may or may not have quantities)
        Table of galaxies, with the added morphology data:
          - GZ_edge_on : Likelihood that the galaxy is edge-on
          - GZ_bar : Likelihood of a bar
          - GZ_spiral : Likelihood that the galaxy is a spiral
    '''

    ############################################################################
    # Initialize morphology columns in the data table
    #---------------------------------------------------------------------------
    data['DL_ttype'] = np.nan
    data['DL_s0'] = np.nan
    data['DL_edge_on'] = np.nan
    data['DL_bar_GZ2'] = np.nan
    data['DL_bar_N10'] = np.nan
    data['DL_merge'] = np.nan
    ############################################################################


    ############################################################################
    # Load in morphology data
    #---------------------------------------------------------------------------
    data_directory = '/Users/kellydouglass/Documents/Research/data/SDSS/dr15/manga/morphology/deep_learning/1.0.1/'

    morph_filename = data_directory + 'manga-morphology-dl-DR15.fits'

    morph_data = Table.read(morph_filename, format='fits')

    #print(morph_data.colnames)
    ############################################################################


    ############################################################################
    # Build galaxy reference dictionary
    #---------------------------------------------------------------------------
    data_dict = galaxies_dict(data)
    ############################################################################


    ############################################################################
    # Insert morphology data into table
    #---------------------------------------------------------------------------
    for i in range(len(morph_data)):

        ########################################################################
        # Deconstruct galaxy ID
        #-----------------------------------------------------------------------
        plate, IFU = morph_data['PLATEIFU'][i].split('-')
        ########################################################################


        if (int(plate), int(IFU)) in data_dict:
            ####################################################################
            # Find galaxy's row number in the data table
            #-------------------------------------------------------------------
            gal_i = data_dict[(int(plate), int(IFU))]
            ####################################################################


            ####################################################################
            # Insert morphology data into data table
            #-------------------------------------------------------------------
            data['DL_ttype'][gal_i] = morph_data['TTYPE'][i]
            data['DL_s0'][gal_i] = morph_data['P_S0'][i]
            data['DL_edge_on'][gal_i] = morph_data['P_EDGE_ON'][i]
            data['DL_bar_GZ2'][gal_i] = morph_data['P_BAR_GZ2'][i]
            data['DL_bar_N10'][gal_i] = morph_data['P_BAR_N10'][i]
            data['DL_merge'][gal_i] = morph_data['P_MERG'][i]
            ####################################################################
    ############################################################################

    return data


