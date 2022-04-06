
import astropy.units as u
from astropy.table import Table

import numpy as np


################################################################################
################################################################################

def galaxies_dict(ref_table):
    '''
    Built dictionary of (plate, IFU) tuples that refer to the galaxy's row 
    index in the ref_table.


    PARAMETERS
    ==========

    ref_table : astropy table
        Data table with columns
          - MaNGA_plate (int) : MaNGA plate number
          - MaNGA_IFU (int)   : MaNGA IFU number


    RETURNS
    =======

    ref_dict : dictionary
        Dictionary with keys (plate, IFU) and values are the row index in 
        ref_table
    '''


    # Initialize dictionary to store (plate, IFU) and row index
    ref_dict = {}


    for i in range(len(ref_table)):

        galaxy_ID = (ref_table['MaNGA_plate'][i], ref_table['MaNGA_IFU'][i])

        ref_dict[galaxy_ID] = i


    return ref_dict




################################################################################
################################################################################

def galaxies_dict_SDSS(ref_table):
    '''
    Built dictionary of (plate, MJD, fiber) tuples that refer to the galaxy's 
    row index in the ref_table.


    PARAMETERS
    ==========

    ref_table : astropy table
        Data table with columns
          - NSA_plate (int)   : SDSS plate number
          - NSA_MJD (int)     : SDSS MJD number
          - NSA_fiberID (int) : SDSS fiber ID number


    RETURNS
    =======

    ref_dict : dictionary
        Dictionary with keys (plate, MJD, fiber) and values are the row index in 
        ref_table
    '''


    # Initialize dictionary to store (plate, IFU) and row index
    ref_dict = {}


    for i in range(len(ref_table)):

        galaxy_ID = (ref_table['NSA_plate'][i], ref_table['NSA_MJD'][i], ref_table['NSA_fiberID'][i])

        ref_dict[galaxy_ID] = i


    return ref_dict




################################################################################
################################################################################

def match_H2_MASCOT(master_table, units=False):
    '''
    Locate the H2 mass for each galaxy from the MASCOT survey.


    PARAMETERS
    ==========

    master_table : astropy QTable
        Data table with N rows, each row containing one MaNGA galaxy for which 
        the rotation curve has been measured.

    units : boolean
        Whether or not to include units in the data columns (necessary if the 
        table is a QTable).  Default is False - no units will be added.


    RETURNS
    =======

    master_table : astropy QTable
        Same as the input master_table object, but with the additional H2 mass 
        column:
          - logH2 : log(M_H2) in units of log(M_sun)
          - H2_source : survey source of the H2 mass (1 == MASCOT)
    '''


    ############################################################################
    # Initialize H2 column in master_table
    #---------------------------------------------------------------------------
    if 'logH2' not in master_table.colnames:

        if units:
            master_table['logH2'] = np.nan*np.ones(len(master_table), 
                                                   dtype=float) * u.dex(u.M_sun)
            master_table['logH2_err'] = np.nan*np.ones(len(master_table), 
                                                       dtype=float) * u.dex(u.M_sun)
        else:
            master_table['logH2'] = np.nan*np.ones(len(master_table), 
                                                   dtype=float)
            master_table['logH2_err'] = np.nan*np.ones(len(master_table), 
                                                       dtype=float)

        master_table['H2_source'] = np.zeros(len(master_table), dtype=int)
    ############################################################################


    ############################################################################
    # Load in H2 data
    #---------------------------------------------------------------------------
    MASCOT_filename = '/Users/kellydouglass/Documents/Research/data/H2/MASCOT/MASCOT-dr1.fits'

    MASCOT = Table.read(MASCOT_filename, format='fits')
    ############################################################################


    ############################################################################
    # Build galaxy reference dictionary
    #---------------------------------------------------------------------------
    master_table_dict = galaxies_dict(master_table)
    ############################################################################


    ############################################################################
    # Insert MASCOT measurements into table
    #---------------------------------------------------------------------------
    for i in range(len(MASCOT)):

        ########################################################################
        # Deconstruct galaxy ID
        #-----------------------------------------------------------------------
        plate, IFU = MASCOT['MaNGA ID'][i].split('-')
        ########################################################################


        if (int(plate), int(IFU)) in master_table_dict:
            ####################################################################
            # Find galaxy's row number in master_table
            #-------------------------------------------------------------------
            gal_i = master_table_dict[(int(plate), int(IFU))]
            ####################################################################


            ####################################################################
            # Insert H2 data into master table
            #-------------------------------------------------------------------
            if units:
                master_table['logH2'][gal_i] = MASCOT['log(H2_mass)'][i] * u.dex(u.M_sun)
            else:
                master_table['logH2'][gal_i] = MASCOT['log(H2_mass)'][i]

            master_table['H2_source'][gal_i] = 1
            ####################################################################
    ############################################################################


    return master_table





################################################################################
################################################################################

def match_H2_ALMaQUEST(master_table, units=False):
    '''
    Locate the H2 mass from the ALMaQUEST survey.


    PARAMETERS
    ==========

    master_table : astropy QTable
        Data table with N rows, each row containing one MaNGA galaxy for which 
        the rotation curve has been measured.

    units : boolean
        Whether or not to include units in the data columns (necessary if the 
        table is a QTable).  Default is False - no units will be added.


    RETURNS
    =======

    master_table : astropy QTable
        Same as the input master_table object, but with the additional H2 mass 
        column:
          - logH2 : log(M_H2) in units of log(M_sun)
          - H2_source : survey source of the H2 mass (2 == ALMaQUEST)
    '''


    ############################################################################
    # Initialize H2 columns in master_table
    #---------------------------------------------------------------------------
    if 'logH2' not in master_table.colnames:

        if units:
            master_table['logH2'] = np.nan*np.ones(len(master_table), 
                                                   dtype=float) * u.dex(u.M_sun)
            master_table['logH2_err'] = np.nan*np.ones(len(master_table), 
                                                       dtype=float) * u.dex(u.M_sun)
        else:
            master_table['logH2'] = np.nan*np.ones(len(master_table), 
                                                   dtype=float)
            master_table['logH2_err'] = np.nan*np.ones(len(master_table), 
                                                       dtype=float)

        master_table['H2_source'] = np.zeros(len(master_table), dtype=int)
    ############################################################################


    ############################################################################
    # Load in H2 data
    #---------------------------------------------------------------------------
    ALMaQUEST_filename = '/Users/kellydouglass/Documents/Research/data/H2/ALMaQUEST/table2.dat'

    ALMaQUEST = Table.read(ALMaQUEST_filename, 
                           format='ascii.no_header', 
                           names=['PLATEIFU', 'AREA', 'LOGMSUN', 'SFR', 'S_CO', 'eS_CO', 'LOGH2', 'eLOGH2', 'SSFR', 'SFE', 'fH2'])
    ############################################################################


    ############################################################################
    # Build galaxy reference dictionary
    #---------------------------------------------------------------------------
    master_table_dict = galaxies_dict(master_table)
    ############################################################################


    ############################################################################
    # Insert H2 measurements into table
    #---------------------------------------------------------------------------
    for i in range(len(ALMaQUEST)):

        ########################################################################
        # Deconstruct galaxy ID
        #-----------------------------------------------------------------------
        plate, IFU = ALMaQUEST['PLATEIFU'][i].split('-')
        ########################################################################


        if (int(plate), int(IFU)) in master_table_dict:
            ####################################################################
            # Find galaxy's row number in master_table
            #-------------------------------------------------------------------
            gal_i = master_table_dict[(int(plate), int(IFU))]
            ####################################################################


            ####################################################################
            # Insert H2 data into master table
            #-------------------------------------------------------------------
            if units:
                master_table['logH2'][gal_i] = ALMaQUEST['LOGH2'][i] * u.dex(u.M_sun)
                master_table['logH2_err'][gal_i] = ALMaQUEST['eLOGH2'][i] * u.dex(u.M_sun)
            else:
                master_table['logH2'][gal_i] = ALMaQUEST['LOGH2'][i]
                master_table['logH2_err'][gal_i] = ALMaQUEST['eLOGH2'][i]

            master_table['H2_source'][gal_i] = 2
            ####################################################################
    ############################################################################


    return master_table





################################################################################
################################################################################

def match_H2_xCOLDGASS(master_table, units=False):
    '''
    Locate the H2 mass from the xCOLDGASS survey.


    PARAMETERS
    ==========

    master_table : astropy QTable
        Data table with N rows, each row containing one MaNGA galaxy for which 
        the rotation curve has been measured.

    units : boolean
        Whether or not to include units in the data columns (necessary if the 
        table is a QTable).  Default is False - no units will be added.


    RETURNS
    =======

    master_table : astropy QTable
        Same as the input master_table object, but with the additional H2 mass 
        columns:
          - logH2 : log(M_H2) in units of log(M_sun)
          - logH2_err : Error on the log(M_H2) in units of log(M_sun)
          - H2_source : survey source of the H2 mass (3 == xCOLDGASS)
    '''

    ############################################################################
    # Initialize H2 columns in master_table
    #---------------------------------------------------------------------------
    if 'logH2' not in master_table.colnames:

        if units:
            master_table['logH2'] = np.nan*np.ones(len(master_table), 
                                                   dtype=float) * u.dex(u.M_sun)
            master_table['logH2_err'] = np.nan*np.ones(len(master_table), 
                                                       dtype=float) * u.dex(u.M_sun)
        else:
            master_table['logH2'] = np.nan*np.ones(len(master_table), 
                                                   dtype=float)
            master_table['logH2_err'] = np.nan*np.ones(len(master_table), 
                                                       dtype=float)

        master_table['H2_source'] = np.zeros(len(master_table), dtype=int)
    ############################################################################


    ############################################################################
    # Load in H2 data
    #---------------------------------------------------------------------------
    xCOLDGASS_filename = '/Users/kellydouglass/Documents/Research/data/H2/xCOLDGASS/xCOLDGASS_PubCat.fits'

    xCOLDGASS = Table.read(xCOLDGASS_filename, format='fits')
    ############################################################################


    ############################################################################
    # Build galaxy reference dictionary
    #---------------------------------------------------------------------------
    master_table_dict = galaxies_dict_SDSS(master_table)
    ############################################################################


    ############################################################################
    # Insert H2 measurements into table
    #---------------------------------------------------------------------------
    for i in range(len(xCOLDGASS)):

        ########################################################################
        # Extract galaxy's plate-MJD-fiber
        #-----------------------------------------------------------------------
        plate = xCOLDGASS['PLATEID'][i]
        MJD = xCOLDGASS['MJD'][i]
        fiberID = xCOLDGASS['FIBERID'][i]
        ########################################################################


        if (plate, MJD, fiberID) in master_table_dict:
            ####################################################################
            # Find galaxy's row number in master_table
            #-------------------------------------------------------------------
            gal_i = master_table_dict[(plate, MJD, fiberID)]
            ####################################################################


            ####################################################################
            # Insert H2 data into master table
            #-------------------------------------------------------------------
            if units:
                master_table['logH2'][gal_i] = xCOLDGASS['LOGMH2'][i] * u.dex(u.M_sun)
                master_table['logH2_err'][gal_i] = xCOLDGASS['LOGMH2_ERR'][i] * u.dex(u.M_sun)
            else:
                master_table['logH2'][gal_i] = xCOLDGASS['LOGMH2'][i]
                master_table['logH2_err'][gal_i] = xCOLDGASS['LOGMH2_ERR'][i]

            master_table['H2_source'][gal_i] = 3
            ####################################################################
    ############################################################################


    return master_table










