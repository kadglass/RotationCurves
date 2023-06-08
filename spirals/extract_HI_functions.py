
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

def match_HI( master_table):
    '''
    Locate the HI mass, velocity width for each galaxy


    PARAMETERS
    ==========

    master_table : astropy QTable
        Data table with N rows, each row containing one MaNGA galaxy for which 
        the rotation curve has been measured.


    RETURNS
    =======

    master_table : astropy QTable
        Same as the input master_table object, but with the additional HI mass 
        and velocity width columns:
          - logHI : log(M_HI) in units of log(M_sun)
          - WF50  : width of the HI line profile at 50% of the peak's height, 
                    measured from a fit to the line profile (units are km/s)
          - WP20  : width of the HI line profile at 20% of the peak's height 
                    (units are km/s)
    '''


    ############################################################################
    # Initialize HI columns in master_table
    #---------------------------------------------------------------------------
    master_table['logHI'] = np.nan*np.ones(len(master_table), dtype=float) * u.dex(u.M_sun)
    master_table['WF50'] = np.nan*np.ones(len(master_table), dtype=float) * (u.km/u.s)
    master_table['WP20'] = np.nan*np.ones(len(master_table), dtype=float) * (u.km/u.s)
    ############################################################################


    ############################################################################
    # Load in HI data
    #---------------------------------------------------------------------------
    # ALFALFA_filename = '/Users/kellydouglass/Documents/Research/data/SDSS/dr16/manga/HI/v1_0_2/manga_alfalfa-dr15.fits'
    # GBT_filename = '/Users/kellydouglass/Documents/Research/data/SDSS/dr16/manga/HI/v1_0_2/mangaHIall.fits'

    ALFALFA_filename = '/Users/nityaravi/Documents/Research/data/manga/manga_alfalfa-dr15.fits'
    GBT_filename = '/Users/nityaravi/Documents/Research/data/manga/mangaHIall.fits'

    ALFALFA = Table.read(ALFALFA_filename, format='fits')
    GBT = Table.read(GBT_filename, format='fits')
    ############################################################################


    ############################################################################
    # Build galaxy reference dictionary
    #---------------------------------------------------------------------------
    master_table_dict = galaxies_dict( master_table)
    ############################################################################


    ############################################################################
    # Insert ALFALFA measurements into table
    #---------------------------------------------------------------------------
    for i in range(len(ALFALFA)):

        ########################################################################
        # Deconstruct galaxy ID
        #-----------------------------------------------------------------------
        plate, IFU = ALFALFA['PLATEIFU'][i].split('-')
        ########################################################################


        if (int(plate), int(IFU)) in master_table_dict:
            ####################################################################
            # Find galaxy's row number in master_table
            #-------------------------------------------------------------------
            gal_i = master_table_dict[(int(plate), int(IFU))]
            ####################################################################


            ####################################################################
            # Calculate sin(i)
            #-------------------------------------------------------------------
            if 'ba_map' in master_table.colnames and master_table['ba_map'][gal_i] > 0:
                sini = np.sqrt(1 - master_table['ba_map'][gal_i]**2)
            else:
                sini = np.sqrt(1 - master_table['NSA_ba'][gal_i]**2)

            if sini == 0:
                sini = 1
            ####################################################################


            ####################################################################
            # Insert HI data into master table
            #-------------------------------------------------------------------
            master_table['logHI'][gal_i] = ALFALFA['LOGMHI'][i] * u.dex(u.M_sun)
            master_table['WF50'][gal_i] = ALFALFA['WF50'][i]/sini * (u.km/u.s)
            master_table['WP20'][gal_i] = ALFALFA['WP20'][i]/sini * (u.km/u.s)
            ####################################################################
    ############################################################################


    ############################################################################
    # Insert GBT measurements into table
    #---------------------------------------------------------------------------
    for i in range(len(GBT)):

        ########################################################################
        # Deconstruct galaxy ID
        #-----------------------------------------------------------------------
        plate, IFU = GBT['plateifu'][i].split('-')
        ########################################################################


        if (int(plate), int(IFU)) in master_table_dict:
            ####################################################################
            # Find galaxy's row number in master_table
            #-------------------------------------------------------------------
            gal_i = master_table_dict[(int(plate), int(IFU))]
            ####################################################################


            ####################################################################
            # Calculate sin(i)
            #-------------------------------------------------------------------
            if 'ba_map' in master_table.colnames and master_table['ba_map'][gal_i] > 0:
                sini = np.sqrt(1 - master_table['ba_map'][gal_i]**2)
            else:
                sini = np.sqrt(1 - master_table['NSA_ba'][gal_i]**2)

            if sini == 0:
                sini = 1
            ####################################################################


            ####################################################################
            # Insert HI data into master table
            #-------------------------------------------------------------------
            master_table['logHI'][gal_i] = GBT['logMHI'][i] * u.dex(u.M_sun)
            master_table['WF50'][gal_i] = GBT['WF50'][i]/sini * (u.km/u.s)
            master_table['WP20'][gal_i] = GBT['WP20'][i]/sini * (u.km/u.s)
            ####################################################################
    ############################################################################


    return master_table





################################################################################
################################################################################

def match_HI_dr2( master_table):
    '''
    Locate the HI mass, velocity width for each galaxy with data taken from the 
    DR2 of the HI-MaNGA survey.


    PARAMETERS
    ==========

    master_table : astropy QTable
        Data table with N rows, each row containing one MaNGA galaxy for which 
        the rotation curve has been measured.


    RETURNS
    =======

    master_table : astropy QTable
        Same as the input master_table object, but with the additional HI mass 
        and velocity width columns:
          - logHI : log(M_HI) in units of log(M_sun)
          - WF50  : width of the HI line profile at 50% of the peak's height, 
                    measured from a fit to the line profile (units are km/s)
          - WP20  : width of the HI line profile at 20% of the peak's height 
                    (units are km/s)
    '''


    ############################################################################
    # Initialize HI columns in master_table
    #---------------------------------------------------------------------------
    master_table['logHI'] = np.nan*np.ones(len(master_table), dtype=float)# * u.dex(u.M_sun)
    master_table['WF50'] = np.nan*np.ones(len(master_table), dtype=float)# * (u.km/u.s)
    master_table['WP20'] = np.nan*np.ones(len(master_table), dtype=float)# * (u.km/u.s)
    ############################################################################


    ############################################################################
    # Load in HI data
    #---------------------------------------------------------------------------
    #GBT_filename = '/Users/kellydouglass/Documents/Research/data/SDSS/dr16/manga/HI/v1_0_2/himanga_dr2.fits'
    GBT_filename='/Users/nityaravi/Documents/Research/RotationCurves/data/manga/mangaHIall.fits'
    GBT = Table.read(GBT_filename, format='fits')
    ############################################################################


    ############################################################################
    # Build galaxy reference dictionary
    #---------------------------------------------------------------------------
    master_table_dict = galaxies_dict( master_table)
    ############################################################################


    ############################################################################
    # Insert GBT measurements into table
    #---------------------------------------------------------------------------
    for i in range(len(GBT)):

        ########################################################################
        # Deconstruct galaxy ID
        #-----------------------------------------------------------------------
        plate, IFU = GBT['PLATEIFU'][i].split('-')
        ########################################################################


        if (int(plate), int(IFU)) in master_table_dict:
            ####################################################################
            # Find galaxy's row number in master_table
            #-------------------------------------------------------------------
            gal_i = master_table_dict[(int(plate), int(IFU))]
            ####################################################################


            ####################################################################
            # Calculate sin(i)
            #-------------------------------------------------------------------
            if 'ba_map' in master_table.colnames and master_table['ba_map'][gal_i] > 0:
                sini = np.sqrt((1 - master_table['ba_map'][gal_i]**2)/(1 - 0.2**2))
            else:
                sini = np.sqrt((1 - master_table['NSA_ba'][gal_i]**2)/(1 - 0.2**2))

            if sini == 0:
                sini = 1
            ####################################################################


            ####################################################################
            # Insert HI data into master table
            #-------------------------------------------------------------------
            master_table['logHI'][gal_i] = GBT['LOGMHI'][i]# * u.dex(u.M_sun)
            master_table['WF50'][gal_i] = GBT['WF50'][i]/sini# * (u.km/u.s)
            master_table['WP20'][gal_i] = GBT['WP20'][i]/sini# * (u.km/u.s)
            ####################################################################
    ############################################################################


    return master_table






################################################################################
################################################################################

def match_HI_dr3( master_table):

    '''
    Locate the HI mass, velocity width for each galaxy with data taken from the 
    DR2 of the HI-MaNGA survey.


    PARAMETERS
    ==========

    master_table : astropy QTable
        Data table with N rows, each row containing one MaNGA galaxy for which 
        the rotation curve has been measured.


    RETURNS
    =======

    master_table     : astropy QTable
        Same as the input master_table object, but with the additional HI mass 
        and velocity width columns:
          - logHI    : log(M_HI) in units of log(M_sun)
          - logHIlim : upper limit of HI mass in units of log(M_sun)
          - WF50     : width of the HI line profile at 50% of the peak's height, 
                    measured from a fit to the line profile (units are km/s)
          - WF50_err : uncertainty on width of HI line (km/s)
    '''


    ############################################################################
    # Initialize HI columns in master_table
    #---------------------------------------------------------------------------
    master_table['logHI'] = np.nan*np.ones(len(master_table), dtype=float)# * u.dex(u.M_sun)
    master_table['logHIlim'] = np.nan*np.ones(len(master_table), dtype=float) # * u.dex(u.M_sun)
    master_table['WF50'] = np.nan*np.ones(len(master_table), dtype=float)# * (u.km/u.s)
    master_table['WF50_err'] = np.nan*np.ones(len(master_table), dtype=float) # * (u.km/u.s)

    ############################################################################
    # Load in HI data
    #---------------------------------------------------------------------------
    GBT_filename= '/Users/nityaravi/Documents/Research/RotationCurves/data/manga/mangaHIall.fits'
    GBT = Table.read(GBT_filename, format='fits')
    
    
    
    ############################################################################
    # Find duplicates in HI table
    #---------------------------------------------------------------------------
    gal_IDs, counts = np.unique(GBT['PLATEIFU'], return_counts=True)
    duplicate_IDs = gal_IDs[np.where(counts > 1)[0]]


    ############################################################################
    # Insert HI measurements into table, if there are ALFALFA and GBT
    # measurements, use GBT
    #---------------------------------------------------------------------------
    for i in range(0, len(gal_IDs)):
        
        gal_ID = gal_IDs[i]

        ####################################################################
        # Find gal in master_table
        #-------------------------------------------------------------------
        i_master = np.where(master_table['plateifu'] == gal_ID)[0]


        ####################################################################
        # Find gal in GBT table
        #-------------------------------------------------------------------
        GBT_row = GBT[GBT['PLATEIFU'] == gal_ID]
        if len(GBT_row) > 1:
            GBT_row = GBT_row[GBT_row['SESSION'] != 'ALFALFA']



        ####################################################################
        # Calculate sin(i)
        #-------------------------------------------------------------------
        sini = np.sqrt((1 - master_table['nsa_elpetro_ba'][i_master]**2)/(1 - 0.2**2))

        if sini == 0:
            sini = 1
        ####################################################################


        master_table['logHI'][i_master] = GBT_row['LOGMHI']
        master_table['logHIlim'][i_master]= GBT_row['LOGHILIM200KMS']
        master_table['WF50'][i_master] = GBT_row['WF50'] / sini
        master_table['WF50_err'][i_master] = GBT_row['EV'] / sini

        if i % 50 == 0:
            print(i)

    return master_table