
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

        if GBT_row['WF50'] < 0:
            master_table['WF50'][i_master] = -999
            master_table['WF50_err'][i_master] = -999
        
        else:
            master_table['WF50'][i_master] = GBT_row['WF50'] / sini
            master_table['WF50_err'][i_master] = GBT_row['EV'] / sini

        if i % 50 == 0:
            print(i)

    return master_table

def calculate_HI_R90(logMHI, R90):
    '''
    Calculate the HI mass within R90 using the method in Wang et al. 2020. First
    calculate the HI radius r_HI. If R90 > 1.5 r_HI, thne the total HI mass is
    returned. Else, the HI mass inside R90 is approximated by subtracting off
    the HI mass between R90 and 1.5 r_HI assuming an exponential profile.

    PARAMETERS
    ==========
    logMHI : float
        HI mass in units of log(M_sun)
    R90 : float
        R90 in units of kpc

    RETURNS
    =======
    logMHI_90 : float
        HI mass within R90 in units of log(M_sun)
    
    logMHI_90_err : float
        uncertainty on logMHI_90



    '''

    a = 0.506
    a_err = 0.003
    b = -3.293
    b_err = 0.009
    Sigma_0 = np.exp(5) # [M_sun / pc^2]

    r_HI = 0.5 * 10**(a * logMHI + b)


    if R90 >= 1.5* r_HI:
        return logMHI, 0 

    else:

        MHI_out = -2 * np.pi * Sigma_0 * (np.exp(-1.5 / 0.2) * 0.2 * 1.7 * r_HI**2 \
                                   - np.exp(-R90 / (0.2 * r_HI)) * 0.2 * r_HI * (R90 + 0.2 * r_HI))
        
        logMHI_in = np.log10(10**logMHI - MHI_out)

        # full uncertainties

        #r_HI_err2 = (np.log(10) * r_HI)**2 * \
        #    (logMHI**2 * a_err**2 + a**2 * logMHI_err**2 + b_err**2)

        #MHI_out_err2 = 4 * np.exp(-10*R90/r_HI) * np.pi**2 * Sigma_0**2 \
        #    * (R90**2 * R90_err**2) \
        #        + (R90**2 + 0.4*R90*r_HI +\
        #            (0.08 - 0.28 * np.exp(-1.5/2) * np.exp(5*R90/R)) * r_HI**2 * r_HI_err2) \
        #                / r_HI**2

        #logMHI_in_err = 1/(10**logMHI - MHI_out) \
        #    * np.sqrt((10**logMHI * logMHI_err)**2 + MHI_out_err2 / np.log(10)**2)

        # since logMHI_err = 0 and R90_err = 0

        r_HI_err2 = (np.log(10) * r_HI)**2 * (logMHI**2 * a_err**2  + b_err**2)
        
        MHI_out_err2 = 4 * np.exp(-10*R90/r_HI) * np.pi**2 * Sigma_0**2 \
            * (R90**2 + 0.4*R90*r_HI +\
                    (0.08 - 0.28 * np.exp(-1.5/2) * np.exp(5*R90/r_HI)) * r_HI**2)**2 * r_HI_err2 \
                        / r_HI**2
        
        logMHI_in_err = 1/(10**logMHI - MHI_out) * np.sqrt(MHI_out_err2 / np.log(10)**2)


        return logMHI_in, logMHI_in_err
