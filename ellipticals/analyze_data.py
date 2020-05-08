################################################################################
# IMPORT MODULES
#-------------------------------------------------------------------------------
import numpy as np
import numpy.ma as ma

from astropy.io import fits
from astropy.table import Table

from IO_data import construct_filename, open_map
from parse_data import find_data_DRPall
from plot_data import FJ_plot
################################################################################






################################################################################
# CONSTANTS
#-------------------------------------------------------------------------------
Msun = 1.989e30 # kg
c = 3e5    		# km/s
G = 6.674e-11   # m^3 kg^-1 s^-2

h = 1
H0 = 100*h 		# km/s/Mpc
################################################################################




################################################################################
#-------------------------------------------------------------------------------
def median_err(ivar):
    '''
    Calculate the uncertainty in the median of a set of values.


    PARAMETERS
    ==========

    ivar : numpy array of shape (n,n)
        Inverse variances of the array of which the median was determined


    RETURNS
    =======

    med_err : float
        Uncertainty in the median
    '''


    N = ivar.size

    mean_ivar = np.sum(ivar)

    med_err = np.sqrt(np.pi*(2*N + 1)/(4*N*mean_ivar))

    return med_err
################################################################################





################################################################################
#-------------------------------------------------------------------------------
def find_veldisp(ID, directory, disp):
    '''
    Extract galaxy velocity dispersion from stellar velocity dispersion map.


    PARAMETERS
    ==========

    ID : length-2 tuple
    	Galaxy (plate, fiberID) identification

    directory : string
    	Path to where data lives on local machine

    disp : string
    	Location / type of velocity dispersion.  Options include:
    	- 'median'  : returns the median value of the velocity dispersion map
    	- 'central' : returns the central value of the velocity dispersion map


    RETURNS
    =======

    vel_disp : float
    	Velocity dispersion of galaxy, in units of km/s

    vel_disp_err : float
        Uncertainty in the velocity dispersion of the galaxy, in units of km/s
    '''


    ############################################################################
    # Construct file name of galaxy data cube
    #---------------------------------------------------------------------------
    cube_filename = construct_filename(ID, directory)
    ############################################################################


    ############################################################################
    # Import the masked velocity dispersion map, masked r-band image
    #---------------------------------------------------------------------------
    vel_disp_map = open_map(cube_filename, 'STELLAR_SIGMA')
    vel_disp_ivar_map = open_map(cube_filename, 'STELLAR_SIGMA_IVAR')

    r_band_image = open_map(cube_filename, 'SPX_MFLUX')
    ############################################################################


    if disp is 'median':
        ########################################################################
        # Find the median velocity dispersion of the stellar velocity dispersion 
        # map.
        #-----------------------------------------------------------------------
        vel_disp = ma.median(vel_disp_map)

        vel_disp_err = median_err(vel_disp_ivar_map)
        ########################################################################

    else:
        ########################################################################
        # Find the central velocity dispersion of the galaxy
        #
        # Center of galaxy is defined as the spaxel with the maximum luminosity
        #-----------------------------------------------------------------------
        center_spaxel = np.unravel_index(ma.argmax(r_band_image, axis=None), 
        								 r_band_image.shape)

        vel_disp = vel_disp_map[center_spaxel]

        vel_disp_err = 1/np.sqrt(vel_disp_ivar_map[center_spaxel])
        ########################################################################

    '''
    ############################################################################
    # Convert from km/s to m/s
    #---------------------------------------------------------------------------
    vel_disp_mpers = vel_disp*1000
    ############################################################################
    '''

    return vel_disp, vel_disp_err
################################################################################





################################################################################
#-------------------------------------------------------------------------------
def plot_FaberJackson(IDs, directory, sigma_type, save_fig):
    '''
    Extract data and plot Faber-Jackson relation (stellar mass v. velocity 
    dispersion).


    PARAMETERS
    ==========

    IDs : list of length-2 tuples
        List of galaxy (plate, fiberID) combinations to include in Faber-Jackson 
        plot.

    directory : string
        Path to where data lives on local machine

    sigma_type : string
        Location / type of velocity dispersion.  Options include:
        - 'median'  : returns the median value of the velocity dispersion map
        - 'central' : returns the central value of the velocity dispersion map

    save_fig : boolean
        Determines wether or not to save the figure.
    '''


    ############################################################################
    # Read in DRPall table
    #---------------------------------------------------------------------------
    drp_filename = '../data/MaNGA/drpall-v2_4_3.fits'

    DRPall = fits.open(drp_filename)

    DRPall_table = DRPall[1].data
    ############################################################################


    ############################################################################
    # Extract stellar mass, velocity dispersion for each galaxy
    #
    # Stellar masses come from the NSA catalog via the DRPall file
    # Velocity dispersions are taken from the stellar velocity dispersion map.
    #---------------------------------------------------------------------------
    Mstar = np.zeros(len(IDs))

    vel_disp = np.zeros(len(IDs))
    vel_disp_err = np.zeros(len(IDs))


    for i,galaxy in enumerate(IDs):

        Mstar[i] = find_data_DRPall(DRPall_table, galaxy, 'nsa_elpetro_mass')

        vel_disp[i], vel_disp_err[i] = find_veldisp(galaxy, directory, sigma_type)
    ############################################################################


    ############################################################################
    # Plot Faber-Jackson relation
    #---------------------------------------------------------------------------
    FJ_plot(Mstar, vel_disp, sigma_type, save_fig)
    ############################################################################
################################################################################






################################################################################
#-------------------------------------------------------------------------------
def virial_mass(star_sigma, star_sigma_err, r_half, r_half_err):
    '''
    Calculate the virial mass of a galaxy.


    PARAMETERS
    ==========

    star_sigma : float or numpy array
        stellar velocity dispersion in m/s

    star_sigma_err : float or numpy array
        Uncertainty in the velocity dispersion in m/s

    r_half : float or numpy array
        Half-light radius in m

    r_half_err : float or numpy array
        Uncertainty in the half-light radius in m


    RETURNS
    =======

    Mvir_Msun : float or numpy array
        Virial mass in solar masses

    Mvir_err : float or numpy array
        Uncertainty in the virial mass in solar masses
    '''


    ############################################################################
    # Virial mass (in kg)
    #---------------------------------------------------------------------------
    Mvir = 7.5*star_sigma*star_sigma*r_half/G

    # Convert kg to Msun
    Mvir_Msun = Mvir/Msun
    ############################################################################


    ############################################################################
    # Uncertainty in the virial mass
    #---------------------------------------------------------------------------
    Mvir_err = Mvir*np.sqrt( (4*star_sigma_err*star_sigma_err/(star_sigma*star_sigma)) \
               + (r_half_err*r_half_err/(r_half*r_half)))
    ############################################################################

    return Mvir_Msun, Mvir_err
################################################################################






################################################################################
#-------------------------------------------------------------------------------
def determine_masses(IDs, directory):
    '''
    Extract data and calculate the total, dark matter mass of the elliptical 
    galaxies.


    PARAMETERS
    ==========

    IDs : length-N list of length-2 tuples
        List of galaxy (plate, fiberID) combinations

    directory : string
        Path to where data lives on local machine


    RETURNS
    =======

    mass_table : astropy table of length N
        Table containing all calculated masses, ratios, and associated errors.
    '''


    ############################################################################
    # Read in DRPall table
    #---------------------------------------------------------------------------
    drp_filename = '../data/MaNGA/drpall-v2_4_3.fits'

    DRPall = fits.open(drp_filename)

    DRPall_table = DRPall[1].data
    ############################################################################


    ############################################################################
    # Calculate virial mass for each galaxy
    #
    # Note: There is no given uncertainty to the half-light radius.
    #---------------------------------------------------------------------------
    vel_disp = np.zeros(len(IDs))
    vel_disp_err = np.zeros(len(IDs))

    r_half = np.zeros(len(IDs))
    r_half_err = np.zeros(len(IDs))

    for i,galaxy in enumerate(IDs):

        #-----------------------------------------------------------------------
        # Get velocity dispersion (units of km/s)
        vel_disp_kmpers, vel_disp_err_kmpers = find_veldisp(galaxy, directory, 'median')

        # Convert km/s to m/s
        vel_disp[i] = vel_disp_kmpers*1000
        vel_disp_err[i] = vel_disp_err_kmpers*1000
        #-----------------------------------------------------------------------


        #-----------------------------------------------------------------------
        # Get half-light radius (units of arcsec)
        r_half_arcsec = find_data_DRPall(DRPall_table, galaxy, 'nsa_elpetro_th50_r')
        z = find_data_DRPall(DRPall_table, galaxy, 'z')

        # Convert arcsec to m
        r_half_kpc = (1000*c*z/H0)*np.tan(r_half_arcsec*np.pi/(60*60*180))
        r_half[i] = 3.0857e19*r_half_kpc
        #-----------------------------------------------------------------------

    Mtot, Mtot_err = virial_mass(vel_disp, vel_disp_err, r_half, r_half_err)
    ############################################################################


    ############################################################################
    # Calculate dark matter halo mass
    #
    # Note: There is no given uncertainty for the stellar mass, so the 
    #       uncertainty in the dark matter halo mass is equal to the uncertainty 
    #       in the total mass.
    #---------------------------------------------------------------------------
    Mstar = np.zeros(len(IDs))

    for i,galaxy in enumerate(IDs):

        # Stellar mass
        Mstar[i] = find_data_DRPall(DRPall_table, galaxy, 'nsa_elpetro_mass')

    Mdark = Mtot - Mstar
    Mdark_err = Mtot_err
    ############################################################################


    ############################################################################
    # Calculate ratio of dark matter halo mass to stellar mass
    #---------------------------------------------------------------------------
    Mratio = Mdark/Mstar

    Mratio_err = Mdark_err/Mstar
    ############################################################################


    ############################################################################
    # Create table for output
    #---------------------------------------------------------------------------
    mass_table = Table()

    mass_table['Mtot'] = Mtot
    mass_table['Mtot_err'] = Mtot_err
    mass_table['Mdark'] = Mdark
    mass_table['Mdark_err'] = Mdark_err
    mass_table['Mratio'] = Mratio
    mass_table['Mratio_err'] = Mratio_err
    ############################################################################


    return mass_table
################################################################################














