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

h = 1
H0 = 100*h 		# km/s/Mpc
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

    r_band_image = open_map(cube_filename, 'SPX_MFLUX')
    ############################################################################


    if disp is 'median':
        ########################################################################
        # Find the median velocity dispersion of the stellar velocity dispersion 
        # map.
        #-----------------------------------------------------------------------
        vel_disp = ma.median(vel_disp_map)
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
        ########################################################################

    '''
    ############################################################################
    # Convert from km/s to m/s
    #---------------------------------------------------------------------------
    vel_disp_mpers = vel_disp*1000
    ############################################################################
    '''

    return vel_disp
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

	for i,galaxy in enumerate(IDs):

		Mstar[i] = find_data_DRPall(DRPall_table, galaxy, 'nsa_elpetro_mass')

		vel_disp[i] = find_veldisp(galaxy, directory, sigma_type)
	############################################################################


	############################################################################
	# Plot Faber-Jackson relation
	#---------------------------------------------------------------------------
	FJ_plot(Mstar, vel_disp, sigma_type, save_fig)
	############################################################################
################################################################################






################################################################################
#-------------------------------------------------------------------------------
def virial_mass(star_sigma, r_half):
	'''
	Calculate the virial mass of a galaxy.


	PARAMETERS
	==========

	star_sigma : float or numpy array
		stellar velocity dispersion in m/s

	r_half : float or numpy array
		Half-light radius in m


	RETURNS
	=======

	Mvir_Msun : float or numpy array
		Virial mass in solar masses
	'''


	# Virial mass (in kg)
	Mvir = 7.5*star_sigma*star_sigma*r_half/G

	# Convert kg to Msun
	Mvir_Msun = Mvir/Msun

	return Mvir_Msun
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
	#---------------------------------------------------------------------------
	vel_disp = np.zeros(len(IDs))
	r_half_m = np.zeros(len(IDs))

	for i,galaxy in enumerate(IDs):

		#-----------------------------------------------------------------------
		# Get velocity dispersion
		vel_disp[i] = find_veldisp(galaxy, directory, 'median')
		#-----------------------------------------------------------------------


		#-----------------------------------------------------------------------
		# Get half-light radius (units of arcsec)
		r_half = find_data_DRPall(DRPall_table, galaxy, 'nsa_elpetro_th50_r')
		z = find_data_DRPall(DRPall_table, galaxy, 'z')

		# Convert arcsec to m
		r_half_kpc = (1000*c*z/H0)*np.tan(r_half*np.pi/(60*60*180))
		r_half_m[i] = 3.0857e19*r_half_kpc
		#-----------------------------------------------------------------------

	Mtot = virial_mass(vel_disp, r_half_m)
	############################################################################


	############################################################################
	# Calculate dark matter halo mass
	#---------------------------------------------------------------------------
	Mstar = np.zeros(len(IDs))

	for i,galaxy in enumerate(IDs):

		# Stellar mass
		Mstar[i] = find_data_DRPall(DRPall_table, galaxy, 'nsa_elpetro_mass')

	Mdark = Mtot - Mstar
	############################################################################


	############################################################################
	# Calculate ratio of dark matter halo mass to stellar mass
	#---------------------------------------------------------------------------
	Mratio = Mdark/Mstar
	############################################################################


	############################################################################
	# Create table for output
	#---------------------------------------------------------------------------
	mass_table = Table()

	mass_table['Mtot'] = Mtot
	mass_table['Mtot_err'] = []
	mass_table['Mdark'] = Mdark
	mass_table['Mdark_err'] = []
	mass_table['Mratio'] = Mratio
	mass_table['Mratio_err'] = []
	############################################################################


	return mass_table
################################################################################














