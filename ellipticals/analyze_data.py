################################################################################
# IMPORT MODULES
#-------------------------------------------------------------------------------
import numpy as np
import numpy.ma as ma

from astropy.io import fits

from IO_data import construct_filename, open_map
from plot_data import FJ_plot
################################################################################




################################################################################
#-------------------------------------------------------------------------------
def plot_FaberJackson(IDs, directory):
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
	'''


	############################################################################
	# Read in DRPall table
	#---------------------------------------------------------------------------
	drp_filename = '../manga_files/drpall-v2_4_3.fits'

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

		Mstar[i] = find_Mstar(DRPall_table, galaxy)

		vel_disp[i] = find_veldisp(galaxy, directory, 'median')
	############################################################################


	############################################################################
	# Plot Faber-Jackson relation
	#---------------------------------------------------------------------------
	FJ_plot(Mstar, vel_disp, 'median')
	############################################################################
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