################################################################################
# IMPORT MODULES
#-------------------------------------------------------------------------------
import numpy as np

import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D
################################################################################





################################################################################
#-------------------------------------------------------------------------------
def FJ_plot(M, sigma, sigma_type, save):
	'''
	Plot the Faber-Jackson relation.


	PARAMETERS
	==========

	M : numpy array of shape (N,)
		Stellar masses of the galaxies, in units of log(M_sun)

	sigma : numpy array of shape (N,)
		Velocity dispersions of the galaxies, in units of km/s

	sigma_type : string
		Location / type of velocity dispersion.  Options include:
		- 'median'  : returns the median value of the velocity dispersion map
		- 'central' : returns the central value of the velocity dispersion map

	save : boolean
		Flag to determine whether or not to save the figure.  Value of True 
		saves the figure, while a value of False only displays the figure.
	'''


	plt.figure()

	plt.loglog(M, sigma, '.')

	#plt.xlabel('$\log (M_* / M_\odot$)')
	plt.xlabel('$M_r$')
	plt.ylabel(sigma_type + ' $\log \sigma_*$ [km/s]')

	if save:
		plt.savefig('Images/FJ_' + sigma_type + '.eps', format='eps', dpi=300, transparent=True)

	plt.show()
################################################################################




################################################################################
#-------------------------------------------------------------------------------
def FP_plot(M, R, sigma, sigma_type, save):
	'''
	Plot the Fundamental plane.


	PARAMETERS
	==========

	M : numpy array of shape (N,)
		Stellar masses of the galaxies, in units of log(M_sun)

	R : numpy array of shape (N,)
		Radius of the galaxies (assumed to be 50% Petrosian radius)

	sigma : numpy array of shape (N,)
		Velocity dispersions of the galaxies, in units of km/s

	sigma_type : string
		Location / type of velocity dispersion.  Options include:
		- 'median'  : returns the median value of the velocity dispersion map
		- 'central' : returns the central value of the velocity dispersion map

	save : boolean
		Flag to determine whether or not to save the figure.  Value of True 
		saves the figure, while a value of False only displays the figure.
	'''


	fig = plt.figure()
	ax = fig.gca(projection='3d')

	ax.plot(np.log10(M), np.log10(sigma), R, '.')

	ax.set_xlabel('$\log (M_* / M_\odot$)')
	ax.set_ylabel(sigma_type + ' $\log \sigma_*$ [km/s]')
	ax.set_zlabel('Petrosian 50% radius [arcsec]')

	ax.set_zlim(zmin=0)

	if save:
		plt.savefig('Images/FP_' + sigma_type + '.eps', format='eps', dpi=300, transparent=True)

	plt.show()
################################################################################