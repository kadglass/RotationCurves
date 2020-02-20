################################################################################
# IMPORT MODULES
#-------------------------------------------------------------------------------
import matplotlib.pyplot as plt
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

	plt.xlabel('$M_*$ [$\log M_\odot$]')
	plt.ylabel(sigma_type + ' $\log \sigma_*$ [km/s]')

	if save:
		plt.savefig('Images/FJ_' + sigma_type + '.eps', format='eps', dpi=300)

	plt.show()
################################################################################