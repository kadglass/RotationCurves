################################################################################
# IMPORT MODULES
#-------------------------------------------------------------------------------
from parse_data import build_galaxy_IDs

from analyze_data import plot_FaberJackson, determine_masses

from IO_data import write_masses
################################################################################



################################################################################
#-------------------------------------------------------------------------------
def FaberJackson(galaxy_ID, 
                 data_directory, 
                 master_filename, 
                 sigma_type, 
                 save_fig=False):
    '''
    Plot the Faber-Jackson relation for the galaxies in galaxy_ID.


    PARAMETERS
    ==========

    galaxy_ID : string
        Either the plate-IFU identification for a particular MaNGA galaxy to be 
        analyzed, or 'all'.  If 'all', then all elliptical galaxies should be 
        analyzed.

    data_directory : string
        Location of the data stored on the local computer.

    master_filename : string
        File name of the master table.  This table is a list of all the MaNGA
        galaxies, along with their associated NSA data and other parameters 
        previously calculated by us.

    sigma_type : string
        Location / type of velocity dispersion.  Options include:
        - 'median'  : returns the median value of the velocity dispersion map
        - 'central' : returns the central value of the velocity dispersion map

    save_fig : boolean
        Determines wether or not to save the figure.  Default is False (do not 
        save).
    '''


    ############################################################################
    # Create list of tuples of elliptical galaxy ID(s) to be analyzed
    #---------------------------------------------------------------------------
    elliptical_IDs = build_galaxy_IDs(galaxy_ID, master_filename)
    ############################################################################


    ############################################################################
    # Plot the Faber-Jackson relation
    #---------------------------------------------------------------------------
    plot_FaberJackson(elliptical_IDs, data_directory, sigma_type, save_fig)
    ############################################################################
################################################################################



################################################################################
#-------------------------------------------------------------------------------
def elliptical_masses(galaxy_ID, data_directory, master_filename):
    '''
    Parse through the data files and calculate the mass of elliptical galaxies.


    PARAMETERS
    ==========

    galaxy_ID : string
        Either the plate-IFU identification for a particular MaNGA galaxy to be 
        analyzed, or 'all'.  If 'all', then all elliptical galaxies should be 
        analyzed.

    data_directory : string
        Location of the data stored on the local computer.

    master_filename : string
        File name of the master table.  This table is a list of all the MaNGA
        galaxies, along with their associated NSA data and other parameters 
        previously calculated by us.
    '''


    ############################################################################
    # Create list of tuples of elliptical galaxy ID(s) to be analyzed
    #---------------------------------------------------------------------------
    elliptical_IDs = build_galaxy_IDs(galaxy_ID, master_filename)
    ############################################################################


    ############################################################################
    # Calculate mass for each galaxy
    #---------------------------------------------------------------------------
    elliptical_masses = determine_masses(elliptical_IDs, data_directory)
    ############################################################################


    ############################################################################
    # Add masses to value-added catalog (master file)
    #---------------------------------------------------------------------------
    write_masses(elliptical_masses, elliptical_IDs, master_filename)
    ############################################################################
################################################################################