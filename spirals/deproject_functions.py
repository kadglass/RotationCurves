import numpy as np


def deproject_spaxel(coords, center, phi, i_angle):
    '''
    Calculate the deprojected radius for the given coordinates in the map.


    PARAMETERS
    ==========

    coords : length-2 tuple
        (i,j) coordinates of the current spaxel

    center : length-2 tuple
        (i,j) coordinates of the galaxy's center

    phi : float
        Rotation angle (in radians) east of north of the semi-major axis.

    i_angle : float
        Inclination angle (in radians) of the galaxy.


    RETURNS
    =======

    r : float
        De-projected radius from the center of the galaxy for the given spaxel
        coordinates.
    '''


    # Distance components between center and current location
    delta = np.subtract(coords, center)

    # x-direction distance relative to the semi-major axis
    dx_prime = (delta[1]*np.cos(phi) + delta[0]*np.sin(phi))/np.cos(i_angle)

    # y-direction distance relative to the semi-major axis
    dy_prime = (-delta[1]*np.sin(phi) + delta[0]*np.cos(phi))

    # De-projected radius for the current point
    r = np.sqrt(dx_prime**2 + dy_prime**2)

    # Angle (counterclockwise) between North and current position
    theta = np.arctan2(-dx_prime, dy_prime)

    return r, theta
