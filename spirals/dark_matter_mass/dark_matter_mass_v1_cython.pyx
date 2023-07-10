'''
Cythonized versions of some of the functions found in 
dark_matter_mass_v1.py
'''

cimport cython 
cimport numpy as np 

#from typedefs cimport DTYPE_F64_t

ctypedef np.float64_t DTYPE_F64_t

from libc.math cimport fabs






################################################################################
# BB velocity model
#-------------------------------------------------------------------------------

cpdef DTYPE_F64_t rot_fit_BB(DTYPE_F64_t depro_radius, list params):
    '''
    BB function to fit rotation curve data to

    PARAMETERS
    ==========

    depro_radius : float
        Deprojected radius as taken from the [PLATE]-[FIBERID] rotation curve 
        data file (in units of kpc); the "x" data of the rotation curve equation
    params : list
        model parameter values

    RETURNS
    =======
    v : float
        The rotation curve equation with the given parameters at the given depro_radius


    '''

    cdef DTYPE_F64_t v_max
    cdef DTYPE_F64_t r_turn
    cdef DTYPE_F64_t alpha
    cdef DTYPE_F64_t v

    v_max, r_turn, alpha = params

    v = v_max * fabs(depro_radius) / (r_turn**alpha + fabs(depro_radius)**alpha)**(1/alpha)

    if depro_radius < 0:
        v = v * -1


    return v

################################################################################




################################################################################
# tail velocity model
#-------------------------------------------------------------------------------

cpdef DTYPE_F64_t rot_fit_tail(DTYPE_F64_t depro_radius, list params):
    '''
    tail velocity function to fit rotation curve data to

    PARAMETERS
    ==========

    depro_radius : float
        Deprojected radius as taken from the [PLATE]-[FIBERID] rotation curve 
        data file (in units of kpc); the "x" data of the rotation curve equation
    params : list
        model parameter values

    RETURNS
    =======
    v : float
        The rotation curve equation with the given parameters at the given depro_radius


    '''


    cdef DTYPE_F64_t v_max
    cdef DTYPE_F64_t r_turn
    cdef DTYPE_F64_t alpha
    cdef DTYPE_F64_t b
    cdef DTYPE_F64_t v

    v_max, r_turn, alpha, b = params

    v = v_max*(fabs(depro_radius)*(1 + b * fabs(depro_radius))/(r_turn**alpha+fabs(depro_radius)**alpha)**(1/alpha))

    #v = v*np.sign(depro_radius)

    if depro_radius < 0:
        v = v * -1

    return v

################################################################################
