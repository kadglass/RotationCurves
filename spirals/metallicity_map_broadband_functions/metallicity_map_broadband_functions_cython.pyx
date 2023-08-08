'''
Cythonized versions of some of the functions found in 
metallicity_map_broadband_functions.py
'''

cimport cython 
cimport numpy as np 

#from typedefs cimport DTYPE_F64_t

ctypedef np.float64_t DTYPE_F64_t

from libc.math cimport fabs


cpdef DTYPE_F64_t sersic_profile(list params, DTYPE_F64_t r):

    '''
    
    Sersic profile for surface brightness of galaxy

    PARAMETERS
    ==========

    

    RETURNS
    =======
    Sigma : float
        brightness at radius r


    '''

    cdef DTYPE_F64_t Sigma_e
    cdef DTYPE_F64_t Re
    cdef DTYPE_F64_t n
    cdef DTYPE_F64_t Sigma_0_in
    cdef DTYPE_F64_t h_in
    cdef DTYPE_F64_t Sigma_0_out
    cdef DTYPE_F64_t h_out
    cdef DTYPE_F64_t R_break

    cdef DTYPE_F64_t bn

    bn = 2*n - 1/3

    Sigma_e, Re, n, Sigma_0_in, h_in, Sigma_0_out, h_out, R_break = params

    if r <= R_break:
        Sigma = Sigma_e * np.exp(-bn*((r/Re)**(1/n) - 1)) + Sigma_0_in*np.exp(-r/h_in)

    else:
        Sigma = Sigma_e*np.exp(-bn*((r/Re)**(1/n)-1)) + Sigma_0_out*np.exp(-r/h_out)


    return Sigma