import numpy as np

def exponential_sphere(r, rho_c, a):
    '''
    exponential sphere mass distribution
    
    PARAMETERS
    ==========
    r : float
        radius [kpc]
        
    rho_c : float
        central density [M_sun/kpc^3]
        
    a : float
        scale radius [kpc]
        
    RETURN
    ======
    
    M : float
        mass within radius r [M_sun]
    
    '''
    
    x = r/a
    F = 1 - np.exp(-x)*(1 + x + x**2/2)
    M_0 = 8 * np.pi * a**3 * rho_c
    M = M_0 * F
    
    return M
    

def calc_tot_stellar_mass(gal_ID, rho_c, a, COV_DIR=''):
    '''
    
    calculate total mass of exponential sphere
    
    PARAMETERS
    ==========
    gal_ID : string
        galaxy plateifu
        
    cov_dir : string
        covariance directory
        
    rho_c : float
        best fit value for central density [M_sun/kpc^3]
        
    a : float
        best fit value for scale radius [kpc]
        
    RETURN
    ======
    M_0 : float
        total mass of exponential sphere [M_sun]
    
    M_0_err : float
        uncertainty on M_0
    
    '''
    
    # load covariance matrix
    cov = np.load(COV_DIR + gal_ID + '_cov.npy')
    
    # calculate total mass
    M_0 = 8 * np.pi * a**3 * rho_c
    
    # calculate uncertainty on total mass
    M_0_err = M_0 * np.sqrt(9 / a**2 * cov[1,1] + cov[0,0] / rho_c**2 + 6 * cov[1,0] / (a * rho_c))
    
    return M_0, M_0_err
    

def chi2_mass(params, r, m_star, m_star_err):
    '''
    
    calculate reduced chi2 of exponential sphere mass curve
    
    '''
    
    model = exponential_sphere(r, params[0], params[1])
    
    chi2 = np.sum((model - m_star)**2/m_star_err**2)
    n_chi2 = np.sqrt(chi2)
    
    return n_chi2