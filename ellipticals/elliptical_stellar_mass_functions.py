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

def exponential_disk(r, Sigd, Rd):
    '''
    mass distribution for exponential sphere and disk

    PARAMETERS
    ==========
    r : float
        radius [kpc]

    Sigd : float
        disk central surface density [M_sun/kpc^2]

    Rd : float
        disk scale radius [kpc]
    
    RETURNS
    =======
    M : float
        mass within radius r [M_sun]
    '''


    M = 2 * np.pi * Sigd * Rd *(Rd - np.exp(-r/Rd)*(r+Rd))
    return M

def exponential_sphere_disk(r, rho_c, a, Sigd, Rd):

    '''
    mass distribution for exponential sphere and disk

    PARAMETERS
    ==========
    r : float
        radius [kpc]
    
    rho_c : float
        sphere central density [M_sun/kpc^3]
        
    a : float
        sphere scale radius [kpc]

    Sigd : float
        disk central surface density [M_sun/kpc^2]

    Rd : float
        disk scale radius [kpc]
    
    RETURNS
    =======
    M : float
        mass within radius r [M_sun]
    '''


    sph = exponential_sphere(r, rho_c, a)
    disk = exponential_disk(r, Sigd, Rd)

    M = sph + disk
    return M


def hernquist_profile(r, R_scale, Mtot):

    '''
    mass distribution according to hernquist profile (Hernquist, L. 1990)

    PARAMETERS
    ==========
    r : float
        radius [kpc]
    
    R_scale : float
        scale radius [kpc]
        
    Mtot : float
        total stellar mass [M_sun]
    
    RETURNS
    =======
    M : float
        mass within radius r [M_sun]
    '''

    M = Mtot * r**2 / (r + R_scale)**2
    return M

    

def calc_tot_stellar_mass(params, stellar_profile):
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
    

    if stellar_profile == 'sphere':

        rho_c = params['rho_c']
        a = params['R_scale']
        rho_c_err = params['rho_c_err']
        a_err = params['R_scale_err']
    
        # calculate total mass
        M_0 = 8 * np.pi * a**3 * rho_c
        
        # calculate uncertainty on total mass
        M_0_err = M_0 * np.sqrt(9 / a**2 * a_err**2 + rho_c_err**2 / rho_c**2)

    elif stellar_profile == 'sphere_disk':

        rho_c = params['rho_c']
        a = params['R_scale']
        sigd = params['Sigma_d']
        rd = params['R_d']
        rho_c_err = params['rho_c_err']
        a_err = params['R_scale_err']
        sigd_err = params['Sigma_d_err']
        rd_err = params['R_d_err']

        Mb = 8 * np.pi * a**3 * rho_c
        Md = 2 * np.pi * sigd * rd**2

        M_0 = Mb + Md

        M_0_err = np.sqrt(Mb**2 * (9 / a**2 * a_err**2 + rho_c_err**2 / rho_c**2) +\
                          Md**2 * (sigd_err**2 / sigd**2 + 4 * rd_err**2 / rd**2))
    
    return M_0, M_0_err
    

def chi2_mass(params, r, m_star, m_star_err, stellar_profile):
    '''
    
    calculate reduced chi2 of exponential sphere mass curve
    
    '''
    
    if stellar_profile == 'sphere':
        model = exponential_sphere(r, params[0], params[1])
    
        chi2 = np.sum((model - m_star)**2/m_star_err**2)
        n_chi2 = chi2 / (len(r) - 2)

    if stellar_profile == 'sphere_disk':
        model = exponential_sphere_disk(r, params[0], params[1], params[2], params[3])
    
        chi2 = np.sum((model - m_star)**2/m_star_err**2)
        n_chi2 = chi2 / (len(r) - 4)

    if stellar_profile == 'hernquist':
        model = hernquist_profile(r, params[0], params[1])
    
        chi2 = np.sum((model - m_star)**2/m_star_err**2)
        n_chi2 = chi2 / (len(r) - 2)
    
    return n_chi2