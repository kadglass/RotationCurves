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


    sph = exponential_sphere(r, 10**rho_c, a)
    disk = exponential_disk(r, 10**Sigd, Rd)

    M = sph + disk
    return M

def exponential_sphere_disk_err(r, rho_c, rho_c_err, a, a_err, Sigd, Sigd_err, 
                                Rd, Rd_err):
    
    '''
    mass distribution for exponential sphere and disk

    PARAMETERS
    ==========
    r : float
        radius [kpc]
    
    rho_c, rho_c_err : float
        sphere central density log[M_sun/kpc^3]
        
    a, a_err : float
        sphere scale radius [kpc]

    Sigd, Sigd_err : float
        disk central surface density log[M_sun/kpc^2]

    Rd_err : float
        disk scale radius [kpc]
    
    RETURNS
    =======
    M, M_err : float
        mass within radius r [M_sun]
    '''

    rho_c = 10**rho_c
    rho_c_err = 10**rho_c_err
    Sigd_err = 10**Sigd_err
    Sigd = 10**Sigd

    Mb = exponential_sphere(r, rho_c, a)
    Md = exponential_disk(r, Sigd, Rd)

    Md_err = (Md * Sigd_err/Sigd)**2 + (2*Md/Rd - 2*np.pi*Sigd*np.exp(-r/Rd)\
                                         *(r**2/Rd + r*Rd))**2 * Rd_err**2
    
    
    x = r/a
    M0 = 8 * np.pi * a**3 * rho_c
    F = 1 - np.exp(-x) * (1+x+x**2 /2)
    
    Mb_err = (M0 * F * rho_c_err/ rho_c)**2 \
                     + (3/a * Mb - M0*np.exp(-x)*x**3/(2*a) * a_err)**2
    
    M_err = np.sqrt(Md_err + Mb_err)
    M = Mb + Md

    return M, M_err


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


def modified_hernquist_profile(r, R_scale, Mtot, gamma):

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

    M = Mtot * r**(gamma) / (r + R_scale)**(gamma)
    return M
    # M = Mtot + np.log10(r**gamma) - np.log10((r+R_scale)**gamma)
    # return M
    

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

        rho_c = params['sph_rho_c']
        a = params['sph_R_scale']
        rho_c_err = params['sph_rho_c_err']
        a_err = params['sph_R_scale_err']
    
        # calculate total mass
        M_0 = 8 * np.pi * a**3 * rho_c
        
        # calculate uncertainty on total mass
        M_0_err = M_0 * np.sqrt(9 / a**2 * a_err**2 + rho_c_err**2 / rho_c**2)

    elif stellar_profile == 'sphere_disk':

        rho_c = 10**params['sphd_rho_c']
        a = params['sphd_R_scale']
        sigd = 10**params['sphd_Sigma_d']
        rd = params['sphd_R_d']
        rho_c_err = 10**params['sphd_rho_c_err']
        a_err = params['sphd_R_scale_err']
        sigd_err = 10**params['sphd_Sigma_d_err']
        rd_err = params['sphd_R_d_err']

        Mb = 8 * np.pi * a**3 * rho_c
        Md = 2 * np.pi * sigd * rd**2

        M_0 = Mb + Md

        # M_0_err = np.sqrt(Mb**2 * (9 / a**2 * a_err**2 + rho_c_err**2 / rho_c**2) +\
        #                   Md**2 * (sigd_err**2 / sigd**2 + 4 * rd_err**2 / rd**2))
    
        # Mb_err2 = Mb**2 * (np.log(10)**2 * rho_c_err**2 + (3/a)**2 * a_err**2)
        # Md_err2 = Md**2 * (np.log(10)**2 * sigd_err**2 + (2/rd)**2 * rd_err**2)

        Mb_err2 = Mb**2 * (9*a_err**2/a**2 + rho_c_err**2/rho_c**2)
        Md_err2 = Md**2 * (sigd_err**2/sigd**2 + 4*rd_err**2/rd**2)

        M_0_err = np.sqrt(Mb_err2 + Md_err2)



    return M_0, M_0_err
    

def chi2_mass(params, r, m_star, m_star_err, stellar_profile):
    '''
    
    calculate chi2 of exponential sphere mass curve
    
    '''
    
    if stellar_profile == 'sphere':

        model = exponential_sphere(r, params[0], params[1])
    
        chi2 = np.sum((model - m_star)**2/m_star_err**2)
        # n_chi2 = chi2 / (len(r) - 2)

    elif stellar_profile == 'sphere_disk':
        model = exponential_sphere_disk(r, params[0], params[1], params[2], params[3])
    
        chi2 = np.sum((model - m_star)**2/m_star_err**2)
        # n_chi2 = chi2 / (len(r) - 4)

    elif stellar_profile == 'hernquist':
        model = hernquist_profile(r, params[0], params[1])
    
        chi2 = np.sum((model - m_star)**2/m_star_err**2)
        # n_chi2 = chi2 / (len(r) - 2)

    elif stellar_profile == 'mod_hernquist':
        model = modified_hernquist_profile(r, params[0], params[1], params[2])
    
        chi2 = np.sum((model - m_star)**2/m_star_err**2)
        # n_chi2 = chi2 / (len(r) - 3)
    
    return chi2

def sum_Pipe3D_mass(sMass_density):

    '''
    calculate the total stellar mass and uncertainty from Pipe3D map

    PARAMETERS
    ==========
    sMass_density : array
        stellar mass density map [log Msun/spax^2]

    RETURNS
    =======
    M_star : float
        total stellar mass in map [log M_sun]

    M_star_err : float
        RMS / sqrt(N) uncertainty on M_star [log M_sun]

    
    '''

    msMass = np.ma.array(sMass_density, mask=np.isnan(sMass_density))
    msMass_flat = np.ma.MaskedArray.compressed(msMass)
    N = len(msMass_flat)

    sMass = 10**msMass_flat

    M_star = np.log10(np.sum(sMass))

    M_star_err = np.log10(np.sqrt(np.mean(sMass**2)) / np.sqrt(N))

    return M_star, M_star_err