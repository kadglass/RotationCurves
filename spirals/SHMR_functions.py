import numpy as np

def shmr_RP11(logMstar, color='all'):
    '''
    relationship between stellar mass and halo mass defined for red and 
    blue galaxies

    PARAMETERS
    ==========
    logMstar : float
        stellar mass in log(M_sun/h) units
    color : string
        color of galaxies to calculate relations
        can be 'all', 'blue', or 'red'
    '''

    logh = np.log10(0.7)


    if color == 'all':
        logM_0h  = 11.97 - logh
        log_Ms = 10.40 - logh
        beta = 0.24
        alpha = 0.34
        gamma = 0.90
        a = np.array([0, 0.095])
    
    elif color == 'blue':
        logM_0h  = 11.99 - logh
        log_Ms = 10.30 - logh
        beta = 0.37
        alpha = 0.90
        gamma = 0.90
        a = np.array([0.125, 0.125])


    elif color == 'red':
        logM_0h  = 11.87 - logh
        log_Ms = 10.40 - logh
        beta = 0.18
        alpha = 1.50
        gamma = 0.90
        a = np.array([0, 0.093])

    
    less_ms = np.multiply(logMstar < log_Ms,1)
    Mstar_Ms = logMstar - log_Ms

    first_term = logM_0h - gamma*np.log10(2) \
        + gamma*np.log10((10**Mstar_Ms)**(beta/gamma) \
                         + (10**Mstar_Ms)**(alpha/gamma))
    sec_term =a[less_ms]*(10**(Mstar_Ms)-1)

    return first_term + sec_term
    