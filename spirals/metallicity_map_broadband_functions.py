import numpy as np

def surface_brightness_profile(params, r):

    '''
    
    Sersic bulge + broken exponential profile for surface brightness of galaxy

    PARAMETERS
    ==========

    

    RETURNS
    =======
    Sigma : float
        brightness at radius r

    '''
    


    Sigma_e, Re, n, Sigma_0_in, h_in, Sigma_0_out, h_out, R_break = params

    bn = 1.9992*n -0.3271

    if r <= R_break:
        Sigma = Sigma_e * np.exp(-bn*((r/Re)**(1/n) - 1)) + Sigma_0_in*np.exp(-r/h_in)

    else:
        Sigma = Sigma_e*np.exp(-bn*((r/Re)**(1/n)-1)) + Sigma_0_out*np.exp(-r/h_out)


    return Sigma