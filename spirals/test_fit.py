import numpy as np

from dark_matter_mass_v1 import rot_fit_BB

from scipy.optimize import minimize

import matplotlib.pyplot as plt




################################################################################
################################################################################
################################################################################

def rot_residual_BB( params, r_data, v_data):
    """
    Function to fit the rotation curve data to.


    PARAMETERS
    ==========
    
    depro_radius : float
        Deprojected radius as taken from the [PLATE]-[FIBERID] rotation curve 
        data file (in units of kpc); the "x" data of the rotation curve equation

    v_max : float
        The maximum velocity (or in the case of fitting the negative, the
        absolute value of the minimum velocity) parameter of the rotation curve 
        equation (given in km/s)

    r_turn : float
        The radius at which the rotation curve trasitions from increasing to 
        flat-body for the rotation curve equation (given in kpc)

    alpha : float
        The exponential parameter for the rotation curve equation


    RETURNS
    =======
        
    The rotation curve equation with the given '@param' parameters and
    'depro_radius' data
    """

    v_max, r_turn, alpha = params

    delta_v = v_data - rot_fit_BB(r_data, v_max, r_turn, alpha)

    return delta_v


################################################################################
################################################################################
################################################################################

def rot_loss_BB(params, r_data, v_data, v_scale, r_scale, alpha_scale):

    v_max, r_turn, alpha = params

    residuals = v_data - rot_fit_BB(r_data, v_max, r_turn, alpha)

    res_squared = residuals*residuals

    penalty = v_max/v_scale + r_turn/r_scale + alpha/alpha_scale

    total_loss = np.sum(res_squared)/len(r_data) + penalty

    return total_loss



################################################################################
################################################################################
################################################################################

def rot_fit_BB_noise( depro_radius, v_max, r_turn, alpha):

    return v_max * (depro_radius / (r_turn**alpha + depro_radius**alpha)**(1/alpha)) + 10*(np.random.rand(len(depro_radius)) - 0.5)


################################################################################
################################################################################
################################################################################



v_max = 100
r_turn = 10
alpha = 1.1

r_vals = np.linspace(0.1, 100, 300)

v_vals = rot_fit_BB_noise(r_vals, v_max, r_turn, alpha)

#plt.plot(r_vals, v_vals, '.')

#plt.show()





################################################################################
# Parameter initial guess
#-------------------------------------------------------------------------------
v_guess_i = np.argmax(v_vals)
v_guess = v_vals[v_guess_i]
r_guess = r_vals[v_guess_i]
alpha_guess = 0.1

x0 = [v_guess, r_guess, alpha_guess]
################################################################################


################################################################################
# Parameter bounds
#-------------------------------------------------------------------------------
v_max_low = 0.1
v_max_high = 41000.
v_max_bounds = (v_max_low, v_max_high)

r_turn_low = 0.001
r_turn_high = 100.
r_turn_bounds = (r_turn_low, r_turn_high)

alpha_low = np.nextafter(0, 1)
alpha_high = 10.
alpha_bounds = (alpha_low, alpha_high)

#low_bound = [v_max_low, r_turn_low, alpha_low]
#high_bound = [v_max_high, r_turn_high, alpha_high]
################################################################################


################################################################################
# Penalties
#-------------------------------------------------------------------------------
vmax_scale = 500.
rturn_scale = 5.
alpha_scale = 2.5
################################################################################



result = minimize(rot_loss_BB, 
                  x0, 
                  args=(r_vals, v_vals, vmax_scale, rturn_scale, alpha_scale), 
                  bounds=[v_max_bounds, r_turn_bounds, alpha_bounds]
                 )

H_inv = result.hess_inv

print(vars(H_inv))
print(dir(H_inv))
print(H_inv.H)


uncertainty = np.sqrt(np.diag(result.hess_inv.todense()))

print(uncertainty)





