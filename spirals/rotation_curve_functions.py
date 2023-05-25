################################################################################
# All the libraries used & constant values
#-------------------------------------------------------------------------------
from scipy import integrate as inte
import numpy as np
import matplotlib.pyplot as plt
from numpy import log as ln

G = 6.67E-11  # m^3 kg^-1 s^-2
Msun = 1.989E30  # kg

from astropy.table import QTable
from scipy.optimize import minimize
import astropy.units as u
from scipy.special import kn
from scipy.special import iv

#from galaxy_component_functions import disk_vel, halo_vel_iso

'''
import warnings
warnings.filterwarnings('error')
'''
################################################################################





################################################################################
# bulge (Di Paolo et al. 2019)
#-------------------------------------------------------------------------------
def vel_b(r, A, Vin, Rd):
    '''
    Calculate the velocity due to the bulge component.
    
    Formula is from DiPaolo19 (eqn. 8)
    
    
    PARAMETERS
    ==========
    
    r : float
        The projected radius (pc)
    
    A: float
        Scale factor [unitless]
    Vin : float
        The scale velocity in the bulge (km/s)
    
    Rd : float
        The scale radius of the disk (pc)
        
    
    RETURNS
    =======
    
    v : float
        The rotational velocity of the bulge (km/s)
    '''

    v2 = A * (Vin ** 2) / (r / (0.2 * Rd))
    
    '''
    try:
        v = np.sqrt(v2)
        
    except Warning:
        print('RuntimeWarning in vel_b')
        print('v2 =', v2)
        print('A =', A)
        print('r =', r)
        print('Vin =', Vin)
        print('Rd =', Rd)
    '''

    #v = np.sqrt(np.abs(v2))
    v = np.sqrt(v2)

    return v



def vel_b2(r, A, Vin, Rd):
    '''
    Calculate the square of the velocity due to the bulge component.
    
    Formula is from DiPaolo19 (eqn. 8)
    
    
    PARAMETERS
    ==========
    
    r : float
        The projected radius (pc)
    
    A: float
        Scale factor [unitless]
    Vin : float
        The scale velocity in the bulge (km/s)
    
    Rd : float
        The scale radius of the disk (pc)
        
    
    RETURNS
    =======
    
    v2 : float
        The square of the rotational velocity of the bulge (km/s)
    '''

    v2 = A * (Vin ** 2) / (r / (0.2 * Rd))
    
    return v2


################################################################################





################################################################################
# de Vaucouleur's bulge model
#-------------------------------------------------------------------------------
gamma = 3.3308 # unitless
kappa = gamma*ln(10) # unitless


#-------------------------------------------------------------------------------
# surface density - sigma
#-------------------------------------------------------------------------------
def sigma_b(x,a,b):
    """
    parameters:
    x (projected radius): The projected radius  (pc)
    a (central density): The central density of the bulge (M_sol/pc^2)
    b (central radius): The central radius of the bulge (kpc)

    return: surface density of the bulge (g/pc^2)
    """
    return a*np.exp(-1*kappa*((x/b)**0.25-1)) #M_sol/pc^2
#-------------------------------------------------------------------------------


#-------------------------------------------------------------------------------
# derivative of sigma with respect to r
#-------------------------------------------------------------------------------
def dsdx(x,a,b):
    """
    parameters:
    x (projected radius): The projected rdius  (pc)
    a (central density): The central density of the bulge (M/pc^2)
    b (central radius): The central radius of the bulge (kpc)

    return: derivative of sigma (g/pc^3)
    """
    return sigma_b(x,a,b)*(-0.25*kappa)*(b**-0.25)*(x**-0.75) # M_sol/pc^2
#-------------------------------------------------------------------------------


#-------------------------------------------------------------------------------
# integrand for getting volume density
#-------------------------------------------------------------------------------
def density_integrand(x,r,a,b):
    """
    parameters:
    x (projected radius): The projected radius  (pc)
    r (radius): The a distance from the centre (pc)
    a (central density): The central density of the bulge (M/pc^2)
    b (central radius): The central radius of the bulge (kpc)

    return: integrand for volume density of the bulge (g/pc^3)
    """
    return -(1/np.pi)*dsdx(x,a,b)/np.sqrt(x**2-r**2)
#-------------------------------------------------------------------------------


#-------------------------------------------------------------------------------
# mass integrand
#-------------------------------------------------------------------------------
def mass_integrand(r,a,b):
    """
    parameters:
    x (projected radius): The projected rdius  (pc)   
    r (radius): The a distance from the centre (pc)
    a (central density): The central density of the bulge (M/pc^2)
    b (central radius): The central radius of the bulge (kpc)

    return: volume density of the bulge
    """
    # integrating for volume density
    vol_den, vol_den_err = inte.quad(density_integrand, r, np.inf, args=(r,a,b))
    return 4*np.pi*vol_den*r**2
#-------------------------------------------------------------------------------


#-------------------------------------------------------------------------------
# getting a velocity
#-------------------------------------------------------------------------------
def bulge_vel(r,a,b):
    """
    parameters:
    r (radius): The a distance from the centre (pc)
    a (central density): The central density of the bulge (M/pc^2)
    b (central radius): The central radius of the bulge (kpc)

    return: rotational velocity of the bulge (pc/s)
    """
    # integrating to get mass
    if isinstance(r, float):
        bulge_mass, m_err = inte.quad(mass_integrand, 0, r, args=(a, b))
    else:
        bulge_mass = np.zeros(len(r))
        err = np.zeros(len(r))

        for i in range(len(r)):
            bulge_mass[i],err[i] = inte.quad(mass_integrand, 0, r[i], args=(a,b))

    # v = sqrt(GM/r) for circular velocity
    vel = np.sqrt(G*(bulge_mass*1.988E30)/(r*3.08E16))
    vel /= 1000

    return vel
#-------------------------------------------------------------------------------
################################################################################




################################################################################
# Disk Mass
#-------------------------------------------------------------------------------
def surface_density(r, SigD, Rd):
    return SigD*np.exp(-r/Rd)


def mass_integrand(r, SigD, Rd):
    return 2*np.pi*r*surface_density(r, SigD, Rd)


def disk_mass(function_parameters, r):
    '''
    Calculate the total mass within some radius r from the disk mass function.


    PARAMETERS
    ==========

    function_parameters : dictionary
        Parameter values (and their uncertainties) for the disk velocity 
        function (Sigma_d, R_d).  Sigma_D has units of Msun/pc^2, and R_d has 
        units of kpc.

    r : float
        Radius within which to calculate the total mass.  Units are kpc

    
    RETURNS
    =======

    Mdisk : float
        Mass of the disk within the radius r.  Units are log(solar masses).

    Mdisk_err : float
        Uncertainty in the disk mass.  Units are log(solar masses).
    '''

    SigD = function_parameters['Sigma_disk']*1e6 # Converting from Msun/pc^2 to Msun/kpc^2
    Rd = function_parameters['R_disk']
    
    #Mdisk, Mdisk_err = inte.quad(mass_integrand, 0, r, args=[SigD, Rd])
    Mdisk = 2*np.pi*SigD*Rd*(Rd - np.exp(-r/Rd)*(r + Rd))

    Mdisk_err = np.sqrt((2*np.pi*SigD*Rd*(2*(1 - np.exp(-r/Rd)) - (r/Rd)**2*np.exp(-r/Rd) - 2*(r/Rd)*np.exp(-r/Rd)))**2*function_parameters['Sigma_disk_err']**2 \
                        + (Mdisk/SigD)**2*function_parameters['R_disk_err']**2)
    
    return np.log10(Mdisk), np.log10(Mdisk_err)
################################################################################




################################################################################
# Disk velocity from Paolo et al. 2019 or Sofue 2013
#-------------------------------------------------------------------------------
# Fitting for disk mass
def v_d(r, Mdisk, Rd):
    '''
    :param r: The a distance from the centre [pc]
    :param Mdisk: The total mass of the disk [M_sun]
    :param Rd: The scale radius of the disk [pc]
    :return: The rotational velocity of the disk [km/s]
    '''
    # Unit conversion
    Mdisk_kg = Mdisk * Msun

    bessel_component = (iv(0, r / (2 * Rd)) * kn(0, r / (2 * Rd)) - iv(1, r / (2 * Rd)) * kn(1, r / (2 * Rd)))
    vel2 = ((0.5) * G * Mdisk_kg * (r / Rd) ** 2 / (Rd * 3.086E16)) * bessel_component

    return np.sqrt(vel2) / 1000


# Fitting for central surface density
#def disk_vel(params, r):
def disk_vel(r, SigD, Rd):
    '''
    :param SigD: Central surface density for the disk [M_sol/pc^2]
    :param Rd: The scale radius of the disk [kpc]
    :r: The distance from the centre [kpc]
    :return: The rotational velocity of the disk [km/s]
    '''
    #SigD, Rd = params

    
    y = r / (2 * Rd)

    bessel_component = (iv(0, y) * kn(0, y) - iv(1, y) * kn(1, y))
    vel2 = (4 *np.pi * G * SigD * y ** 2 * ((Rd*1000)/3.086E16)*Msun) * bessel_component

    return np.sqrt(vel2) / 1000
################################################################################

def disk_bulge_vel(r, SigD, Rd, rho_bulge, R_bulge):
    '''
    Calculate the total mass within some radius r from the disk mass function.


    PARAMETERS
    ==========

    rho_bulge : float
        bulge central mass density [M_sol/kpc^3]

    R_bulge : float
        bulge scale radius [kpc]
    r
        [kpc]
    
    RETURNS
    =======

    v_disk : velocity of disk and bulge component [km/s]
    '''

    #v_d = disk_vel(r=r, SigD=SigD, Rd=Rd) #km/s

    coeff = 4 * np.pi * G *SigD *  ((Rd*1000)/3.086E16)*Msun
    y = r / (2*Rd)
    bessel_component = (iv(0, y) * kn(0, y) - iv(1, y) * kn(1, y))
    vd_2 = coeff * y**2 * bessel_component / 10**6

    # bulge component
    x = r / R_bulge # unitless
    F = 1 - np.exp(-x) * (1 + x + x**2 / 2) # unitless
    M0 = 8 * np.pi * R_bulge**3 * rho_bulge # sol mass
    coeff_2 = G * M0 * Msun * 1/(1000**3 * 3.086E16)
    #vb_2 = G * M0 * Msun / r * F * 1/(1000**3 * 3.086E16)
    vb_2 = coeff_2 * F / r


    v_disk = np.sqrt(vd_2 + vb_2) #km/s
    return v_disk






################################################################################
# halo (isothermal)
#-------------------------------------------------------------------------------
def rho0_iso(Vinf, Rh_kpc):
    '''
    parameters:
    Vinf (rotational velocity): The rotational velocity when r approaches infinity (km/s)
    Rh (scale radius): The scale radius of the dark matter halo [kpc]

    return: volume density of the isothermal halo (g/pc^3)
    '''
    return 0.740 * (Vinf / 200)**2 * (Rh_kpc) ** (-2)


def rho_iso(r, Vinf, Rh):
    '''
    parameters:
    r (radius): The a distance from the centre (pc)
    Vinf (rotational velocity): The rotational velocity when r approaches infinity (km/s)
    Rh (scale radius): The scale radius of the dark matter halo (pc)

    return: volume density of the isothermal halo (g/pc^3)
    '''

    rho_0 = rho0_iso(Vinf, Rh / 1000)

    return rho_0 / (1 + (r / Rh) ** 2)


def integrand_h_iso(r, Vinf, Rh):
    '''
    parameters:
    r (radius): The a distance from the centre (pc)
    Vinf (rotational velocity): The rotational velocity when r approaches infinity (km/s)
    Rh (scale radius): The scale radius of the dark matter halo (pc)
    return: integrand for getting the mass of the isothermal halo
    '''

    return 4 * np.pi * (rho_iso(r, Vinf, Rh)) * r ** 2


def mass_h_iso(r, Vinf, Rh):
    '''
    parameters:
    r (radius): The a distance from the centre (pc)
    e (rotational velocity): The rotational velocity when r approaches infinity (km/s)
    f (scale radius): The scale radius of the dark matter halo (pc)

    return: mass of the isothermal halo (g)
    '''
    halo_mass, m_err = inte.quad(integrand_h_iso, 0, r, args=(Vinf, Rh))
    return halo_mass


def vel_h_iso(r, Vinf, Rh):
    '''
    parameters:
    r (radius): The a distance from the centre (pc)
    Vinf (rotational velocity): The rotational velocity when r approaches infinity (km/s)
    Rh (scale radius): The scale radius of the dark matter halo (pc)

    return: rotational velocity of the isothermal halo (pc/s)
    '''
    if isinstance(r, float):
        halo_mass = mass_h_iso(r, Vinf, Rh)
    else:
        halo_mass = np.zeros(len(r))
        for i in range(len(r)):
            halo_mass[i] = mass_h_iso(r[i], Vinf, Rh)

    vel2 = G * (halo_mass * Msun) / (r * 3.08E16)

    try:
        vel = np.sqrt(vel2)/1000
    except Warning:
        print('RuntimeWarning in vel_h_iso')
        print('vel2 = ',vel2)
        print('halo_mass = ',halo_mass)
        print('r = ',r)

    return np.sqrt(vel2)/1000
################################################################################




################################################################################
# Circular velocity for isothermal Halo without the complicated integrals
# from eqn (51) & (52) from Sofue 2013.
#-------------------------------------------------------------------------------
def halo_vel_iso(r, rho0_h, Rh):
    '''
    :param r: The a distance from the centre (pc)
    :param rho_iso: The central density of the halo (M_sol/pc^3)
    :param Rh: The scale radius of the dark matter halo (pc)
    :return: rotational velocity
    '''

    v_inf = np.sqrt(4*np.pi*G*rho0_h*Rh**2)
    # the part in the square root would be unitless
    vel = v_inf * np.sqrt(1 - ((Rh/r)*np.arctan2(Rh,r)))

    return vel
################################################################################




################################################################################
# halo (NFW)
#-------------------------------------------------------------------------------
def rho_NFW(r, rho0_h, Rh):
    '''
    parameters:
    r (radius): The a distance from the centre (pc)
    rho0_h (central density): The central density of the halo (M_sol/pc^3)
    Rh (scale radius): The scale radius of the dark matter halo (pc)

    return: volume density of the isothermal halo (M/pc^3)
    '''
    return rho0_h / ((r / Rh) * ((1 + (r / Rh)) ** 2))


def integrand_h_NFW(r, rho0_h, Rh):
    '''
    parameters:
    r (radius): The a distance from the centre (pc)
    rho0_h (central density): The central density of the halo (M_sol/pc^3)
    Rh (scale radius): The scale radius of the dark matter halo (pc)

    return: integrand for getting the mass of the isothermal halo
    '''

    return 4 * np.pi * (rho_NFW(r, rho0_h, Rh)) * r ** 2


def mass_h_NFW(r, rho0_h, Rh):
    '''
    parameters:
    r (radius): The a distance from the centre (pc)
    rho0_h (central density): The central density of the halo (M_sol/pc^3)
    Rh (scale radius): The scale radius of the dark matter halo (pc)

    return: mass of the isothermal halo (g)
    '''
    halo_mass, m_err = inte.quad(integrand_h_NFW, 0, r, args=(rho0_h, Rh))
    return halo_mass


def vel_h_NFW(r, rho0_h, Rh):
    '''
    parameters:
    r (radius): The a distance from the centre (pc)
    rho0_h (central density): The central density of the halo (M_sol/pc^3)
    Rh (scale radius): The scale radius of the dark matter halo (pc)

    return: rotational velocity of the NFW halo (pc/s)
    '''
    if isinstance(r, float):
        halo_mass = mass_h_NFW(r, rho0_h, Rh)
    else:
        halo_mass = np.zeros(len(r))
        for i in range(len(r)):
            halo_mass[i] = mass_h_NFW(r[i], rho0_h, Rh)

    vel2 = G * (halo_mass * Msun) / (r * 3.08E16)

    try:
        vel = np.sqrt(vel2) / 1000
    except Warning:
        print('RuntimeWarning in vel_h_NFW')
        print('vel2 = ',vel2)
        print('halo_mass = ',halo_mass)
        print('r = ',r)

    return np.sqrt(vel2)/1000
################################################################################




################################################################################
# NFW_halo
# mass -- already evaluated integral
#-------------------------------------------------------------------------------
def halo_vel_NFW(r, rho0_h, Rh):
    halo_mass = 4*np.pi*rho0_h*Rh**3*((-r/(Rh+r)) + np.log(Rh + r) - np.log(Rh))
    vel2 = G * (halo_mass * Msun) / (r * 3.08E16)
    return np.sqrt(vel2)/1000
################################################################################




################################################################################
# halo (Burket)
#-------------------------------------------------------------------------------
# e = rho_0_Bur
# f = h

def rho_Burket(r, rho0_h, Rh):
    '''
    :param r: The distance from the centre (pc)
    :param rho0_h: The central density of the halo (M_sol/pc^3)
    :param Rh: The scale radius of the dark matter halo (pc)
    :return: volume density of the isothermal halo (M/pc^3)
    '''
    return (rho0_h) / ((1 + (r/Rh)) * (1 + (r/Rh) ** 2))


def integrand_h_Burket(r, rho0_h, Rh):
    '''
    :param r: The a distance from the centre (pc)
    :param rho0_h: The central density of the halo (M_sol/pc^3)
    :param Rh: The scale radius of the dark matter halo (pc)
    :return: integrand for getting the mass of the isothermal halo
    '''
    return 4 * np.pi * (rho_Burket(r, rho0_h, Rh)) * r ** 2


def mass_h_Burket(r, rho0_h, Rh):
    '''
    :param r: The a distance from the centre (pc)
    :param rho0_h: The central density of the halo (M_sol/pc^3)
    :param Rh: The scale radius of the dark matter halo (pc)
    :return: mass of the isothermal halo (g)
    '''
    halo_mass, m_err = inte.quad(integrand_h_Burket, 0, r, args=(rho0_h, Rh))
    return halo_mass


def vel_h_Burket(r, rho0_h, Rh):
    '''
    r (radius): The a distance from the centre [pc]
    rho0_h (central density): The central density of the halo [M_sol/pc^3]
    Rh (scale radius): The scale radius of the dark matter halo [pc]
    :return: rotational velocity of the Burket halo [km/s]
    '''
    if isinstance(r, float):
        halo_mass = mass_h_Burket(r, rho0_h, Rh)
    else:
        halo_mass = np.zeros(len(r))
        for i in range(len(r)):
            halo_mass[i] = mass_h_Burket(r[i], rho0_h, Rh)

    vel2 = G * (halo_mass * Msun) / (r * 3.08E16)

    try:
        vel = np.sqrt(vel2)/1000
    except Warning:
        print('RuntimeWarning in vel_h_Burket')
        print('vel2 = ', vel2)
        print('halo_mass = ', halo_mass)
        print('r = ', r)

    return np.sqrt(vel2)/1000
################################################################################




################################################################################
# Burket halo
# mass -- already evaluated integral
#-------------------------------------------------------------------------------
def halo_vel_Bur(r,rho0_h, Rh):

    halo_mass = np.pi * (-rho0_h) * (Rh**3) * (-np.log(Rh**2 + r**2) - 2*np.log(Rh + r) + 2*np.arctan2(Rh,r) + np.log(Rh**2)\
                                               + 2*np.log(Rh) - 2*np.arctan2(Rh,0))
    vel2 = G * (halo_mass * Msun) / (r * 3.08E16)

    return np.sqrt(vel2) / 1000
################################################################################




################################################################################
# Total Velocity (Fitting Disk Mass)
#-------------------------------------------------------------------------------
# Isothermal Model
#-------------------------------------------------------------------------------
def v_co_iso(r, params):
    '''
    r (radius): The a distance from the centre (kpc)
    params:
      - (scale factor): Scale factor for the bulge [unitless]
      - (interior velocity): The velocity in the bulge [km/s]
      - (central radius): The central radius of the bulge (kpc)
      - (mass of disk): The total mass of the disk [log(Msun)]
      - (disk radius): The central radius of the disk (kpc)
      - (rotational velocity): The rotational velocity when r approaches infinity (km/s)
      - (scale radius): The scale radius of the dark matter halo (kpc)
    '''

    A, Vin, logMdisk, Rd, Vinf, Rh = params

    # Unit conversion
    r_pc = r * 1000
    Mdisk = 10 ** logMdisk
    Rd_pc = Rd * 1000
    Rh_pc = Rh * 1000

    Vbulge = vel_b(r_pc, A, Vin, Rd_pc)
    Vdisk = v_d(r_pc, Mdisk, Rd_pc)
    Vhalo = vel_h_iso(r_pc, Vinf, Rh_pc)

    return np.sqrt(Vbulge ** 2 + Vdisk ** 2 + Vhalo ** 2)  # km/s
#-------------------------------------------------------------------------------


#-------------------------------------------------------------------------------
# Isothermal Model (No Bulge)
#-------------------------------------------------------------------------------
def v_co_iso_nb(r, params):
    '''
    r (radius): The a distance from the centre (kpc)
    params:
      - (mass of disk): The total mass of the disk [log(Msun)]
      - (disk radius): The central radius of the disk (kpc)
      - (rotational velocity): The rotational velocity when r approaches infinity (km/s)
      - (scale radius): The scale radius of the dark matter halo (kpc)
    '''
    logMdisk, Rd, Vinf, Rh = params

    # Unit conversion
    r_pc = r * 1000
    Mdisk = 10 ** logMdisk
    Rd_pc = Rd * 1000
    Rh_pc = Rh * 1000

    Vdisk = v_d(r_pc, Mdisk, Rd_pc)
    Vhalo = vel_h_iso(r_pc, Vinf, Rh_pc)

    return np.sqrt(Vdisk ** 2 + Vhalo ** 2)  # km/s
#-------------------------------------------------------------------------------


#-------------------------------------------------------------------------------
# NFW Model
#-------------------------------------------------------------------------------
def v_co_NFW(r, params):
    '''
    r (radius): The a distance from the centre (kpc)
    params:
      - (scale factor): Scale factor for the bulge [unitless]
      - (interior velocity): The velocity in the bulge [km/s]
      - (central radius): The central radius of the bulge (kpc)
      - (mass of disk): The total mass of the disk [log(Msun)]
      - (disk radius): The central radius of the disk (kpc)
      - (central density): The central density of the halo (M_sol/pc^3)
      - (scale radius): The scale radius of the dark matter halo (kpc)
    '''

    A, Vin, logMdisk, Rd, rho0_h, Rh = params

    # Unit conversion
    r_pc = r * 1000
    Mdisk = 10 ** logMdisk
    Rd_pc = Rd * 1000
    Rh_pc = Rh * 1000

    Vbulge = vel_b(r_pc, A, Vin, Rd_pc)
    Vdisk = v_d(r_pc, Mdisk, Rd_pc)
    Vhalo = vel_h_NFW(r_pc, rho0_h, Rh_pc)

    return np.sqrt(Vbulge ** 2 + Vdisk ** 2 + Vhalo ** 2)  # km/s
#-------------------------------------------------------------------------------


#-------------------------------------------------------------------------------
# NFW Model (No Bulge)
#-------------------------------------------------------------------------------
def v_co_NFW_nb(r, params):
    '''
    r (radius): The a distance from the centre (kpc)
    params:
      - (mass of disk): The total mass of the disk [log(Msun)]
      - (disk radius): The central radius of the disk (kpc)
      - (central density): The central density of the halo (M_sol/pc^3)
      - (scale radius): The scale radius of the dark matter halo (kpc)
    '''

    logMdisk, Rd, rho0_h, Rh = params

    # Unit conversion
    r_pc = r * 1000
    Mdisk = 10 ** logMdisk
    Rd_pc = Rd * 1000
    Rh_pc = Rh * 1000

    Vdisk = v_d(r_pc, Mdisk, Rd_pc)
    Vhalo = vel_h_NFW(r_pc, rho0_h, Rh_pc)

    return np.sqrt(Vdisk ** 2 + Vhalo ** 2)  # km/s
#-------------------------------------------------------------------------------


#-------------------------------------------------------------------------------
# Burket Model
#-------------------------------------------------------------------------------
def v_co_Burket(r, params):
    '''
    r (radius): The a distance from the centre (kpc)
    params:
      - (scale factor): Scale factor for the bulge [unitless]
      - (interior velocity): The velocity in the bulge [km/s]
      - (central radius): The central radius of the bulge (kpc)
      - (mass of disk): The total mass of the disk [log(Msun)]
      - (disk radius): The central radius of the disk (kpc)
      - (central density): The central density of the halo (M_sol/pc^3)
      - (scale radius): The scale radius of the dark matter halo (kpc)
    '''
    A, Vin, logMdisk, Rd, rho0_h, Rh = params

    # Unit conversion
    r_pc = r * 1000
    Mdisk = 10 ** logMdisk
    Rd_pc = Rd * 1000
    Rh_pc = Rh * 1000

    Vbulge = vel_b(r_pc, A, Vin, Rd_pc)
    Vdisk = v_d(r_pc, Mdisk, Rd_pc)
    Vhalo = vel_h_Burket(r_pc, rho0_h, Rh_pc)

    return np.sqrt(Vbulge ** 2 + Vdisk ** 2 + Vhalo ** 2)  # km/s
#-------------------------------------------------------------------------------


#-------------------------------------------------------------------------------
# Burket Model (No Bulge)
#-------------------------------------------------------------------------------
def v_co_Burket_nb(r, params):
    '''
    r (radius): The a distance from the centre (kpc)
    params:
      - (scale factor): Scale factor for the bulge [unitless]
      - (interior velocity): The velocity in the bulge [km/s]
      - (central radius): The central radius of the bulge (kpc)
      - (mass of disk): The total mass of the disk [log(Msun)]
      - (disk radius): The central radius of the disk (kpc)
      - (central density): The central density of the halo (M_sol/pc^3)
      - (scale radius): The scale radius of the dark matter halo (kpc)
    '''
    logMdisk, Rd, rho0_h, Rh = params

    # Unit conversion
    r_pc = r * 1000
    Mdisk = 10 ** logMdisk
    Rd_pc = Rd * 1000
    Rh_pc = Rh * 1000

    Vdisk = v_d(r_pc, Mdisk, Rd_pc)
    Vhalo = vel_h_Burket(r_pc, rho0_h, Rh_pc)

    return np.sqrt(Vdisk ** 2 + Vhalo ** 2)  # km/s
#-------------------------------------------------------------------------------
################################################################################




################################################################################
# Total Velocity (Fitting Disk Central Density)
#-------------------------------------------------------------------------------
# Isothermal Model
#-------------------------------------------------------------------------------
def v_tot_iso(r, params):
    '''
    r (radius): The a distance from the centre (kpc)
    params:
      - (scale factor): Scale factor for the bulge [unitless]
      - (interior velocity): The velocity in the bulge [km/s]
      - (central radius): The central radius of the bulge (kpc)
      - (mass of disk): The total mass of the disk [log(Msun)]
      - (disk radius): The central radius of the disk (kpc)
      - (rotational velocity): The rotational velocity when r approaches infinity (km/s)
      - (scale radius): The scale radius of the dark matter halo (kpc)
    '''

    ############################################################################
    # Parse the fit parameters
    #---------------------------------------------------------------------------
    A, Vin, SigD, Rd, Vinf, Rh = params

    #print('A in v_tot_iso:', A)
    #print('Vin in v_tot_iso:', Vin)
    ############################################################################


    ############################################################################
    # Calculate the square of the velocity at the orbital radius
    #---------------------------------------------------------------------------
    if r == 0:
        ########################################################################
        # The velocity should go to 0 as r goes to 0.  The current model for the 
        # bulge, though, does not follow this behavior, so we need to fix this 
        # particular value.
        #-----------------------------------------------------------------------
        v2 = 0
        ########################################################################
    else:
        ########################################################################
        # Unit conversion
        #-----------------------------------------------------------------------
        r_pc = r * 1000
        Rd_pc = Rd * 1000
        Rh_pc = Rh * 1000
        ########################################################################


        ########################################################################
        # Velocity due to the bulge
        #-----------------------------------------------------------------------
        Vbulge = vel_b(r_pc, A, Vin, Rd_pc)
        #Vbulge2 = vel_b2(r_pc, A, Vin, Rd_pc)
        ########################################################################


        ########################################################################
        # Velocity due to the disk
        #-----------------------------------------------------------------------
        Vdisk = disk_vel(r_pc, SigD, Rd_pc)
        ########################################################################


        ########################################################################
        # Velocity due to the halo
        #-----------------------------------------------------------------------
        Vhalo = vel_h_iso(r_pc, Vinf, Rh_pc)
        #Vhalo = halo_vel_iso(r_pc, , Rh_pc)
        ########################################################################
        

        ########################################################################
        # Total velocity from all three components
        #-----------------------------------------------------------------------
        v2 = Vbulge**2 + Vdisk**2 + Vhalo**2
        #v2 = Vbulge2 + Vdisk**2 + Vhalo**2
        ########################################################################
    ############################################################################
    

    ############################################################################
    # Calculate the velocity (instead of the square of the velocity)
    #---------------------------------------------------------------------------
    try:
        v = np.sqrt(v2)
    except RuntimeWarning:
        print('Vbulge2:', Vbulge2)
        print('Vdisk2:', Vdisk**2)
        print('Vhalo2:', Vhalo**2)
    ############################################################################

    return v  # km/s
#-------------------------------------------------------------------------------


#-------------------------------------------------------------------------------
# Isothermal Model (No Bulge)
#-------------------------------------------------------------------------------
def v_tot_iso_nb(r, params):
    '''
    r (radius): The a distance from the centre (kpc)
    params:
      - (mass of disk): The total mass of the disk [log(Msun)]
      - (disk radius): The central radius of the disk (kpc)
      - (rotational velocity): The rotational velocity when r approaches infinity (km/s)
      - (scale radius): The scale radius of the dark matter halo (kpc)
    '''
    SigD, Rd, Vinf, Rh = params


    # Unit conversion
    r_pc = r * 1000
    Rd_pc = Rd * 1000
    Rh_pc = Rh * 1000

    Vdisk = disk_vel(r_pc, SigD, Rd_pc)
    Vhalo = vel_h_iso(r_pc, Vinf, Rh_pc)
    v2 = Vdisk ** 2 + Vhalo ** 2

    v = np.sqrt(v2)

    return v  # km/s
#-------------------------------------------------------------------------------


#-------------------------------------------------------------------------------
# NFW Model
#-------------------------------------------------------------------------------
def v_tot_NFW(r, params):
    '''
    r (radius): The a distance from the centre (kpc)
    params:
      - (scale factor): Scale factor for the bulge [unitless]
      - (interior velocity): The velocity in the bulge [km/s]
      - (central radius): The central radius of the bulge (kpc)
      - (mass of disk): The total mass of the disk [log(Msun)]
      - (disk radius): The central radius of the disk (kpc)
      - (central density): The central density of the halo (M_sol/pc^3)
      - (scale radius): The scale radius of the dark matter halo (kpc)
    '''

    A, Vin, SigD, Rd, rho0_h, Rh = params


    # Unit conversion
    r_pc = r * 1000
    Rd_pc = Rd * 1000
    Rh_pc = Rh * 1000

    Vbulge = vel_b(r_pc, A, Vin, Rd_pc)
    Vdisk = disk_vel(r_pc, SigD, Rd_pc)
    Vhalo = vel_h_NFW(r_pc, rho0_h, Rh_pc)
    v2 = Vbulge ** 2 + Vdisk ** 2 + Vhalo ** 2

    v = np.sqrt(v2)

    return v  # km/s
#-------------------------------------------------------------------------------


#-------------------------------------------------------------------------------
# NFW Model (No Bulge)
#-------------------------------------------------------------------------------
def v_tot_NFW_nb(r, params):
    '''
    r (radius): The a distance from the centre (kpc)
    params:
      - (mass of disk): The total mass of the disk [log(Msun)]
      - (disk radius): The central radius of the disk (kpc)
      - (central density): The central density of the halo (M_sol/pc^3)
      - (scale radius): The scale radius of the dark matter halo (kpc)
    '''

    SigD, Rd, rho0_h, Rh = params

    if r == 0:
        v2 = 0
    else:
        # Unit conversion
        r_pc = r * 1000
        Rd_pc = Rd * 1000
        Rh_pc = Rh * 1000

        Vdisk = disk_vel(r_pc, SigD, Rd_pc)
        Vhalo = vel_h_NFW(r_pc, rho0_h, Rh_pc)
        v2 = Vdisk ** 2 + Vhalo ** 2

    v = np.sqrt(v2)

    return v  # km/s
#-------------------------------------------------------------------------------


#-------------------------------------------------------------------------------
# Burket Model
#-------------------------------------------------------------------------------
def v_tot_Burket(r, params):
    '''
    r (radius): The a distance from the centre (kpc)
    params:
      - (scale factor): Scale factor for the bulge [unitless]
      - (interior velocity): The velocity in the bulge [km/s]
      - (central radius): The central radius of the bulge (kpc)
      - (mass of disk): The total mass of the disk [log(Msun)]
      - (disk radius): The central radius of the disk (kpc)
      - (central density): The central density of the halo (M_sol/pc^3)
      - (scale radius): The scale radius of the dark matter halo (kpc)
    '''
    A, Vin, SigD, Rd, rho0_h, Rh = params


    # Unit conversion
    r_pc = r * 1000
    Rd_pc = Rd * 1000
    Rh_pc = Rh * 1000

    Vbulge = vel_b(r_pc, A, Vin, Rd_pc)
    Vdisk = disk_vel(r_pc, SigD, Rd_pc)
    Vhalo = vel_h_Burket(r_pc, rho0_h, Rh_pc)
    v2 = Vbulge ** 2 + Vdisk ** 2 + Vhalo ** 2

    v = np.sqrt(v2)

    return v  # km/s
#-------------------------------------------------------------------------------


#-------------------------------------------------------------------------------
# Burket Model (No Bulge)
#-------------------------------------------------------------------------------
def v_tot_Burket_nb(r, params):
    '''
    r (radius): The a distance from the centre (kpc)
    params:
      - (scale factor): Scale factor for the bulge [unitless]
      - (interior velocity): The velocity in the bulge [km/s]
      - (central radius): The central radius of the bulge (kpc)
      - (mass of disk): The total mass of the disk [log(Msun)]
      - (disk radius): The central radius of the disk (kpc)
      - (central density): The central density of the halo (M_sol/pc^3)
      - (scale radius): The scale radius of the dark matter halo (kpc)
    '''
    SigD, Rd, rho0_h, Rh = params

    if r == 0:
        v2 = 0
    else:
        # Unit conversion
        r_pc = r * 1000
        Rd_pc = Rd * 1000
        Rh_pc = Rh * 1000

        Vdisk = disk_vel(r_pc, SigD, Rd_pc)
        Vhalo = vel_h_Burket(r_pc, rho0_h, Rh_pc)
        v2 = Vdisk ** 2 + Vhalo ** 2

    v = np.sqrt(v2)

    return v # km/s
#-------------------------------------------------------------------------------
################################################################################


################################################################################
# Loglike function (Burket w/ bulge)
#-------------------------------------------------------------------------------
def loglike_Iso(theta, r, v, v_err):
    model = v_co_iso(np.array(r), theta)

    inv_sigma2 = 1.0 / (np.array(v_err) ** 2)

    logL = -0.5 * (np.sum((np.array(v) - model) ** 2 * inv_sigma2 - np.log(inv_sigma2)))

    # Additional (physical) penalties
    if theta[3] < theta[1]:
        logL += 1E6

    return logL


# Negative likelihood
def nloglike_Iso(theta, r, v, v_err):
    return -loglike_Iso(theta, r, v, v_err)
#################################################################################




################################################################################
# Loglike function (Burket no bulge)
#-------------------------------------------------------------------------------
def loglike_Iso_nb(theta, r, v, v_err):
    model = v_co_iso_nb(np.array(r), theta)

    inv_sigma2 = 1.0 / (np.array(v_err) ** 2)

    logL = -0.5 * (np.sum((np.array(v) - model) ** 2 * inv_sigma2 - np.log(inv_sigma2)))

    # Additional (physical) penalties
    if theta[3] < theta[1]:
        logL += 1E10

    return logL

# Negative likelihood
def nloglike_Iso_nb(theta, r, v, v_err):
    return -loglike_Iso_nb(theta, r, v, v_err)
#################################################################################




#################################################################################
# Fitting Function (Isothermal)
#-------------------------------------------------------------------------------
def RC_fitting_Iso(r,m,v,v_err):
    '''

    :param r: The a distance from the centre (kpc)
    :param m: Mass of the object (M_sol)
    :param v: rotational velocity (km/s)
    :param v_err: error in the rotational velocity (km/s)
    :return: The fitted parameters
    '''
    # variables for initial guesses
    a_guess = 0.2
    v_inf_b_guess = 150
    logM_guess = np.log10(m)+0.5
    r_d_guess = max(np.array(r)) / 3
    v_inf_h_guess = 200
    r_h_guess = max(list(r))*10
    if max(list(r)) < 5:
        logM_guess += 0.5
    half_idx = int(0.5 * len(r))
    if v[half_idx] < max(v[:half_idx]):
            p0 = [a_guess, v_inf_b_guess,logM_guess , r_d_guess , v_inf_h_guess, r_h_guess]
            param_bounds = [[0.2, 1],  # Scale Factor [unitless]
                            [0.001, 1000],  # Bulge Scale Velocity [km/s]
                            [8, 12],  # Disk mass [log(Msun)]
                            [0.1,20],  # Disk radius [kpc]
                            [0.001, 1000],  # Halo density [Msun/pc^2]
                            [0.1, 1000]]  # Halo radius [kpc]

            bestfit = minimize(nloglike_Iso, p0, args=(r, v, v_err),
                              bounds=param_bounds)
            print('---------------------------------------------------')
            print(bestfit)
    else: # No Bulge
            p0 = [logM_guess, r_d_guess, v_inf_h_guess, r_h_guess]
            param_bounds = [[8, 12],  # Disk mass [log(Msun)]
                            [0.1, 20],  # Disk radius [kpc]
                            [0.001, 1000],  # Halo density [Msun/pc^2]
                            [0.1, 1000]]  # Halo radius [kpc]

            bestfit = minimize(nloglike_Iso_nb, p0, args=(r, v, v_err),
                              bounds=param_bounds)
            print('---------------------------------------------------')
            print(bestfit)
    return bestfit
#################################################################################




#################################################################################
# Plotting (Isohermal)
#-------------------------------------------------------------------------------
def RC_plotting_Iso(r, v, v_err, bestfit, ID):
    half_idx = int(0.5 * len(r))
    if v[half_idx] < max(v[:half_idx]):
            if max(list(r)) < bestfit.x[3]:
                r_plot = np.linspace(0,3*bestfit.x[3],100)
            else:
                r_plot = np.linspace(0,3*max(list(r)),100)
            plt.errorbar(r, v, yerr=v_err, fmt='g*', label='data')
            plt.plot(r_plot, v_co_iso(np.array(r_plot), bestfit.x), '--', label='fit')
            plt.plot(r_plot, vel_b(np.array(r_plot) * 1000, bestfit.x[0], bestfit.x[1], bestfit.x[3] * 1000),color='green',
                 label='bulge')
            plt.plot(r_plot, v_d(np.array(r_plot) * 1000, 10 ** bestfit.x[2], bestfit.x[3] * 1000),color='orange',label='disk')
            plt.plot(r_plot, vel_h_iso(np.array(r_plot) * 1000, bestfit.x[4],bestfit.x[5] * 1000),color='blue',
                 label='Isothermal halo')
            plt.legend()
            plt.xlabel('$r_{dep}$ [kpc]')
            plt.ylabel('$v_{rot}$ [km/s]')
            plt.title(ID)
            plt.show()
    else:
            if max(list(r)) < bestfit.x[1]:
                r_plot = np.linspace(0,3*bestfit.x[1],100)
            else:
                r_plot = np.linspace(0,3*max(list(r)),100)
            plt.errorbar(r, v, yerr=v_err, fmt='g*', label='data')
            plt.plot(r_plot, v_co_iso_nb(np.array(r_plot), bestfit.x), '--', label='fit')
            plt.plot(r_plot, v_d(np.array(r_plot) * 1000, 10 ** bestfit.x[0], bestfit.x[1] * 1000),color='orange',
                    label='disk')
            plt.plot(r_plot, vel_h_iso(np.array(r_plot) * 1000, bestfit.x[2], bestfit.x[3] * 1000),color='blue',
                    label='Isothermal halo')
            plt.legend()
            plt.xlabel('$r_{dep}$ [kpc]')
            plt.ylabel('$v_{rot}$ [km/s]')
            plt.title(ID)
            plt.show()
#################################################################################




################################################################################
# Loglike function (Burket w/ bulge)
#-------------------------------------------------------------------------------
def loglike_Bur(theta, r, v, v_err, WF50):
    model = v_co_Burket(np.array(r), theta)

    inv_sigma2 = 1.0 / (np.array(v_err) ** 2)

    logL = -0.5 * (np.sum((np.array(v) - model) ** 2 * inv_sigma2 - np.log(inv_sigma2)))
    return logL


# Negative likelihood
def nloglike_Bur(theta, r, v, v_err,WF50):
    nlogL = -loglike_Bur(theta, r, v, v_err, WF50)
    # Additional (physical) penalties
    # If disk radius greater than halo radius
    if theta[3] < theta[1]:
        nlogL += 1E6
    # If max velocity greater than HI
    if v_co_Burket(5 * max(np.array(r)), theta) > WF50:
        nlogL += 1E3
    return nlogL
#################################################################################




################################################################################
# Loglike function (Burket no bulge)
#-------------------------------------------------------------------------------
def loglike_Bur_nb(theta, r, v, v_err,WF50):
    model = v_co_Burket_nb(np.array(r), theta)

    inv_sigma2 = 1.0 / (np.array(v_err) ** 2)

    logL = -0.5 * (np.sum((np.array(v) - model) ** 2 * inv_sigma2 - np.log(inv_sigma2)))
    return logL

# Negative likelihood
def nloglike_Bur_nb(theta, r, v, v_err,WF50):
    nlogL = -loglike_Bur_nb(theta, r, v, v_err,WF50)
    # Additional (physical) penalties
    # If disk radius greater than halo radius
    if theta[3] < theta[1]:
        nlogL += 1E6
    # If max velocity greater than HI
    if v_co_Burket_nb(5*max(np.array(r)),theta) > WF50:
        nlogL += 1E3
    return nlogL
#################################################################################




#################################################################################
# Fitting Function (Burket)
#-------------------------------------------------------------------------------
def RC_fitting_Bur(r,m,v,v_err,WF50):
    '''
    :param r: The a distance from the centre (kpc)
    :param m: Mass of the object (M_sol)
    :param v: rotational velocity (km/s)
    :param v_err: error in the rotational velocity (km/s)
    :param WF50: HI data (km/s)
    :return: The fitted parameters
    '''
    # variables for initial guesses
    a_guess = 0.2
    v_inf_guess = 150
    logM_guess = np.log10(m)
    r_d_guess = max(np.array(r))/5.25
    rho_dc_guess = 0.0051
    r_h_guess = max(list(r))*1.1
    if max(list(r)) < 5:
        rho_dc_guess /= 100
    half_idx = int(0.5 * len(r))
    if v[half_idx] < max(v[:half_idx]):
            p0 = [a_guess, v_inf_guess,logM_guess , r_d_guess , rho_dc_guess, r_h_guess]
            param_bounds = [[0.2, 1],  # Scale Factor [unitless]
                            [0.001, 1000],  # Bulge Scale Velocity [km/s]
                            [8, 12],  # Disk mass [log(Msun)]
                            [0.1,20],  # Disk radius [kpc]
                            [0.0001, 1],  # Halo density [Msun/pc^2]
                            [0.1, 1000]]  # Halo radius [kpc]

            bestfit = minimize(nloglike_Bur, p0, args=(r, v, v_err,WF50),
                              bounds=param_bounds)
            print('---------------------------------------------------')
            print(bestfit)
    else: # No Bulge
            p0 = [logM_guess, r_d_guess, rho_dc_guess, r_h_guess]
            param_bounds = [[8, 12],  # Disk mass [log(Msun)]
                            [0.1, 20],  # Disk radius [kpc]
                            [0.0001, 1],  # Halo density [Msun/pc^2]
                            [0.1, 1000]]  # Halo radius [kpc]

            bestfit = minimize(nloglike_Bur_nb, p0, args=(r, v, v_err,WF50),
                              bounds=param_bounds)
            print('---------------------------------------------------')
            print(bestfit)
    return bestfit
################################################################################




################################################################################
# Plotting (Burket)
#-------------------------------------------------------------------------------
def RC_plotting_Bur(r,v, v_err, bestfit, ID):
    half_idx = int(0.5 * len(r))
    if v[half_idx] < max(v[:half_idx]):
            if max(list(r)) < bestfit.x[3]:
                r_plot = np.linspace(0,3*bestfit.x[3],100)
            else:
                r_plot = np.linspace(0,3*max(list(r)),100)
            plt.errorbar(r, v, yerr=v_err, fmt='g*', label='data')
            plt.plot(r_plot, v_co_Burket(np.array(r_plot), bestfit.x), '--', label='fit')
            plt.plot(r_plot, vel_b(np.array(r_plot) * 1000, bestfit.x[0], bestfit.x[1], bestfit.x[3] * 1000),color='green',
                 label='bulge')
            plt.plot(r_plot, v_d(np.array(r_plot) * 1000, 10 ** bestfit.x[2], bestfit.x[3] * 1000),color='orange',label='disk')
            plt.plot(r_plot, vel_h_Burket(np.array(r_plot) * 1000, bestfit.x[4],bestfit.x[5] * 1000),color='blue',
                 label='Burket halo')
            plt.legend()
            plt.xlabel('$r_{dep}$ [kpc]')
            plt.ylabel('$v_{rot}$ [km/s]')
            plt.title(ID)
            plt.show()
    else:
            if max(list(r)) < bestfit.x[1]:
                r_plot = np.linspace(0,3*bestfit.x[1],100)
            else:
                r_plot = np.linspace(0,3*max(list(r)),100)
            plt.errorbar(r, v, yerr=v_err, fmt='g*', label='data')
            plt.plot(r_plot, v_co_Burket_nb(np.array(r_plot), bestfit.x), '--', label='fit')
            plt.plot(r_plot, v_d(np.array(r_plot) * 1000, 10 ** bestfit.x[0], bestfit.x[1] * 1000),color='orange',
                    label='disk')
            plt.plot(r_plot, vel_h_Burket(np.array(r_plot) * 1000, bestfit.x[2], bestfit.x[3] * 1000),color='blue',
                    label='Burket halo')
            plt.legend()
            plt.xlabel('$r_{dep}$ [kpc]')
            plt.ylabel('$v_{rot}$ [km/s]')
            plt.title(ID)
            plt.show()
################################################################################
