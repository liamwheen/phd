"""
Calculate Qyear from paper but numerically rather than analytically. This
allows for albedo function with SZA dependance to be incorporated. It might
also be used in the sub-year resolution model."""
import numpy as np
from scipy.optimize import root_scalar
from scipy.integrate import quad_vec
from scipy.interpolate import interp1d, interp2d
from insol_sympy import calc_yearly_average
from matplotlib import pyplot as plt

def cartesian_product(a, b):
    arr = np.empty((len(a), len(b), 2))
    for i, val in enumerate(np.ix_(a,b)):
        arr[...,i] = val
    return arr.reshape(-1,2).T

year2sec = 31556952 #Translate time dependent units to 'per year' instead of 'per second'
year = 365.2425

K = 3.8284e+26
au = 149597870700
beta0 = 0.4090
eps0 = 0.0167
rho0 = 2.9101
eta0 = 0.94
phi_n, gamma_n = 50, 50
#phi = np.arcsin(np.linspace(0,1,phi_n))
phi = np.linspace(0,np.pi/2,phi_n)
y = np.linspace(0,1,phi_n)
# Longitudinal symmetry means we can save time just looking at one hemisphere
# for Q_year
gamma = np.linspace(0, np.pi, gamma_n) 
year_res = 20
t_span = np.linspace(0, year, year_res)[:-1]

rep_phi, rep_gamma = cartesian_product(phi,gamma)

def s_b(beta):
    c_b = (5/16)*(3*np.sin(beta)**2 - 2)
    return 1 + 0.5*c_b*(3*y**2 - 1)
Q_0 = 340.327

def main():
    Q_e = Q_0/(np.sqrt(1-eps0**2))
    Q = Q_year(beta0,rho0,eps0)(y)
    Q_bud = Q_e*s_b()
    plt.plot(Q)
    plt.plot(Q_bud)
    plt.show()

def sza_albedo(I):
    albedo = (1+1.21)/(1+1.57*I)*0.25
    # Ice albedo from y=0.94 onwards
    ice_inds = int((1-eta)*phi_n*gamma_n)
    albedo[-ice_inds:] = (1+1.21)/(1+1.57*I[-ice_inds:])*0.42
    return albedo

def trig_coefs(beta, rho):
    # mag and phase for Asin(x)+Bcos(x) formula
    mag = np.sqrt(2*np.sin(beta)*np.sin(rep_phi)*np.cos(beta)*np.cos(rep_gamma)*np.cos(rep_phi) +
            np.cos(beta)**2*np.cos(rep_gamma)**2*np.cos(rep_phi)**2 +
            np.cos(beta)**2*np.cos(rep_phi)**2 - np.cos(beta)**2 -
            np.cos(rep_gamma)**2*np.cos(rep_phi)**2 + 1) 

    phase = np.arctan2((-np.sin(beta)*np.sin(rep_phi)*np.cos(rho) +
            np.sin(rep_gamma)*np.sin(rho)*np.cos(rep_phi) -
            np.cos(beta)*np.cos(rep_gamma)*np.cos(rep_phi)*np.cos(rho)),
            (-np.sin(beta)*np.sin(rep_phi)*np.sin(rho) -
            np.sin(rep_gamma)*np.cos(rep_phi)*np.cos(rho) -
            np.sin(rho)*np.cos(beta)*np.cos(rep_gamma)*np.cos(rep_phi)))
    return mag, phase

def I_fast(mag, phase, theta):
    return np.maximum(mag*np.sin(theta+phase),0)

def Q_year(beta, rho, eps):
    Q_e = Q_0/(np.sqrt(1-eps**2))
    return interp1d(y,s_b(beta)*Q_e)
    # Returns Qs interpolated function 
    mag, phase = trig_coefs(beta, rho)
    Is = np.zeros(phi_n*gamma_n)
    rs, thetas = polar_pos(eps, t_span)
    for r, theta in zip(rs,thetas):
        Is += I_fast(mag, phase, theta)/r**2
    Is = Is.reshape(phi_n,gamma_n)/len(t_span)
    return interp1d(np.sin(phi),K/(4*np.pi)*np.mean(Is,1),'cubic')

def midpoint_E(M, eps):
    E_func = lambda E: E - eps*np.sin(E) - M
    mid_E = root_scalar(E_func, method='brentq', bracket=(M-eps,M+eps))
    return mid_E.root

def calc_theta(E, eps):
    sign = np.ones(len(E))
    sign[(E>np.pi)] = -1
    return np.pi+sign*2*np.arctan(np.sqrt((1+eps)/(1-eps)*(np.tan(E/2))**2))

def polar_pos(eps, t):
    """ Using Kepler's law:
    https://en.wikipedia.org/wiki/Kepler%27s_laws_of_planetary_motion#Position_as_a_function_of_time
    """
    t = (t+year/2)%year
    M = t*2*np.pi/year
    E = np.array([midpoint_E(m, eps) for m in M])
    theta = calc_theta(E, eps)
    r = au*(1 - eps*np.cos(E))
    return r, theta

def I_t(t, mag, phase):
    r, theta = polar_pos(eps, np.array([t]))
    I = I_fast(mag, phase, theta)
    I[I<0] = 0
    I*=1-sza_albedo(I)
    I = I.reshape(phi_n,gamma_n)
    return I

def anim_main():
    import matplotlib.animation as animation
    mag, phase = trig_coefs(beta, rho)
    def anim(I):
        im.set_data(np.flipud(I))
    def frames():
        for t in np.linspace(0,year,300):
            yield I_t(t, mag, phase)
    fig, ax = plt.subplots()
    im = plt.imshow(np.zeros((len(phi),len(gamma))),vmin=0, vmax=1)
    ani = animation.FuncAnimation(fig,anim,frames=frames,interval=1,repeat=True)
    plt.show()


if __name__ == '__main__':
    import cProfile, pstats
    profiler = cProfile.Profile()
    profiler.enable()
    main()
    profiler.disable()
    stats = pstats.Stats(profiler).sort_stats('cumulative')
    #stats.print_stats()
