"""
Calculate Qyear from paper but numerically rather than analytically. This
allows for albedo function with SZA dependance to be incorporated. It might
also be used in the sub-year resolution model."""
import numpy as np
from scipy.optimize import root_scalar
from scipy.integrate import dblquad
from insol_sympy import calc_yearly_average
from matplotlib import pyplot as plt

def cartesian_product(a, b):
    arr = np.empty((phi_n, gamma_n, 2))
    for i, val in enumerate(np.ix_(a,b)):
        arr[...,i] = val
    return arr.reshape(-1,2).T

year2sec = 31556952 #Translate time dependent units to 'per year' instead of 'per second'
year = 365.2425

K = 3.8284e+26
au = 149597870700
beta = 0.4090
eps = 0.0167
rho = 2.9101
eta = 0.94
phi_n, gamma_n = 200, 100
phi = np.arcsin(np.linspace(0,1,phi_n))
# Longitudinal symmetry means we can save time just looking at one hemisphere
# for Q_year, but if using Q_day, we need the full range
gamma = np.linspace(0, 2*np.pi,gamma_n) 

rep_phi, rep_gamma = cartesian_product(phi,gamma)

def main():
    num_Q = Q_year(beta, rho, eps)
    print(np.mean(num_Q))
    ice_inds = int((1-eta)*phi_n)
    c_b = (5/16)*(3*np.sin(beta)**2 - 2)
    Qs = 0.722*(1 + 0.58*c_b*(3*np.sin(phi)**2 -
        1))*K/(16*np.pi*au**2*np.sqrt(1-eps**2))
    Qs[-ice_inds:] = 0.522*(1 + 0.68*c_b*(3*np.sin(phi[-ice_inds:])**2 -
        1))*K/(16*np.pi*au**2*np.sqrt(1-eps**2))
    print(np.mean(Qs))
    #ana_Q = calc_yearly_average(beta, phi, eps)*(1-0.29)
    #print(np.mean(num_Q))
    #print(np.mean(ana_Q))
    plt.plot(np.linspace(0,1,phi_n), num_Q)
    plt.plot(np.linspace(0,1,phi_n), Qs)
    #plt.plot(np.linspace(0,1,phi_n), ana_Q)
    plt.show()
    print('Mean Error: ',np.mean((abs(num_Q-Qs)/num_Q)))
    print('Mean Grad Error: ',np.mean(abs(np.diff(num_Q)-np.diff(Qs))/np.diff(num_Q)))

def sza_albedo(I):
    albedo = (1+1.21)/(1+1.57*I)*0.25
    # Ice albedo from y=0.94 onwards
    ice_inds = int((1-eta)*phi_n*gamma_n)
    albedo[-ice_inds:] = (1+1.21)/(1+1.57*I[-ice_inds:])*0.42
    #albedo[I==0] = 0
    #plt.plot(np.mean(albedo.reshape(phi_n,gamma_n),1))
    #plt.show()
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
    return mag*np.sin(theta+phase)

def Q_day(t, beta, rho, eps):
    # Takes day of year, 0 is theta=0, 183 is theta=pi
    r, theta = polar_pos(eps, np.array([t]))
    mag, phase = trig_coefs(beta, rho)
    I = I_fast(mag, phase, theta)
    I[I<0] = 0
    return K/(4*np.pi*r**2)*np.mean(I.reshape(phi_n,gamma_n),1)

def Q_year(beta, rho, eps):
    t_span = np.linspace(0, year, 500)
    mag, phase = trig_coefs(beta, rho)
    Is = np.zeros(phi_n*gamma_n)
    rs, thetas = polar_pos(eps, t_span)
    for r, theta in zip(rs,thetas):
        I = I_fast(mag, phase, theta)
        I[I<0] = 0
        #I*=1-sza_albedo(I)
        Is+=I/r**2
    Is = Is.reshape(phi_n,gamma_n)/len(t_span)
    return K/(4*np.pi)*np.mean(Is,1)

def midpoint_E(M, eps):
    E_func = lambda E: E - eps*np.sin(E) - M
    mid_E = root_scalar(E_func, method='brentq', bracket=(M-eps,M+eps))
    return mid_E.root

def calc_theta(E, eps):
    sign = np.ones(len(E))# if E < np.pi else -1
    sign[(E<np.pi)] = -1
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
