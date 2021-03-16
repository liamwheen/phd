"""
Calculate Qyear from paper but numerically rather than analytically. This
allows for albedo function with SZA dependance to be incorporated. It might
also be used in the sub-year resolution model."""
import numpy as np
from scipy.optimize import root_scalar
from scipy.integrate import dblquad
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
phi_n, gamma_n = 50, 200
phi = np.arcsin(np.linspace(0,1,phi_n))
y = np.linspace(0,1,phi_n)
gamma = np.linspace(0, 2*np.pi, gamma_n) 

rep_phi, rep_gamma = cartesian_product(phi,gamma)

def tester():
    Q = Q_days(np.linspace(0,365/2,10),beta0,rho0,eps0)
    print(Q(np.linspace(0,1,300),np.linspace(0,365/2,20)).T.shape)
    plt.plot(Q(np.linspace(0,1,300),np.linspace(0,365/2,20)).T)
    plt.show()

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
    return mag*np.sin(theta.reshape(-1,1)+phase)

def Q_days(t, beta, rho, eps):
    # Takes array of days of year, 0 is theta=0, 183 is theta=pi
    # Returns interp2d of shape len(t)*phi_n
    r, theta = polar_pos(eps, t)
    mag, phase = trig_coefs(beta, rho)
    I = I_fast(mag, phase, theta)
    I[I<0] = 0
    Q_days = K/(4*np.pi*r**2)*np.mean(I.reshape(-1,phi_n,gamma_n),2).T
    return interp2d(y, t, Q_days.T)

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
    tester()
    """
    import cProfile, pstats
    profiler = cProfile.Profile()
    profiler.enable()
    main()
    profiler.disable()
    stats = pstats.Stats(profiler).sort_stats('cumulative')
    #stats.print_stats()
    """
