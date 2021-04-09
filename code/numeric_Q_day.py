"""
Calculate Qday from paper but numerically rather than analytically. This
can be used in the sub-year resolution model."""
import numpy as np
from scipy.optimize import root_scalar
from scipy.integrate import quad_vec
from scipy.interpolate import interp1d
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
phi_n, gamma_n = 60, 50
phi = np.arcsin(np.linspace(-1,1,phi_n))
#phi = np.linspace(-np.pi/2,np.pi/2,phi_n)
gamma = np.linspace(0, 2*np.pi, gamma_n) 

rep_phi, rep_gamma = cartesian_product(phi,gamma)

def tester():
    #Q_now = np.array([Q_day(t, beta0, rho0, eps0)(np.linspace(-1,1,100)) for t in
    #        np.linspace(0,year,300)])
    qgood = Q_day(100, beta0, rho0, eps0)(np.linspace(0,1,100))
    #qbad= Q_day_bad(100, beta0, rho0, eps0)(np.linspace(0,1,100))
    plt.plot(qgood)
    #plt.plot(qbad)
    plt.show()
    Q_past = Q_day(year/2, beta0, rho0+np.pi, eps0)(np.linspace(0,1,100))
    Q_now = Q_day(0, beta0, rho0, eps0)(np.linspace(0,1,100))
    Q_max = Q_day(year/2, beta0, 0, eps0)(np.linspace(0,1,100))
    plt.plot(Q_now)
    plt.plot(Q_past)
    plt.plot(Q_max)
    #plt.plot(Q_past(np.linspace(-1,1,100)))
    plt.show()

def I(gamma, phi, beta, rho, theta):
    return np.maximum((np.sin(gamma)*np.sin(rho - theta) - np.cos(beta)*np.cos(gamma)*np.cos(rho -
                theta))*np.cos(phi) - np.sin(beta)*np.sin(phi)*np.cos(rho - theta), 0)

def trig_coefs(beta, rho, theta):
    # mag and phase for Asin(x)+Bcos(x) + c formula
    mag = np.sqrt((-np.sin(beta)**2*np.cos(rho - theta)**2 + 1)*np.cos(phi)**2)
    phase = -np.arctan(np.cos(beta)/np.tan(rho - theta))
    c = -np.sin(beta)*np.sin(phi)*np.cos(rho-theta)
    return mag, phase, c

def I_fast(gamma, mag, phase, c):
    return np.maximum(mag*np.sin(gamma+phase) + c, 0)

def Q_day(t, beta, rho, eps):
    r, theta = polar_pos(eps, t)
    mag, phase, c = trig_coefs(beta, rho, theta)
    I_day =  quad_vec(lambda gamma: I_fast(gamma, mag, phase, c),
                0,2*np.pi,epsrel=10)[0]
    Q_day = K/(8*np.pi**2*r**2)*I_day
    #return interp1d(np.sin(phi), Q_day, 'cubic')
    return Q_day

def calc_E(M, eps):
    # Single iteration of newton method is sufficient
    E = M + eps*np.sin(M)/(1-eps*np.cos(M))
    return E

def calc_theta(E, eps):
    sign = 1-2*(E>np.pi)
    return np.pi+2*sign*np.arctan(np.sqrt((1+eps)/(1-eps)*(np.tan(E/2))**2))

def polar_pos(eps, t):
    """ Using Kepler's law:
    https://en.wikipedia.org/wiki/Kepler%27s_laws_of_planetary_motion#Position_as_a_function_of_time
    """
    t = (t+year/2)%year # Equations assume theta=0 at perihelion 
    M = t*2*np.pi/year
    E = calc_E(M, eps)
    r = au*(1 - eps*np.cos(E))
    theta = calc_theta(E,eps)
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
