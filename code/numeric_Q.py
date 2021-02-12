"""
Calculate Qyear from paper but numerically rather than analytically. This
allows for albedo function with SZA dependance to be incorporated. It might
also be used in the sub-year resolution model."""
import numpy as np
from scipy.optimize import root
from insol_sympy import calc_yearly_average
from matplotlib import pyplot as plt

def cartesian_product(a, b):
    arr = np.empty((phi_n, gamma_n, 2))
    for i, val in enumerate(np.ix_(a,b)):
        arr[...,i] = val
    return arr.reshape(-1,2).T

K = 3.8284e+26
au = 149597870700
year = 365.2425
beta = 0.4090
eps = 0.0167
rho = 2.9101
phi_n, gamma_n = 100, 100
phi = np.arcsin(np.linspace(-1,1,phi_n))
# Longitudinal symmetry means we can save time just looking at one hemisphere
gamma = np.linspace(0, np.pi,gamma_n) 

rep_phi, rep_gamma = cartesian_product(phi,gamma)

# P and alpha for Asin(x)+Bcos(x) formula
P = np.sqrt(2*np.sin(beta)*np.sin(rep_phi)*np.cos(beta)*np.cos(rep_gamma)*np.cos(rep_phi) +
        np.cos(beta)**2*np.cos(rep_gamma)**2*np.cos(rep_phi)**2 +
        np.cos(beta)**2*np.cos(rep_phi)**2 - np.cos(beta)**2 -
        np.cos(rep_gamma)**2*np.cos(rep_phi)**2 + 1) 
alph = np.arctan2((-np.sin(beta)*np.sin(rep_phi)*np.cos(rho) +
    np.sin(rep_gamma)*np.sin(rho)*np.cos(rep_phi) -
    np.cos(beta)*np.cos(rep_gamma)*np.cos(rep_phi)*np.cos(rho)),(-np.sin(beta)*np.sin(rep_phi)*np.sin(rho)
        - np.sin(rep_gamma)*np.cos(rep_phi)*np.cos(rho) -
        np.sin(rho)*np.cos(beta)*np.cos(rep_gamma)*np.cos(rep_phi)))

def main():
    num_Q = Q_year(rep_phi,rep_gamma)
    ana_Q = calc_yearly_average(beta, np.arcsin(np.linspace(-1,1,phi_n)),
            eps)#*(1-0.32) # Average albedo applied over entire Earth
    #plt.plot(np.linspace(-90,90,phi_n), num_Q)
    #plt.plot(np.linspace(-90,90,phi_n), ana_Q)
    #plt.show()
    print('Mean Error: ',np.mean((abs(num_Q-ana_Q)/ana_Q)))

def add_albedo(I):
    """ Using fit curve of ocean albedo from
    https://agupubs.onlinelibrary.wiley.com/doi/10.1029/2004GL021180"""
    # I is the same as cos(sza)
    #I[I==0]=np.nan
    a,b,c = np.array([0.296875, -0.59375, 0.326875])*2.5 # Account for clouds (mean=0.32 still)
    albedo = a*I**2 + b*I+ c
    return albedo

def I_fast(theta):
    return P*np.sin(theta+alph)

def I_t(t):
    r, theta = polar_pos(eps, np.array([t]))
    I = I_fast(theta)
    I[I<0] = 0
    I = I.reshape(phi_n,gamma_n)
    return I

def Q_year(phi, gamma):
    t_span = np.linspace(0,year,1000)
    Is = np.zeros(phi_n*gamma_n)
    rs, thetas = polar_pos(eps, t_span)
    for r, theta in zip(rs,thetas):
        I = I_fast(theta)
        I[I<0] = 0
        #I*=1-add_albedo(I)
        Is+=I/r**2
    Is = Is.reshape(phi_n,gamma_n)/len(t_span)
    return K/(4*np.pi)*np.mean(Is,1)

def midpoint_E(M, eps):
    E_func = lambda E: E - eps*np.sin(E) - M
    mid_E = root(E_func, M)
    return mid_E.x[0]

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

def anim_main():
    import matplotlib.animation as animation
    def anim(I):
        im.set_data(np.flipud(I))
    def frames():
        for t in np.linspace(0,year,300):
            yield I_t(t)
    fig, ax = plt.subplots()
    im = plt.imshow(np.zeros((len(phi),len(gamma))),vmin=0, vmax=1)
    ani = animation.FuncAnimation(fig,anim,frames=frames,interval=1,repeat=True)
    plt.show()


if __name__ == '__main__':
    #anim_main()
    import cProfile, pstats
    profiler = cProfile.Profile()
    profiler.enable()
    main()
    profiler.disable()
    stats = pstats.Stats(profiler).sort_stats('cumulative')
    stats.print_stats()
