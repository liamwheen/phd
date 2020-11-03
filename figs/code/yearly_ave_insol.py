import sys
sys.path.append('../../code/')
sys.path.append('../../code/data/')
import numpy as np
import milanko_params
from scipy.interpolate import interp1d
from scipy.optimize import root
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import insol_sympy

"""
https://en.wikipedia.org/wiki/Solar_constant#Relationship_to_other_measurements
"""

k2day = 365250
au = 149597870700 # metres
q = 152096508529**2*1.321e3 # scaled irradiance const such that this over R**2 is irradiance at atmosphere

class Insolation:

    def __init__(self, tmin, tmax, milanko_direction='forward'):
        # Interpolate milankovitch data to fit timescale
        # Loads milanko data for future
        milanko_t, milanko_ecc, milanko_obliq, milanko_l_peri = milanko_params.load_milanko(milanko_direction)
        krange = range(tmin//k2day,2+max(1,tmax//k2day))
        self.eps_func = interp1d(milanko_t[krange], milanko_ecc[krange])
        self.beta_func = interp1d(milanko_t[krange], milanko_obliq[krange])
        self.l_peri_func = interp1d(milanko_t[krange], milanko_l_peri[krange])
        self.milanko_update(tmin)

    def milanko_update(self, t):
        self.eps = float(self.eps_func(t/k2day))
        self.beta = float(self.beta_func(t/k2day))
        self.l_peri = float(self.l_peri_func(t/k2day))
        # Here we shift from (vernal eq to perihelion) to (aphelion (x-axis) to summer eq)
        self.rho = ((1/2)*np.pi - self.l_peri)%(2*np.pi) - np.pi
        self.a = self.ellipse_axes(self.eps)[0]

    def I_lat_ave(self, lat, t):
        """ Daily average insolation recieved at lat on Earth on day 't'"""
        lat = lat*np.pi/180
        R  = self.rotate_mat(self.beta, self.rho)
        theta = self.polar_pos(t)[1]
        insol_ave = -self.insol*insol_sympy.daily_insol_ratio(self.rho, self.beta, theta, lat)
        return insol_ave

    def latlon2unit(self, lat, lon):
        """ Turn lat/lon coords into unit vector with Earth's centre as origin
            Gives in Earth based axes, not inertial axes"""
        return np.array([np.cos(lat)*np.cos(lon), 
                         np.cos(lat)*np.sin(lon), 
                         np.sin(lat)            ])

    def rotate_mat(self, b, p):
        """ Combined rotation matrix for Earth vectors to account for obliquity and precession"""
        Ub = np.array([[np.cos(b) , 0, np.sin(b)],
                       [0         , 1, 0        ],
                       [-np.sin(b), 0, np.cos(b)]])

        Up = np.array([[np.cos(p) , -np.sin(p), 0],
                       [np.sin(p) , np.cos(p) , 0],
                       [0         , 0         , 1]])

        return Up.dot(Ub)

    def midpoint_E(self, M, eps):
        E_func = lambda E: E - eps*np.sin(E) - M
        mid_E = root(E_func, M)
        return mid_E.x[0]

    def theta(self, E, eps):
        sign = 1 if E < np.pi else -1 
        return np.pi+sign*2*np.arctan(np.sqrt((1+eps)/(1-eps)*(np.tan(E/2))**2))

    def polar_pos(self, t):
        """ Using Kepler's law:
        https://en.wikipedia.org/wiki/Kepler%27s_laws_of_planetary_motion#Position_as_a_function_of_time
        """
        t = (t+182.625)%365.25 
        M = t*2*np.pi/365.25
        E = self.midpoint_E(M, self.eps)
        theta = self.theta(E, self.eps)
        r = self.a*(1 - self.eps*np.cos(E))
        return r, theta

    def pol2cart(self, r, theta):
        return np.array([r*np.cos(theta), r*np.sin(theta), 0])

    def ellipse_axes(self, eps):
        """ Estimates ellipse min axis using approximation that max axis
        remains constant as shown in Laskar '04 pg273"""
        ecc = self.eps
        a = au #Assumes the semimajor axis is 1au
        b = a*np.sqrt(1-ecc**2)
        return a, b

    def yearly_average(self, t):
        """ Take start time and run 365 day from then, return yearly average at
        each latitude"""
        res = 122
        insol = np.zeros((res,yearly_ave_insol.shape[1]))
        for i, day in enumerate(np.linspace(t,t+365.25,res)):
            self.insol = q/self.polar_pos(day)[0]**2
            insol[i,:] = [self.I_lat_ave(lat,day) for lat in
                    np.linspace(-90,90,insol.shape[1])]
        
        year_ave = np.sum(insol,0)/res
        return year_ave

    def update(self, t):
            self.milanko_update(t)
            self.insol = q/self.polar_pos(t)[0]**2
            return self.yearly_average(t)

if __name__ =="__main__":
    tmin = -150*k2day
    tmax = 0
    num_steps = 151
    t_span = np.linspace(tmin,tmax,num_steps)
    model = Insolation(tmin, tmax, 'backward')
    yearly_ave_insol = np.zeros((num_steps,91))
    for i, t in enumerate(t_span):
        yearly_ave_insol[i,:] = model.update(t)
        print(t)
        print(yearly_ave_insol[i,45])

    np.savetxt('yearly_insol_vals.csv',yearly_ave_insol,delimiter=',')
