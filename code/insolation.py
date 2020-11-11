import numpy as np
from data import milanko_params
from scipy.interpolate import interp1d
from scipy.optimize import root
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from insol_sympy import calc_daily_average

k2day = 365250
au = 149597870700 # metres
# scaled irradiance const such that this over R**2 is irradiance at atmosphere
# using aphelion as 1.52e11 and irradiance at equator atmosphere as 1.321e3
# min irradiance val taken from https://en.wikipedia.org/wiki/Solar_constant#Relationship_to_other_measurements
#q = 152096508529**2*1.321e3
q = 3.86e+26/(4*np.pi) #Suns total irradiance over 4pi, needs dividing by r^2


class Insolation:

    def __init__(self, tmin=0, tmax=100*k2day, milanko_direction='forward'):
        # Interpolate milankovitch data to fit timescale
        # Loads milanko data for future
        milanko_t, milanko_ecc, milanko_obliq, milanko_l_peri = milanko_params.load_milanko(milanko_direction)
        krange = range(tmin//k2day,2+max(1,tmax//k2day))
        self.eps_func = interp1d(milanko_t[krange], milanko_ecc[krange])
        self.beta_func = interp1d(milanko_t[krange], milanko_obliq[krange])
        self.l_peri_func = interp1d(milanko_t[krange], milanko_l_peri[krange])
        self.milanko_update(tmin)
        # Using approximation that max axis remains constant as shown in Laskar '04 pg273
        self.a = au

    def milanko_update(self, t):
        self.eps = float(self.eps_func(t/k2day))
        self.beta = float(self.beta_func(t/k2day))
        self.l_peri = float(self.l_peri_func(t/k2day))
        # Here we shift from (vernal eq to perihelion) to (aphelion (x-axis) to
        # north pole direction in ecliptic plane)
        self.rho = ((1/2)*np.pi - self.l_peri)%(2*np.pi) - np.pi

    def I_lat_ave(self, lats, t):
        """ Daily average insolation recieved at lat on Earth on day 't'"""
        lats = np.array(lats)*np.pi/180
        theta = self.polar_pos(t)[1]
        insol_ave = calc_daily_average(self.rho, self.beta,
                theta, lats, self.eps)
        return insol_ave

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

    def last_sum_solst(self, t):
        """ Tracks back over the past year to find the day of the summer solstice"""
        for t_summer in np.linspace(t,t-365,366):
            theta = self.polar_pos(t_summer)[1]
            if abs((np.pi-self.rho) - (2*np.pi-theta)) < 0.02:
                return t_summer

    def update(self, t, lats):
        self.milanko_update(t)
        #t = self.last_sum_solst(t)
        self.insol = q/self.polar_pos(t)[0]**2
        return self.I_lat_ave(lats,t)

if __name__ =="__main__":
    tmin = -365.25
    tmax = 0
    num_steps = 1000
    t_span = np.linspace(tmin,tmax,num_steps)
    model = Insolation(tmin, tmax, 'backward')
    insol_vals = np.array([[None]*181]*num_steps)
    for i, t in enumerate(t_span):
        insol_vals[i,:] = model.update(t,np.linspace(-90,90,91))

    #np.savetxt('insol_vals.csv',insol_vals,delimiter=',')
