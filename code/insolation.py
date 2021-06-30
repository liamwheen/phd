import numpy as np
from data import milanko_params
from scipy.interpolate import interp1d
from scipy.optimize import root
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from insol_sympy import calc_daily_average

au = 149597870700 # metres
year = 365.2425

class Insolation:

    def __init__(self, milanko_direction='forward', north_solst=True):
        self.north_solst = north_solst
        # Interpolate milankovitch data to fit timescale
        milanko_t, milanko_ecc, milanko_obliq, milanko_l_peri = milanko_params.load_milanko(milanko_direction)
        self.eps_func = interp1d(milanko_t, milanko_ecc)
        self.beta_func = interp1d(milanko_t, milanko_obliq)
        self.l_peri_func = interp1d(milanko_t, milanko_l_peri)

    def milanko_update(self, t):
        self.eps = float(self.eps_func(t))
        self.beta = float(self.beta_func(t))
        self.l_peri = float(self.l_peri_func(t))
        # Here we shift from (vernal eq to perihelion) to (aphelion (x-axis) to
        # north pole direction in ecliptic plane)
        self.rho = (3/2*np.pi - self.l_peri)%(2*np.pi)

    def I_lat_ave(self, lats, t):
        """ Daily average insolation recieved at lats on Earth on day 't'"""
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
        return np.pi+sign*2*np.arctan2(np.sqrt(((1+eps)*(np.tan(E/2))**2)),np.sqrt((1-eps)))

    def polar_pos(self, t):
        """ Using Kepler's law:
        https://en.wikipedia.org/wiki/Kepler%27s_laws_of_planetary_motion#Position_as_a_function_of_time
        """
        t = (t+year/2)%year
        M = t*2*np.pi/year
        E = self.midpoint_E(M, self.eps)
        theta = self.theta(E, self.eps)
        r = au*(1 - self.eps*np.cos(E))
        return r, theta

    def last_sum_solstice(self):
        theta = (self.rho-self.north_solst*np.pi)%(2*np.pi)
        E = (np.pi+2*np.arctan(np.sqrt((1-self.eps)/(1+self.eps))*np.tan((theta)/2)))%(2*np.pi)
        M = E - self.eps*np.sin(E)
        days_since_aph = year/2+M*year/(2*np.pi)
        return days_since_aph

    def update(self, t, lats):
        self.milanko_update(t)
        days_since_aph = self.last_sum_solstice()
        return self.I_lat_ave(lats,days_since_aph), days_since_aph

    def run_insol(self, lat_vals, tmin=-400, tmax=0):
        num_steps = int(10*(tmax-tmin))
        t_span = np.linspace(tmin,tmax,num_steps)
        insol_vals = np.zeros((num_steps,len(lat_vals)))
        for i, t in enumerate(t_span):
            insol_vals[i],_ = model.update(t,lat_vals)
        return insol_vals


if __name__ =="__main__":
    import sys
    if len(sys.argv) == 1:
        tmin = -400
        tmax = 0
    else:
        tmin = int(sys.argv[1])
        tmax = int(sys.argv[2])
    if len(sys.argv) == 4:
        north_solst = True if 'n' in sys.argv[3] else False
        model = Insolation('backward', north_solst=north_solst)
        lat_vals = np.linspace(0,180*north_solst-90,10)
    else:
        model = Insolation('backward', north_solst=True)
        lat_vals = np.linspace(0,90,10)

    insol_vals = model.run_insol(lat_vals, tmin, tmax)
    np.savetxt('insol_vals.csv',insol_vals,delimiter=',')
    
