import numpy as np
from data import milanko_params
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from insol_sympy import calc_daily_average

au = 149597870700 # metres
year = 365.2425

class Insolation:

    def __init__(self, milanko_direction='backward', north_solst=True):
        self.north_solst = north_solst
        # Interpolate milankovitch data to fit timescale
        milanko_t, milanko_ecc, milanko_obliq, milanko_l_peri = milanko_params.load_milanko(milanko_direction)
        self.eps_func = interp1d(milanko_t, milanko_ecc)
        self.beta_func = interp1d(milanko_t, milanko_obliq)
        self.l_peri_func = interp1d(milanko_t, milanko_l_peri)

    def milanko_update(self, t):
        eps = self.eps_func(t)
        beta = self.beta_func(t)
        l_peri = self.l_peri_func(t)
        # Here we shift from (vernal eq to perihelion) to (aphelion (x-axis) to
        # north pole direction in ecliptic plane)
        rho = (3/2*np.pi - l_peri)%(2*np.pi)
        return beta, rho, eps

    def I_lat_ave(self, lats, t, beta, rho, eps):
        """ Daily average insolation recieved at lats on Earth on day 't'"""
        lats = np.array(lats)*np.pi/180
        theta = self.polar_pos(np.tile(t,np.size(eps)//len(t)), eps)[1]
        insol_ave = calc_daily_average(rho, beta, theta, lats, eps)
        return insol_ave

    def E_midpoint(self, E, M, eps):
        return E - eps*np.sin(E) - M

    def calc_E(self, M, eps):
        E = M + eps*np.sin(M)/(1-eps*np.cos(M))
        return E

    def calc_theta(self, E, eps):
        sign = np.ones(np.shape(E))
        sign[(E>np.pi)] = -1
        return np.pi+2*sign*np.arctan(np.sqrt((1+eps)/(1-eps)*(np.tan(E/2))**2))

    def polar_pos(self, t, eps):
        """ Using Kepler's law:
        https://en.wikipedia.org/wiki/Kepler%27s_laws_of_planetary_motion#Position_as_a_function_of_time
        """
        t = (t+year/2)%year # Equations assume theta=0 at perihelion
        M = t*2*np.pi/year
        E = self.calc_E(M, eps)
        r = au*(1 - eps*np.cos(E))
        theta = self.calc_theta(E,eps)
        return r, theta

    def last_sum_solstice(self, rho, eps):
        theta = (rho-self.north_solst*np.pi)%(2*np.pi)
        E = (np.pi+2*np.arctan(np.sqrt((1-eps)/(1+eps))*np.tan((theta)/2)))%(2*np.pi)
        M = E - eps*np.sin(E)
        days_since_aph = year/2+M*year/(2*np.pi)
        return days_since_aph

    def insol_at_solstice(self, t, lats):
        beta, rho, eps = self.milanko_update(t)
        days_since_aph = self.last_sum_solstice(rho, eps)
        return self.I_lat_ave(lats,np.array([days_since_aph]),beta,rho,eps), days_since_aph
    
    def insol_over_year(self, t, lats, year_res=300):
        year_span = np.linspace(0,year,year_res+1)[:-1]
        beta, rho, eps = self.milanko_update(np.repeat(t,year_res))
        return self.I_lat_ave(lats, year_span, beta, rho, eps).reshape(len(t),year_res)

    def run_insol(self, lats, tmin=-400, tmax=0, num_steps=None):
        if not num_steps: num_steps = int(10*(tmax-tmin))
        t_span = np.linspace(tmin,tmax,num_steps)
        insol_vals = np.zeros((num_steps,len(lats)))
        for i, t in enumerate(t_span):
            insol_vals[i],_ = self.insol_at_solstice(t,lats)
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
        lats = np.linspace(0,180*north_solst-90,10)
    else:
        model = Insolation('backward', north_solst=True)
        lats = np.linspace(0,90,10)

    insol_vals = model.run_insol(lats, tmin, tmax)
    np.savetxt('insol_vals.csv',insol_vals,delimiter=',')
    
