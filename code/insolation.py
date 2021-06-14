import numpy as np
from data import milanko_params
from scipy.interpolate import interp1d
from scipy.optimize import root
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from insol_sympy import calc_daily_average

au = 149597870700 # metres
year = 365.2425
k2day = year*1000

class Insolation:

    def __init__(self, tmin=0, tmax=100*k2day, milanko_direction='forward', north_solst=True):
        self.north_solst = north_solst
        # Interpolate milankovitch data to fit timescale
        milanko_t, milanko_ecc, milanko_obliq, milanko_l_peri = milanko_params.load_milanko(milanko_direction)
        krange = range(int(tmin//k2day),2+max(1,int(tmax//k2day)))
        self.eps_func = interp1d(milanko_t[krange], milanko_ecc[krange])
        self.beta_func = interp1d(milanko_t[krange], milanko_obliq[krange])
        self.l_peri_func = interp1d(milanko_t[krange], milanko_l_peri[krange])
        self.milanko_update(tmin)

    def milanko_update(self, t):
        self.eps = float(self.eps_func(t/k2day))
        self.beta = float(self.beta_func(t/k2day))
        self.l_peri = float(self.l_peri_func(t/k2day))
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
        t_sum = year/2+M*year/(2*np.pi)
        return t_sum

    def update(self, t, lats):
        self.milanko_update(t)
        t_sum = self.last_sum_solstice()
        return self.I_lat_ave(lats,t_sum), t_sum

if __name__ =="__main__":
    tmin = -400*k2day
    tmax = 0
    num_steps = 400
    t_span = np.linspace(tmin,tmax,num_steps)
    model = Insolation(tmin, tmax, 'backward', north_solst=True)
    lat_vals = np.linspace(0,-90,10)
    insol_vals = np.zeros((num_steps,len(lat_vals)))
    #r_vals = np.zeros(num_steps)
    for i, t in enumerate(t_span):
        print(i/len(t_span),end='\r')
        #insol_vals[i,:] = model.update(t,180/np.pi*np.arcsin(np.linspace(-1,1,insol_vals.shape[1])))
        insol_vals[i],_ = model.update(t,lat_vals)
        #r_vals[i] = q/model.polar_pos(t)[0]**2
    np.savetxt('insol_vals.csv',insol_vals,delimiter=',')
    
    #print(min(r_vals),max(r_vals))
    #print(np.mean(r_vals))
    #print(au)
    #print(np.trapz(np.trapz(insol_vals, t_span,
    #    axis=0)/365.25,np.sin(np.linspace(-np.pi/2,np.pi/2,181))/2))
    #plt.plot(np.linspace(0,365.25,500),np.mean(insol_vals,axis=1))
    #plt.plot(np.linspace(-np.pi/2,np.pi/2,181), np.trapz(insol_vals, t_span,
    #    axis=0)/365.25)
    #plt.show()
    #np.savetxt('insol_vals.csv',insol_vals,delimiter=',')
