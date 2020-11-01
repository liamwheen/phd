import numpy as np
from data import milanko_params
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
tmin = 0#30*k2day#1015*k2day# how far in the future to start
tmax = tmin + 500*k2day# days
num_steps = 501
frame_refr = 1#num_steps//(1*10**2)

# Interpolate milankovitch data to fit timescale
# Loads milanko data for future
milanko_t, milanko_ecc, milanko_obliq, milanko_l_peri = milanko_params.load_forward_milanko()
krange = range(tmin//k2day,2+max(1,tmax//k2day))
eps_func = interp1d(milanko_t[krange], milanko_ecc[krange])
beta_func = interp1d(milanko_t[krange], milanko_obliq[krange])
l_peri_func = interp1d(milanko_t[krange], milanko_l_peri[krange])

class Insolation:

    def __init__(self):
        self.t_span = np.linspace(tmin,tmax,num_steps)
        self.insol_vals = np.array([[None]*2]*num_steps)
        self.milanko_update(tmin)
        self.pos = self.polar_pos(tmin)

    def milanko_update(self, t):
        self.eps = float(eps_func(t/k2day))
        self.beta = float(beta_func(t/k2day))
        self.l_peri = float(l_peri_func(t/k2day))
        # Here we shift from (vernal eq to perihelion) to (aphelion (x-axis) to summer eq)
        self.rho = ((1/2)*np.pi - self.l_peri)%(2*np.pi) - np.pi
        self.a = self.ellipse_axes(self.eps)[0]
        insol_sympy.set_milanko(self.rho, self.beta)

    def I_lat_ave(self, lat, t):
        """ Daily average insolation recieved at lat on Earth on day 't'"""
        lat = lat*np.pi/180
        R  = self.rotate_mat(self.beta, self.rho)
        theta = self.polar_pos(t)[1]
        point_on_circ = R.dot(self.latlon2unit(lat,0))
        insol_ave = -self.insol*insol_sympy.calculate_daily_insol(theta, lat, point_on_circ)/(2*np.pi)
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

    def last_sum_solst(self, t):
        """ Tracks back over the past year to find the day of the summer solstice"""
        for t_summer in np.linspace(t,t-365,366):
            theta = self.polar_pos(t_summer)[1]
            if abs((np.pi-self.rho) - (2*np.pi-theta)) < 0.02:
                return t_summer

    def iter_func(self):
        for frame, t in enumerate(self.t_span):
            self.milanko_update(t)
            t = self.last_sum_solst(t)
            self.insol = q/self.polar_pos(t)[0]**2
            self.pos = self.polar_pos(t)
            self.insol_vals[frame,:] = [self.I_lat_ave(lat,t) for lat in [0,65]]
            if frame%frame_refr==0 or t==self.t_span[-1]:
                yield t

    def update(self, t):
        self.ax.set_title('Days Since Aphelion: {:.1f}'.format(t%365.25))
        self.ax.set_ylabel('Semi-minor axis')
        self.ax.set_xlabel('Semi-major axis')
        theta = np.linspace(0, 2*np.pi, 100)
        a,b = self.ellipse_axes(self.eps)
        sun = np.sqrt(a**2-b**2)
        self.ellipse.set_xdata(sun+a*np.cos(theta))
        self.ellipse.set_ydata(b*np.sin(theta))
        earthx, earthy, _ = self.pol2cart(*self.polar_pos(t))
        self.earth.set_xdata(earthx)
        self.earth.set_ydata(earthy)
        latlon_unit = self.latlon2unit(0,0)
        lon0x, lon0y, _ = self.rotate_mat(self.beta, self.rho).dot(latlon_unit)
        self.latlon0.set_xdata([earthx,earthx+1e11*lon0x])
        self.latlon0.set_ydata([earthy,earthy+1e11*lon0y])
        self.insol_plot.set_xdata(np.linspace(tmin/365,tmax/365,len(self.insol_vals)))
        self.insol_plot.set_ydata(self.insol_vals[:,0])
        self.insol_plot2.set_xdata(np.linspace(tmin/365,tmax/356,len(self.insol_vals)))
        self.insol_plot2.set_ydata(self.insol_vals[:,1])
        self.insol_ax.set_xlim([tmin/365,max(t/365,tmin/365+1)])
        self.insol_ax.set_ylim([0,600])

    def init(self):
        self.ellipse, = self.ax.plot([],[],'m--',linewidth=0.5)
        self.earth, = self.ax.plot([],[],'co')
        self.latlon0, = self.ax.plot([],[],'b')
        self.insol_ax = self.fig.add_subplot(333)
        self.insol_ax.set_xlabel('Years')
        self.insol_ax.set_ylabel('Ave Insol')
        self.insol_ax.yaxis.tick_right()
        self.insol_plot, = self.insol_ax.plot([],[],'b')
        self.insol_plot2, = self.insol_ax.plot([],[],'r')
        self.ax.plot([0],[0],'yo',linewidth=4)
        self.ax.set_xlim([-1.5*au,1.5*au])
        self.ax.set_ylim([-1.5*au,1.5*au])
        self.ax.set_aspect('equal')
        self.insol_ax.set_ylim([0,600])
        return self.ellipse, self.earth, self.latlon0, self.insol_plot

    def animate(self):
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111)
        ani = FuncAnimation(self.fig, self.update,
                frames=self.iter_func, init_func=self.init, interval=1, repeat=False)
        plt.show()

model = Insolation()
model.animate()
