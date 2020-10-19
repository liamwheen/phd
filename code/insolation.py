import numpy as np
import milanko_params
from scipy.interpolate import interp1d
from scipy.optimize import root
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

"""
https://en.wikipedia.org/wiki/Solar_constant#Relationship_to_other_measurements
http://lasp.colorado.edu/lisird/ #This is specific data, but not really worth while
"""

k2day = 365250
au = 149597870700 # metres
q = 152096508529**2*1.321e3 # scaled irradiance const such that this over R**2 is irradiance at atmosphere
tmax = 352 # days
num_steps = 1*10**3
frame_refr = num_steps//(1*10**2)

# Interpolate milankovitch data to fit timescale
# Assumes milanko is positive time datta
eps_func = interp1d(milanko_params.t[:1+max(1,tmax//k2day)], milanko_params.ecc[:1+max(1,tmax//k2day)])
beta_func = interp1d(milanko_params.t[:1+max(1,tmax//k2day)], milanko_params.obliq[:1+max(1,tmax//k2day)])
l_peri_func = interp1d(milanko_params.t[:1+max(1,tmax//k2day)], milanko_params.l_peri[:1+max(1,tmax//k2day)])

class Insolation:

    def __init__(self):
        self.t_span = np.linspace(0,tmax,num_steps)
        self.insol_vals = []
        self.milanko_update(0)
        self.pos = self.polar_pos(0)

    def milanko_update(self, t):
        self.eps = eps_func(t/k2day/1000)
        self.beta = beta_func(t/k2day/1000)
        self.l_peri = l_peri_func(t/k2day/1000)
        self.rho = (3/2)*np.pi - self.l_peri
        self.a = self.ellipse_axes(self.eps)[0]
        self.insol = q/self.polar_pos(t)[0]**2
        return self.insol

    def I_latlon(self, lat, lon, t):
        """ Daily average insolation recieved at (lat,lon) on Earth on day 't'"""
        lat, lon = lat*np.pi/180, lon*np.pi/180
        earth_unit = self.pol2cart(1,self.polar_pos(t)[1]) 
        latlon_unit = self.latlon2unit(lat, lon)
        theta = self.polar_pos(t)[1]
        #I = -self.insol*(np.cos(lat)*(np.cos(self.beta)*np.cos(theta-self.rho)*np.cos(lon)\
        #    + np.sin(theta-self.rho)*np.sin(lon)) +\
        #    np.sin(lat)*np.sin(self.beta)*np.cos(theta-self.rho))
        I = -self.insol*earth_unit.dot(self.U_rho().dot(self.U_beta().dot(latlon_unit)))
        return I
    
    def latlon2unit(self, lat, lon):
        """ Turn lat/lon coords into unit vector with Earth's centre as origin"""
        return np.array([np.cos(lat)*np.cos(lon), 
                         np.cos(lat)*np.sin(lon), 
                         np.sin(lat)            ])

    def U_beta(self):
        b = self.beta
        return np.array([[np.cos(b) , 0, np.sin(b)],
                         [0            , 1, 0           ],
                         [-np.sin(b), 0, np.cos(b)]])

    def U_rho(self):
        p = self.rho
        return np.array([[np.cos(p) , -np.sin(p), 0],
                         [np.sin(p) , np.cos(p) , 0],
                         [0           , 0           , 1]])

    def midpoint_E(self, M, eps):
        E_func = lambda E: E - eps*np.sin(E) - M
        mid_E = root(E_func, M)
        return mid_E.x[0]

    def theta(self, E, eps):
        sign = 1 if E < np.pi else -1 
        return np.pi+sign*2*np.arctan(np.sqrt((1+eps)/(1-eps)*(np.tan(E/2))**2))

    def polar_pos(self, t):
        t = (t+182.625)%365.25 
        M = t*2*np.pi/365.25
        E = self.midpoint_E(M, self.eps)
        theta = self.theta(E, self.eps)
        r = self.a*(1 - self.eps*np.cos(E))
        return r, theta

    def pol2cart(self, r, theta):
        return np.array([r*np.cos(theta), r*np.sin(theta), 0])

    def ellipse_axes(self, eps):
        """ Estimates ellipse min and major axes using assumption that the
        perimeter remains the same all the time, Ramanujan approx used"""
        eps0 = eps_func(0)
        a0 = au
        b0 = a0*np.sqrt(1-eps0**2)
        p = self.ellipse_perim(a0,b0)
        a,b = self.midpoint_a_b(p, eps)
        return a, b

    def ellipse_perim(self, a,b):
        """ Gives Ramanujan apprximation to ellipse perimeter given
        semi-major/minor axes"""
        h = (a-b)**2/(a+b)**2
        p = np.pi*(a+b)*(1 + 3*h/(10 + np.sqrt(4 - 3*h)))
        return p

    def midpoint_a_b(self, p, eps):
        """ Solves for a and b using eccentricity equation and perimeter
        equation which is assumed to be constant"""
        ab_func = lambda ab: [np.sqrt(1-(ab[1]/ab[0])**2)-eps, self.ellipse_perim(*ab)-p]
        mid_ab = root(ab_func, [au,au*np.sqrt(1-eps**2)])
        return mid_ab.x

    def iter_func(self):
        for frame, t in enumerate(self.t_span):
            self.milanko_update(t)
            self.pos = self.polar_pos(t)
            self.insol_vals.append([self.I_latlon(50,i,t) for i in
                range(180,540,10)])
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
        lon0x, lon0y, _ = self.U_rho().dot(self.U_beta().dot(latlon_unit))
        self.latlon0.set_xdata([earthx,earthx+3e11*lon0x])
        self.latlon0.set_ydata([earthy,earthy+3e11*lon0y])

    def animate(self):
        self.fig, self.ax = plt.subplots()
        self.ellipse, = plt.plot([],[],'m--',linewidth=0.5)
        self.earth, = plt.plot([],[],'co')
        self.latlon0, = plt.plot([],[],'b')
        self.sun, = plt.plot([],[],'yo',linewidth=4)
        self.sun.set_ydata(0)
        self.sun.set_xdata(0)
        self.ax.set_xlim([-1.5*au,1.5*au])
        self.ax.set_ylim([-1.5*au,1.5*au])
        self.ax.set_aspect('equal')
        ani = FuncAnimation(self.fig, self.update,
                frames=self.iter_func, interval=1, repeat=False)
        plt.show()

model = Insolation()
model.animate()
plt.figure()
plt.plot(model.t_span,model.insol_vals[1:])
plt.show()
