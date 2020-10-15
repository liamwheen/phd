import numpy as np
import milanko_params
from scipy.interpolate import interp1d
from scipy.optimize import root
import matplotlib.pyplot as plt

"""
https://en.wikipedia.org/wiki/Solar_constant#Relationship_to_other_measurements
http://lasp.colorado.edu/lisird/ #This is specific data, but not really worth while
"""

day2sec = 86400
k2day = 365000
au = 149597870700 #metres
tmax = 1000 #days
S_0 = 1361 #W/m^2
#Q_0 = 343

# Interpolate milankovitch data to fit timescale
eps_func = interp1d(milanko_params.t[:1+max(1,tmax//k2day)], milanko_params.ecc[:1+max(1,tmax//k2day)])
beta_func = interp1d(milanko_params.t[:1+max(1,tmax//k2day)], milanko_params.obliq[:1+max(1,tmax//k2day)])
l_peri_func = interp1d(milanko_params.t[:1+max(1,tmax//k2day)], milanko_params.l_peri[:1+max(1,tmax//k2day)])
#Q_e = Q_0/(np.sqrt(1-eps_func(0)**2))
rho = (3/2)*np.pi - l_peri_func(0)

class Insolation:

    def milanko_update(self, t):
        self.eps = eps_func(t/day2sec/1000)
        self.beta = beta_func(t/day2sec/1000)
        self.l_peri = l_peri_func(t/day2sec/1000)
        #Q_e =  Q_0/(np.sqrt(1-eps**2))
        #self.c_b = (5/16)*(3*np.sin(beta)**2 - 2)
        self.rho = (3/2)*np.pi - self.l_peri
        self.insol = S_0*(1+0.034*np.cos(2*np.pi*t/day2sec/365.25))
        self.a = self.ellipse_axes(self.eps)[0]
        return self.insol

    def U_beta(self, beta):
        return np.array([[np.cos(beta) , 0, np.sin(beta)],
                         [0            , 1, 0           ],
                         [-np.sin(beta), 0, np.cos(beta)]])

    def U_rho(self, rho):
        return np.array([[np.cos(rho) , -np.sin(rho), 0],
                         [np.sin(rho) , np.cos(rho) , 0],
                         [0           , 0           , 1]])

    def midpoint_E(self, M, eps):
        E_func = lambda E: E - eps*np.sin(E) - M
        mid_E = root(E_func, M)
        return mid_E.x[0]

    def theta(self, E, eps):
        return 2*np.arctan(np.sqrt((1+eps)/(1-eps)*(np.tan(E/2))**2))

    def polar_pos(self, t):
        t = t%365.25
        M = t*2*np.pi/365.25
        E = self.midpoint_E(M, self.eps)
        theta = self.theta(E+np.pi, self.eps)
        r = self.a*(1 - self.eps*np.cos(E))
        return r, theta

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
        """ Solves for a and b using eccentricity equation and perimeter equation
        (assumed to be constant)"""
        ab_func = lambda ab: [np.sqrt(1-(ab[1]/ab[0])**2)-eps, self.ellipse_perim(*ab)-p]
        mid_ab = root(ab_func, [au,au*np.sqrt(1-eps**2)])
        return mid_ab.x

model = Insolation()

#print(model.ellipse_axes(0.0167))
model.milanko_update(0)
print(model.midpoint_a_b(model.ellipse_perim(au,au*np.sqrt(1-model.eps**2)),0.0167))
print(model.polar_pos(182.5))


#print(theta(midpoint_E(3*np.pi/2,0.0167),0.0167))
#print(3*np.pi/2)
#p_range = np.linspace(0,np.pi*2,100)
#plt.plot(p_range, [theta(midpoint_E(t,0.0167),0.0167) for t in p_range])
#plt.figure()
#plt.plot(np.linspace(0,365,1000), [1+0.034*np.cos(2*np.pi*t/365.25) for
#    t in np.linspace(0,365,1000)])
#plt.show()

#ratio = []
#for t in np.linspace(0,365*day2sec,365):
#    ratio.append(milanko_update(t))
#
#plt.plot(np.linspace(0,365,len(ratio)), ratio)
#plt.show()
