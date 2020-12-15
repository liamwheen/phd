#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from matplotlib.animation import FuncAnimation
from data import milanko_params # Values given in 1000 year time steps

year2sec = 3.154e+7 #Translate time dependent units to 'per year' instead of 'per second'

alpha_1 = 0.32
alpha_2 = 0.62
A = 202 #Wm^-2
B = 1.9 #Wm^-2
C = 3.04 #Wm^-2 K^-1
T_ice = -10 #degC
R = 4*10**8 #some say e9 some say e8 #J m^-2 K^-1
S = 2.5*10**12
Q_0 = 343 #Wm^-2
        
#n0 = 0.2487 # Unstable equilibrium initial iceline
eta0 = 0.5#0.49 # Initial Iceline
tmin = 0
tmax = 40000 # Years

num_steps = 10000
frame_refr = 50
        
class Budyko:
    def __init__(self):
        self.y_span = np.linspace(0,1,1000)
        self.y_delta = self.y_span[1]
        self.t_span = np.linspace(0,tmax*year2sec,num_steps) #years
        self.T_record = np.zeros((self.t_span.size,self.y_span.size)) 
        milanko_t, milanko_ecc, milanko_obliq, milanko_l_peri = milanko_params.load_milanko('forward')
        krange = range(tmin//1000,2+max(1,tmax//1000))
        self.eps_func = interp1d(milanko_t[krange], milanko_ecc[krange])
        self.beta_func = interp1d(milanko_t[krange], milanko_obliq[krange])
        self.l_peri_func = interp1d(milanko_t[krange], milanko_l_peri[krange])
        self.c_b = (5/16)*(3*np.sin(0.4091)**2 - 2)#(5/16)*(3*np.sin(self.beta_func(0))**2 - 2)
        self.delta = self.t_span[1]
        self.eta = eta0 #n initial
        self.milanko_update(0)
        self.T = self.T_star(self.y_span) #set temp profile to eq profile
        #self.T = np.zeros(self.y_span.size)
        self.T_eta = self.T_y(self.T_star(self.y_span), self.eta)

    def s_b(self, y):
        return 1 + 0.5*self.c_b*(3*y**2 - 1)

    def a_eta(self, y):
        # Account for fp errors in ice line positions
        round_eta = self.y_delta*np.round(self.eta/self.y_delta) 
        round_y = self.y_delta*np.round(y/self.y_delta) 
        return ((round_y!=round_eta)*0.5+0.5)*((round_y<=round_eta)*alpha_1+(round_y>=round_eta)*alpha_2)
        # Using Widiasih's smooth a_eta
        #M = 100
        #return 0.47 + 0.15 * (np.tanh(M*(y - self.eta)))

    def int_T(self, T):
        if isinstance(T,float):
            # Tnj passed so will integrate over T_star
            return np.trapz(self.T_star(self.y_span), self.y_span)

        return np.trapz(self.T,self.y_span)

    def T_star(self, y):
        In = np.sum(self.s_b(y)*(1-self.a_eta(y)))*self.y_delta
        T_bar = (self.Q_e*In - A)/B
        T_star = (self.Q_e*self.s_b(y)*(1 - self.a_eta(y)) - A + C*T_bar)/(B + C)
        return T_star

    def dT_dt(self, T, y):
        f = self.Q_e*self.s_b(y)*(1 - self.a_eta(y)) - (A + B*T) - C*(T - self.int_T(T))
        return f/R

    def T_y(self, T, y):
        shift = -1*(y>0.5)
        ind = np.clip(np.round(y*T.size),0,T.size-1).astype(int) + shift
        return T[ind]

    def milanko_update(self, t):
        self.eps = float(self.eps_func(t/year2sec/1000))
        self.beta = float(self.beta_func(t/year2sec/1000))
        self.l_peri = float(self.l_peri_func(t/year2sec/1000))
        # Here we shift from (vernal eq to perihelion) to (aphelion (x-axis) to
        # north pole direction in ecliptic plane)
        self.rho = ((1/2)*np.pi - self.l_peri)%(2*np.pi) - np.pi
        self.Q_e = Q_0/(np.sqrt(1-self.eps**2))
     
    def iter_func(self):
        #Iter over all time points
        for frame, t in enumerate(self.t_span):
            self.milanko_update(t)
            self.T_record[frame,:] = self.T
            self.T += self.delta*self.dT_dt(self.T, self.y_span)
            self.T_eta += self.delta*self.dT_dt(self.T_eta, self.eta)
            self.eta = np.clip(self.eta + self.delta*(self.T_eta - T_ice)/S,0,1)
            if frame%frame_refr==0: 
                print(self.eta)
                yield t

    def update(self, t):
        self.ax.set_ylabel('$T(y$, {:.1e} years) (°C)'.format(t/year2sec))
        self.temp.set_xdata(self.y_span)
        self.temp.set_ydata(self.T)
        self.ice.set_xdata(self.eta)
        #self.T_star_eta.set_xdata(self.eta)
        #self.T_star_eta.set_ydata(self.T_eta)
        #self.equil_T.set_xdata(self.y_span)
        #self.equil_T.set_ydata(self.T_star(self.y_span))
        #self.ax.set_ylim(min(self.T),max(self.T))
        self.ax.set_ylim(-50,50)
        self.ice.set_ydata(self.ax.get_ylim())
        #self.grad.set_xdata(self.y_span[1:])
        #self.grad.set_ydata(10000*np.diff(self.T))

    def animate(self):
        self.fig, self.ax = plt.subplots()
        self.ydata, self.Tdata = [], []
        self.temp, = self.ax.plot([], [], 'r-', label='Temperature Profile',linewidth=3)
        self.ice, = self.ax.plot([], [], 'b-', label='sin(iceline)',linewidth=3)
        #self.T_star_eta, = self.ax.plot([], [], 'ro', label='Temperature at Iceline')
        #self.equil_T, = self.ax.plot([], [], 'm-', label='Equilibrium Temperature Profile')
        #self.grad, = self.ax.plot([], [], 'k', linewidth=0.5, label='Gradient (scaled)')
        self.ax.set_xlim(self.y_span[0], self.y_span[-1])
        self.ax.set_xlabel('y = sin(latitude)')
        self.fig.legend(framealpha=1)
        ani = FuncAnimation(self.fig, self.update, frames=self.iter_func, interval=1, repeat=False)
        plt.show()

model = Budyko()
model.animate()
plt.show()
