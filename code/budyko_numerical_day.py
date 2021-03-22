#!/usr/bin/env python3
"""
Budyko model with milankovitch parameters and numerical Q_year implementation,
useful for adding albedo SZA dependance."""
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from matplotlib.animation import FuncAnimation
from data import milanko_params # Values given in 1000 year time steps
from numeric_Q_day import Q_day

year2sec = 31556952 #Translate time dependent units to 'per year' instead of 'per second'
year = 365.2425

alpha_1 = 0.32#0.279
alpha_2 = 0.62
A = 202.1#Wm^-2
B = 1.9 #Wm^-2
C = 3.04 #Wm^-2 K^-1
T_ice = -10 #degC
R = 2*10**7 #J m^-2 K^-1 (Temp damping)
S = 2*10**9 # degC s (Ice line damping)
Q_0 = 340.327 #Wm^-2
        
eta0 = 0.63 # Initial Iceline
tmin = -60
tmax = 0 # Years
year_res = 50 # Points per year
year_span = np.linspace(0,year,year_res)[:-1] # Avoid repeating last point as first point

y_steps = 1000
t_steps = (tmax-tmin)*year_res
frame_refr = 1
        
def main():
    model = Budyko()
    list(model.iter_func())
    #plt.plot(model.eta_record)
    #plt.show()
    #np.savetxt('budyko_numerical_day_eta.csv',model.eta_record,delimiter=',')

def anim_main():
    model = Budyko()
    model.animate()
    model.eta_record[model.eta_record==0] = np.nan
    print(np.nanmean(model.eta_record))

class Budyko:
    def __init__(self):
        self.y_span = np.linspace(0,1,y_steps)
        self.y_delta = self.y_span[1] - self.y_span[0]
        self.t_span = np.linspace(tmin*year2sec,tmax*year2sec,t_steps) #years
        self.delta = self.t_span[1] - self.t_span[0]
        self.eta_record = np.zeros(t_steps//frame_refr) 
        milanko_t, milanko_ecc, milanko_obliq, milanko_l_peri = milanko_params.load_milanko('backward')
        krange = range(int(tmin//1000),2+max(1,int(tmax//1000)))
        self.eps_func = interp1d(milanko_t[krange], milanko_ecc[krange])
        self.beta_func = interp1d(milanko_t[krange], milanko_obliq[krange])
        self.l_peri_func = interp1d(milanko_t[krange], milanko_l_peri[krange])
        self.eta = eta0 #n initial
        self.milanko_update(tmin*year2sec)
        self.T = np.zeros(self.y_span.size)

    def a_eta(self, y):
        # Account for fp errors in ice line positions
        #round_eta = self.y_delta*np.round(self.eta/self.y_delta) 
        #round_y = self.y_delta*np.round(y/self.y_delta) 
        #return ((round_y<=round_eta)*alpha_1+(round_y>round_eta)*alpha_2)
        M = 200
        return 0.47 + 0.15 * (np.tanh(M*(y - self.eta)))

    def int_T(self, T):
        return np.trapz(self.T,self.y_span)

    def dX_dt(self, X):
        y = self.y_span
        T, eta = X[:-1],X[-1]
        dT = self.Qs(y)*(1-self.a_eta(y)) - (A + B*T) - C*(T - self.int_T(T))
        T_eta = T[self.y_ind(eta)]
        deta = T_eta - T_ice
        return np.append(dT/R, deta/S)

    def y_ind(self, y):
        if isinstance(y,float):
            y_ind = int(y*y_steps)
            return min(y_steps-1,y_ind)
        y_ind = (y*y_steps).astype(int)
        y_ind[y_ind>(y_steps-1)] = y_steps-1
        return y_ind

    def milanko_update(self, t):
        eps = float(self.eps_func(t/year2sec/1000))
        beta = float(self.beta_func(t/year2sec/1000))
        l_peri = float(self.l_peri_func(t/year2sec/1000))
        rho = (3/2*np.pi - l_peri)%(2*np.pi)
        return eps, beta, rho

    def RK4(self, X0, ddt, h):
        "Apply Runge Kutta Formulas to find next values of T and eta"
        y = self.y_span
        k1 = ddt(X0)
        k2 = ddt(X0 + h*k1/2)
        k3 = ddt(X0 + h*k2/2)
        k4 = ddt(X0 + h*k3)
        return X0 + h/6*(k1 + 2*k2 + 2*k3 + k4)

    def iter_func(self):
        #Iter over all time points
        for frame, t in enumerate(self.t_span):
            if frame%year_res==0:
                if frame%(t_steps//100)==0:print(frame/t_steps)
                eps, beta, rho = self.milanko_update(t)
            day_of_year = (frame%year_res)/year_res*year
            self.Qs = Q_day(day_of_year, beta, rho, eps)
            X = self.RK4(np.append(self.T,self.eta), self.dX_dt, self.delta)
            self.T, self.eta = X[:-1],X[-1]
            self.eta_record[frame//frame_refr] = self.eta
            if frame%frame_refr==0:
                yield t

    def update(self, t):
        self.ax.set_ylabel('$T(y$, {:.1e} years) (°C)'.format(t/year2sec))
        self.temp.set_xdata(self.y_span)
        self.temp.set_ydata(self.T)
        self.ice.set_xdata(self.eta)
        self.ax.set_ylim(-50,50)
        self.ice.set_ydata(self.ax.get_ylim())

    def animate(self):
        self.fig, self.ax = plt.subplots()
        self.ydata, self.Tdata = [], []
        self.temp, = self.ax.plot([], [], 'r-', label='Temperature Profile',linewidth=3)
        self.ice, = self.ax.plot([], [], 'b-', label='sin(iceline)',linewidth=3)
        self.ax.set_xlim(self.y_span[0], self.y_span[-1])
        self.ax.set_xlabel('y = sin(latitude)')
        self.fig.legend(framealpha=1)
        ani = FuncAnimation(self.fig, self.update, frames=self.iter_func, interval=1, repeat=False)
        plt.show()

if __name__ == '__main__':
    anim_main()
    """
    import cProfile, pstats
    profiler = cProfile.Profile()
    profiler.enable()
    main()
    profiler.disable()
    stats = pstats.Stats(profiler).sort_stats('cumulative')
    stats.print_stats()
    """

