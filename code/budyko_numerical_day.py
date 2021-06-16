#!/usr/bin/env python3
"""
Budyko model with milankovitch parameters and numerical Q_year implementation,
useful for adding albedo SZA dependance."""
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from matplotlib.animation import FuncAnimation
from data import milanko_params # Values given in 1000 year time steps
from numeric_Q_day import Q_day, phi_n

year2sec = 31556952 #Translate time dependent units to 'per year' instead of 'per second'
year = 365.2425

alpha_1 = 0.32#0.279
alpha_2 = 0.62
A = 202.1#Wm^-2
B = 1.9 #Wm^-2
C = 4.2 #Wm^-2 K^-1
T_ice = -10 #degC
R = 4.2e7 #J m^-2 K^-1 (Temp damping)
S = 1e9 # degC s (Ice line damping)
Q_0 = 340.327 #Wm^-2
year_res = 400 # Points per year
year_span = np.linspace(0,year,year_res+1)[:-1] # Avoid repeating last point as first point

y_steps = 400
frame_refr = year_res*500

eta0 = 0.97 # Initial Iceline
tmin = -400000
tmax = 0 # Years

def run_long_term(tmin=tmin, tmax=tmax, jump=500):
    model = Budyko(tmin=tmin, tmax=tmax)
    years = np.arange(tmin, tmax+1, jump)
    Ts = np.empty((len(years),year_res,y_steps))
    eta = np.empty((len(years),year_res))
    for i,year in enumerate(years):
        Ts[i,...], eta[i,:] = model.get_year_temp(year,10)
        print(year, end='\r')
    return Ts, eta

def main():
    model = Budyko()
    list(model.iter_func())
    plt.plot(model.T_record[:,0])
    plt.plot(model.T_record[:,-1])
    plt.figure()
    plt.plot(model.eta_record)
    plt.show()
    np.savetxt('budyko_numerical_day_T.csv',model.T_record,delimiter=',')
    np.savetxt('budyko_numerical_day_eta.csv',model.eta_record,delimiter=',')
    

def anim_main():
    model = Budyko()
    model.animate()

class Budyko:
    def __init__(self, eta0=eta0, tmin=tmin, tmax=tmax):
        self.eta = eta0 #n initial
        self.t_steps = (tmax-tmin)*year_res
        self.y_span = np.linspace(0,1,y_steps)
        self.y_delta = self.y_span[1] - self.y_span[0]
        self.t_span = np.linspace(tmin*year2sec,tmax*year2sec,self.t_steps) #years
        self.delta = self.t_span[1] - self.t_span[0]
        #self.T_record = np.zeros((self.t_steps//frame_refr,y_steps))
        #self.eta_record = np.zeros(self.t_steps//frame_refr)
        milanko_t, milanko_ecc, milanko_obliq, milanko_l_peri = milanko_params.load_milanko('backward')
        krange = range(int(tmin//1000),2+max(1,int(tmax//1000)))
        self.eps_func = interp1d(milanko_t[krange], milanko_ecc[krange])
        self.beta_func = interp1d(milanko_t[krange], milanko_obliq[krange])
        self.l_peri_func = interp1d(milanko_t[krange], milanko_l_peri[krange])
        self.T = np.zeros(y_steps)

    def a_eta(self, y):
        # Account for fp errors in ice line positions
        round_eta = self.y_delta*np.round(self.eta/self.y_delta)
        round_y = self.y_delta*np.round(y/self.y_delta)
        return ((round_y<=round_eta)*alpha_1+(round_y>round_eta)*alpha_2)
        #M = 200
        #return (alpha_1+alpha_2)/2 + (alpha_2-alpha_1)/2 * (np.tanh(M*(y - self.eta)))

    def int_T(self, T):
        return np.trapz(T,self.y_span)

    def dX_dt(self, T, eta):
        y = self.y_span
        dT = self.Qs*(1-self.a_eta(y)) - (A + B*T) - C*(T - self.int_T(T))
        T_eta = T[self.y_ind(eta)]
        deta = (T_eta - T_ice)**1
        #deta = (1+((T_eta-T_ice)>0))*(T_eta-T_ice)
        return dT/R, deta/S

    def y_ind(self, y):
        y_ind = int(y*y_steps)
        return min(y_steps-1,y_ind)

    def milanko_update(self, t):
        eps = float(self.eps_func(t/1000))
        beta = float(self.beta_func(t/1000))
        l_peri = float(self.l_peri_func(t/1000))
        rho = (3/2*np.pi - l_peri)%(2*np.pi)
        return eps, beta, rho

    def RK4(self, T, eta, ddt, h):
        "Apply Runge Kutta Formulas to find next values of T and ice"
        k1T, k1eta = ddt(T, eta)
        k2T, k2eta = ddt(T + h*k1T/2, eta + h*k1eta/2)
        k3T, k3eta = ddt(T + h*k2T/2, eta + h*k2eta/2)
        k4T, k4eta = ddt(T+ h*k3T, eta + h*k3eta)
        dT = h/6*(k1T + 2*k2T + 2*k3T + k4T)
        dT[[0,-1]] = dT[[1,-2]]
        deta = h/6*(k1eta + 2*k2eta + 2*k3eta + k4eta)
        return T+dT, eta+deta

    def euler(self, T, eta, ddt, h):
        dT, deta = ddt(T,eta)
        return T+h*dT, eta+h*deta

    def iter_func(self):
        #Iter over all time points
        T_year = np.empty((year_res, y_steps))
        for frame, t in enumerate(self.t_span):
            #if frame%(self.t_steps//100)==0:print(f'{100*frame//self.t_steps}%',end='\r')
            if frame%(100*year_res)==0:
                Q_year = self.get_Q_year(t//year2sec)
            self.Qs = Q_year[frame%year_res]
            self.T, eta = self.euler(self.T,self.eta, self.dX_dt, self.delta)
            self.eta = min(eta,1)
            #Take average temperature for first year in frame_refr period
            #if frame%frame_refr < year_res:
            #    T_year[frame%frame_refr] = self.T
            #    if frame%frame_refr == 0: 
            if frame%frame_refr == 0:
                #self.T_record[frame//frame_refr] = self.T
                #self.eta_record[frame//frame_refr] = self.eta
                yield t
    
    def get_year_temp(self, end_year, run_time=10):
        eta_year = np.empty(year_res)
        T_year = np.empty((year_res, y_steps))
        Q_year = self.get_Q_year(end_year)
        for i in range(year_res*run_time):
            self.Qs = Q_year[i%year_res,:]
            self.T, eta = self.euler(self.T, self.eta, self.dX_dt, self.delta)
            self.eta = min(eta,1)
            if i >= (year_res*(run_time-1)):
                eta_year[i%year_res] = self.eta
                T_year[i%year_res,:] = self.T
        return T_year, eta_year

    def get_Q_year(self, year):
        f_name = f'.Q_year_cache/{int(np.ptp(self.y_span))}_{year_res}_{y_steps}_{int(year)}'
        try:
            with open(f_name,'rb') as f:
                Q_year = np.fromfile(f).reshape(year_res,y_steps)
        except FileNotFoundError:
            with open(f_name,'wb') as f:
                eps, beta, rho = self.milanko_update(year)
                Q_year = Q_day(year_span, beta, rho, eps, np.arcsin(self.y_span)).T
                Q_year.tofile(f)
        return Q_year

    def update(self, t):
        self.ax.set_ylabel('$T(y$, {:.1e} years) (°C)'.format(t/year2sec))
        self.temp.set_xdata(self.y_span)
        self.temp.set_ydata(self.T)
        self.insol.set_xdata(self.y_span)
        self.insol.set_ydata(self.Qs/7-40)
        self.ice.set_xdata(self.eta)
        self.ax.set_ylim(-50,50)
        self.ice.set_ydata(self.ax.get_ylim())

    def animate(self):
        self.fig, self.ax = plt.subplots()
        plt.plot([0,1],2*[T_ice])
        self.ydata, self.Tdata = [], []
        self.temp, = self.ax.plot([], [], 'r-', label='Temperature Profile',linewidth=3)
        self.insol, = self.ax.plot([], [], 'orange', label='Insol',linewidth=2)
        self.ice, = self.ax.plot([], [], 'b-', label='sin(iceline)',linewidth=3)
        self.ax.set_xlim(self.y_span[0], self.y_span[-1])
        self.ax.set_xlabel('y = sin(latitude)')
        self.fig.legend(framealpha=1)
        ani = FuncAnimation(self.fig, self.update, frames=self.iter_func, interval=1, repeat=False)
        plt.show()

if __name__ == '__main__':
    anim_main()
    run_long_term()
    """
    import cProfile, pstats
    profiler = cProfile.Profile()
    profiler.enable()
    main()
    profiler.disable()
    stats = pstats.Stats(profiler).sort_stats('cumulative')
    stats.print_stats()
    """

