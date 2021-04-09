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
C = 3.04 #Wm^-2 K^-1
D = 0.5 #Wm^-2 K^-1
T_ice = -10 #degC
T_ice_shift = np.array([-T_ice,T_ice])
R = 1*10**7 #J m^-2 K^-1 (Temp damping)
S = 4*10**8 # degC s (Ice line damping)
Q_0 = 340.327 #Wm^-2
        
etas0 = [-0.8,0.8] # Initial Icelines
tmin = -60000
tmax = 0 # Years
year_res = 1400 # Points per year
year_span = np.linspace(0,year,year_res+1)[:-1] # Avoid repeating last point as first point

y_steps = phi_n#50
t_steps = (tmax-tmin)*year_res
frame_refr = 50

dt = (tmax-tmin)*year2sec/(t_steps-1)
dx = 2/(y_steps-1)
k = R/(2*D)
#stability calculation doesnt seem to work: (dt/dx**2-k)/k<1 was the guess but
#not applicable for very different parameter values

def main():
    model = Budyko()
    list(model.iter_func())
    np.savetxt('budyko_numerical_day_eta.csv',model.eta_record,delimiter=',')

def anim_main():
    model = Budyko()
    model.animate()
    model.eta_record[model.eta_record==0] = np.nan
    print(np.nanmean(model.eta_record[:,1]))
    #print(max(model.max_T))
    #plt.plot(model.eta_record)
    #plt.show()

class Budyko:
    def __init__(self):
        self.y_span = np.linspace(-1,1,y_steps)
        self.y_delta = self.y_span[1] - self.y_span[0]
        self.t_span = np.linspace(tmin*year2sec,tmax*year2sec,t_steps) #years
        self.delta = self.t_span[1] - self.t_span[0]
        self.eta_record = np.zeros((t_steps//frame_refr,2)) 
        self.max_T = np.zeros(t_steps//frame_refr)
        milanko_t, milanko_ecc, milanko_obliq, milanko_l_peri = milanko_params.load_milanko('backward')
        krange = range(int(tmin//1000),2+max(1,int(tmax//1000)))
        self.eps_func = interp1d(milanko_t[krange], milanko_ecc[krange])
        self.beta_func = interp1d(milanko_t[krange], milanko_obliq[krange])
        self.l_peri_func = interp1d(milanko_t[krange], milanko_l_peri[krange])
        self.etas = np.array(etas0) #n initial
        self.milanko_update(tmin*year2sec)
        self.T = np.zeros(self.y_span.size)

    def a_eta(self, y):
        etaS, etaN = self.etas
        M = 200
        arg = (y>0)*(y - etaN) + (y<=0)*(-y + etaS)
        return (alpha_1+alpha_2)/2 + (alpha_2-alpha_1)/2 * (np.tanh(M*arg))

    def diffuse(self):
        dtdx = np.diff(self.T)/self.y_delta
        y_center = self.y_span[1:]-self.y_delta/2
        return D*np.pad(np.diff((1-y_center**2)*dtdx)/self.y_delta,1)
    
    def transport(self, T):
        return -C*(T - np.trapz(T,self.y_span)/2)

    def dX_dt(self, X):
        y = self.y_span
        T, eta = X[:-2],X[-2:]
        dT = self.Qs*(1-self.a_eta(y)) - (A + B*T) + D*self.diffuse()#self.transport(T)#
        dT[[0,-1]]=dT[[1,-2]]
        T_eta = T[self.y_ind(eta)]
        deta = np.array([-1,1])*T_eta - T_ice_shift
        #deta += (deta>0)*deta*2
        return np.append(dT/R, deta/S)

    def y_ind(self, y):
        y = (y+1)/2
        y_ind = (y*y_steps).astype(int)
        return np.minimum(y_steps-1,y_ind)

    def milanko_update(self, t):
        eps = float(self.eps_func(t/year2sec/1000))
        beta = float(self.beta_func(t/year2sec/1000))
        l_peri = float(self.l_peri_func(t/year2sec/1000))
        rho = (3/2*np.pi - l_peri)%(2*np.pi)
        return eps, beta, rho

    def RK4(self, X0, ddt, h):
        "Apply Runge Kutta Formulas to find next values of T and eta"
        k1 = ddt(X0)
        k2 = ddt(X0 + h*k1/2)
        k3 = ddt(X0 + h*k2/2)
        k4 = ddt(X0 + h*k3)
        return X0 + h/6*(k1 + 2*k2 + 2*k3 + k4)

    def iter_func(self):
        #Iter over all time points
        Q_year = np.empty((year_res, y_steps))
        for frame, t in enumerate(self.t_span):
            if frame%(t_steps//100)==0:print(frame/t_steps)
            if frame%(100*year_res) < year_res:
                if frame%(100*year_res) == 0:
                    eps, beta, rho = self.milanko_update(t)
                day_of_year = (frame%year_res)/year_res*year
                Q_year[frame%year_res,:] = Q_day(day_of_year, beta, rho, eps)
            self.Qs = Q_year[frame%year_res]
            X = self.RK4(np.append(self.T,self.etas), self.dX_dt, self.delta)
            self.T, self.etas = X[:-2],np.clip(X[-2:],-1,1)
            self.eta_record[frame//frame_refr,:] = self.etas
            self.max_T[frame//frame_refr] = max(self.T)
            if frame%frame_refr==0:
                yield t

    def update(self, t):
        self.ax.set_ylabel('$T(y$, {:.1e} years) (°C)'.format(t/year2sec))
        self.temp.set_xdata(self.y_span)
        self.temp.set_ydata(self.T)
        self.insol.set_xdata(self.y_span)
        self.insol.set_ydata(self.Qs/7-40)
        self.iceN.set_xdata(self.etas[1])
        self.iceS.set_xdata(self.etas[0])
        self.ax.set_ylim(-80,50)
        self.iceN.set_ydata(self.ax.get_ylim())
        self.iceS.set_ydata(self.ax.get_ylim())

    def animate(self):
        self.fig, self.ax = plt.subplots()
        plt.plot([-1,1],2*[T_ice])
        self.ydata, self.Tdata = [], []
        self.temp, = self.ax.plot([], [], 'r-', label='Temperature Profile',linewidth=3)
        self.insol, = self.ax.plot([], [], 'orange', label='Insol',linewidth=2)
        self.iceN, = self.ax.plot([], [], 'b-', label='Iceline',linewidth=3)
        self.iceS, = self.ax.plot([], [], 'b-',linewidth=3)
        self.ax.set_xlim(self.y_span[0], self.y_span[-1])
        self.ax.set_xlabel('y = sin(latitude)')
        self.fig.legend(framealpha=1)
        ani = FuncAnimation(self.fig, self.update, frames=self.iter_func, interval=1, repeat=False)
        plt.show()

if __name__ == '__main__':
    #main()
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
