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
import datetime as dt

year2sec = 31556952 #Translate time dependent units to 'per year' instead of 'per second'
year = 365.2425

alpha_1 = 0.32#0.279
alpha_2 = 0.62
A = 202.1#Wm^-2
B = 1.9 #Wm^-2
C = 3.04 #Wm^-2 K^-1
D = 0.7 #Wm^-2 K^-1
T_ice = -10 #degC
R = 1*10**7 #J m^-2 K^-1 (Temp damping)
S = 1.23*10**9 # degC s (Ice line damping)
Q_0 = 340.327 #Wm^-2
        
tmin = -100
tmax = 0 # Years
year_res = 5500 # Points per year
year_span = np.linspace(0,year,year_res+1)[:-1] # Avoid repeating last point as first point

y_steps = phi_n#50
t_steps = (tmax-tmin)*year_res
frame_refr = 200

zeros_holder = np.zeros(y_steps)

def main():
    model = Budyko()
    list(model.iter_func())
    np.savetxt('budyko_numerical_day_ice.csv',model.ice_record,delimiter=',')

def anim_main():
    model = Budyko()
    model.animate()
    #model.ice_record[model.ice_record==0] = np.nan
    #print(np.nanmean(model.ice_record[:,1]))
    #print(max(model.max_T))
    #plt.plot(model.ice_record)
    #plt.show()

class Budyko:
    def __init__(self):
        self.y_span = np.linspace(-1,1,y_steps)
        self.y_delta = self.y_span[1] - self.y_span[0]
        self.t_span = np.linspace(tmin*year2sec,tmax*year2sec,t_steps) #years
        self.delta = self.t_span[1] - self.t_span[0]
        milanko_t, milanko_ecc, milanko_obliq, milanko_l_peri = milanko_params.load_milanko('backward')
        krange = range(int(tmin//1000),2+max(1,int(tmax//1000)))
        self.eps_func = interp1d(milanko_t[krange], milanko_ecc[krange])
        self.beta_func = interp1d(milanko_t[krange], milanko_obliq[krange])
        self.l_peri_func = interp1d(milanko_t[krange], milanko_l_peri[krange])
        self.ice = np.zeros(y_steps)
        self.T = np.zeros(y_steps)
        self.ice_record = np.zeros((t_steps//frame_refr,y_steps))

    def a_ice(self):
        return alpha_1 + self.ice*alpha_2

    def diffuse(self, T, y):
        dtdx = np.diff(self.T)/self.y_delta
        y_center = self.y_span[1:]-self.y_delta/2
        zeros_holder[1:-1] = D*np.diff((1-y_center**2)*dtdx)/self.y_delta
        return zeros_holder
    
    def dX_dt(self, T, ice):
        y = self.y_span
        dT = self.Qs*(1-self.a_ice()) - (A + B*T) + D*self.diffuse(T,y)
        dT[[0,-1]]=dT[[1,-2]]
        dice = T_ice - T
        return dT/R, dice/S

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

    def RK4(self, T, ice, ddt, h):
        "Apply Runge Kutta Formulas to find next values of T and ice"
        k1T, k1ice = ddt(T, ice)
        k2T, k2ice = ddt(T + h*k1T/2, ice + h*k1ice/2)
        k3T, k3ice = ddt(T + h*k2T/2, ice + h*k2ice/2)
        k4T, k4ice = ddt(T+ h*k3T, ice + h*k3ice)
        dT = h/6*(k1T + 2*k2T + 2*k3T + k4T)
        dice = h/6*(k1ice + 2*k2ice + 2*k3ice + k4ice)
        return T+dT, ice+dice

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
            self.T, ice_open = self.RK4(self.T, self.ice, self.dX_dt, self.delta)
            self.ice = np.clip(ice_open,0,1)
            if frame%frame_refr==0:
                #day = int((frame%year_res/year_res*year+185)%year)
                #input(dt.datetime.strptime(f'{day}', '%j').strftime('%d %B'))
                self.ice_record[frame//frame_refr,:] = self.ice
                #self.max_T[frame//frame_refr] = max(self.T)
                yield t

    def update(self, t):
        self.ax.set_ylabel('$T(y$, {:.1e} years) (°C)'.format(t/year2sec))
        self.temp.set_ydata(self.T)
        self.insol.set_ydata(self.Qs/7-40)
        for i, rect in enumerate(self.iciness):
            rect.set_height(20*self.ice[i])

    def animate(self):
        self.fig, self.ax = plt.subplots()
        plt.plot([-1,1],2*[T_ice],':k',label='T_c')
        self.temp, = self.ax.plot(self.y_span, self.T, 'r-', label='Temperature',linewidth=3)
        self.insol, = self.ax.plot(self.y_span, y_steps*[np.nan], 'orange', label='Insol',linewidth=2)
        self.iciness = self.ax.bar(self.y_span, self.ice, 2.05/y_steps,
                label='Iciness (/20)')
        self.ax.set_xlim(self.y_span[0], self.y_span[-1])
        self.ax.set_xlabel('y = sin(latitude)')
        self.fig.legend(framealpha=1)
        self.ax.set_ylim(-50,50)
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
