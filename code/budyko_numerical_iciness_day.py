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
from scipy.sparse import csr_matrix

year2sec = 31556952 #Translate time dependent units to 'per year' instead of 'per second'
year = 365.2425

alpha_1 = 0.32#0.279
alpha_2 = 0.62
A = 202.1#Wm^-2
B = 1.9 #Wm^-2
D = 0.5 #Wm^-2 K^-1
T_ice = -10 #degC
R = 3*10**7 #J m^-2 K^-1 (Temp damping)
S = 1*10**9 # degC s (Ice line damping)
Q_0 = 340.327 #Wm^-2
        
TMIN = -100
TMAX = 0 # Years
YEAR_RES = 3000 # Points per year
YEAR_SPAN = np.linspace(0,year,YEAR_RES+1)[:-1] # Avoid repeating last point as first point

Y_STEPS = phi_n
T_STEPS = (TMAX-TMIN)*YEAR_RES
FRAME_REFR = 50

Y_SPAN = np.linspace(-1,1,Y_STEPS)
Y_DELTA = Y_SPAN[1] - Y_SPAN[0]
T_SPAN = np.linspace(TMIN*year2sec,TMAX*year2sec,T_STEPS) #years
DELTA = T_SPAN[1] - T_SPAN[0]

dd1 = np.diag(np.ones(Y_STEPS-1),1) + np.diag(-np.ones(Y_STEPS-1),-1)
dd1[0,1]=0
dd1[-1,-2]=0
#dd1[0,0]=-1
#dd1[-1,-1]=1
dd2 = -2*np.eye(Y_STEPS) + np.diag(np.ones(Y_STEPS-1),1) + np.diag(np.ones(Y_STEPS-1),-1)
dd2[0,1]=2
dd2[-1,-2]=2
DIFFUSE_MAT = Y_DELTA*np.diag(-Y_SPAN)@dd1 + np.diag(1-Y_SPAN**2)@dd2
DIFFUSE_MAT*=1/Y_DELTA**2
#The element changes are to use neumann boundary conds, but because of 1-y^2,
#the diffuse mat (which is multiplied by 1-y^2) is 0 along top and bottom rows.
#This seems like it's messing up the boundary stuff.
print(DIFFUSE_MAT)
DIFFUSE_MAT = csr_matrix(DIFFUSE_MAT)


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
        milanko_t, milanko_ecc, milanko_obliq, milanko_l_peri = milanko_params.load_milanko('backward')
        krange = range(int(TMIN//1000),2+max(1,int(TMAX//1000)))
        self.eps_func = interp1d(milanko_t[krange], milanko_ecc[krange])
        self.beta_func = interp1d(milanko_t[krange], milanko_obliq[krange])
        self.l_peri_func = interp1d(milanko_t[krange], milanko_l_peri[krange])
        self.ice = np.zeros(Y_STEPS)
        self.ice[[0,1,2,3,-4,-3,-2,-1]]=1
        self.T = np.zeros(Y_STEPS)
        self.ice_record = np.zeros((T_STEPS//FRAME_REFR,Y_STEPS))

    def a_ice(self):
        return alpha_1 + self.ice*(alpha_2-alpha_1)

    def diffuse(self, T):
        return DIFFUSE_MAT@T

    def dX_dt(self, T, ice):
        dT = self.Qs*(1-self.a_ice()) - (A + B*T) + D*self.diffuse(T)
        dice = T_ice - T# - 0.1*(self.Qs-220)
        return dT/R, dice/S

    def milanko_update(self, t):
        eps = float(self.eps_func(t/1000))
        beta = float(self.beta_func(t/1000))
        l_peri = float(self.l_peri_func(t/1000))
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
        #dT = h*k1T
        #dice = h*k1ice
        return T+dT, ice+dice

    def iter_func(self):
        #Iter over all time points
        for frame, t in enumerate(T_SPAN):
            if frame%(T_STEPS//100)==0:print(frame/T_STEPS,end='\r')
            if frame%(100*YEAR_RES) == 0:
                Q_year = self.get_Q_year(t//year2sec)
            self.Qs = Q_year[frame%YEAR_RES]
            dT, dice = self.dX_dt(self.T, self.ice)
            self.T, ice_open = self.RK4(self.T, self.ice, self.dX_dt, DELTA)
            ice_open[ice_open<0]=0
            ice_open[ice_open>1]=1
            self.ice = ice_open
            if frame%FRAME_REFR==0:
                #day = int((frame%YEAR_RES/YEAR_RES*year+185)%year)
                #input(dt.datetime.strptime(f'{day}', '%j').strftime('%d %B'))
                self.ice_record[frame//FRAME_REFR,:] = self.ice
                #self.max_T[frame//FRAME_REFR] = max(self.T)
                yield t

    def get_Q_year(self, year):
        f_name = f'.Q_year_cache/{YEAR_RES}_{Y_STEPS}_{year}'
        try:
            with open(f_name,'rb') as f:
                Q_year = np.fromfile(f).reshape(YEAR_RES,Y_STEPS)
        except FileNotFoundError:
            with open(f_name,'wb') as f:
                eps, beta, rho = self.milanko_update(year)
                Q_year = Q_day(YEAR_SPAN, beta, rho, eps).T
                Q_year.tofile(f)
        return Q_year

    def update(self, t):
        self.ax.set_ylabel('$T(y$, {:.1e} years) (°C)'.format(t/year2sec))
        self.temp.set_ydata(self.T)
        self.insol.set_ydata(self.Qs/7-40)
        for i, rect in enumerate(self.iciness):
            rect.set_height(20*self.ice[i])

    def animate(self):
        self.fig, self.ax = plt.subplots()
        plt.plot([-1,1],2*[T_ice],':k',label='T_c')
        self.temp, = self.ax.plot(Y_SPAN, self.T, 'r-', label='Temperature',linewidth=3)
        self.insol, = self.ax.plot(Y_SPAN, Y_STEPS*[np.nan], 'orange', label='Insol',linewidth=2)
        self.iciness = self.ax.bar(Y_SPAN, self.ice, 2.05/Y_STEPS,
                label='Iciness (/20)')
        self.ax.set_xlim(Y_SPAN[0], Y_SPAN[-1])
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
