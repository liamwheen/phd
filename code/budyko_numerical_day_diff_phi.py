#!/usr/bin/env python3
"""
Budyko model with diffusion method for heat transport, also using the latitude
domain instead of the y domain. This solves issues of boundary conditions
vanishing due to (1-y^2) multiplication. This also works as the best
approximation for real world values for temperature"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from matplotlib.animation import FuncAnimation
from data import milanko_params # Values given in 1000 year time steps
from numeric_Q_day import Q_day
from scipy.sparse import csr_matrix
import datetime as dt

year2sec = 31556952 #Translate time dependent units to 'per year' instead of 'per second'
year = 365.2425

alpha_1 = 0.32#0.279
alpha_2 = 0.62
A = 202.1#Wm^-2
B = 1.9 #Wm^-2
C = 4.2 #Wm^-2 K^-1
D = .48 #Wm^-2 K^-1
T_ice = -10 #degC
T_ice_shift = np.array([-T_ice,T_ice])
R = 4*10**7 #J m^-2 K^-1 (Temp damping)
S = 1*10**8 # degC s (Ice line damping)
Q_0 = 340.327 #Wm^-2
        
etas0 = [-1.,1.] # Initial Icelines
tmin = -400000
tmax = -300000 # Years
year_res = 4000 # Points per year
year_span = np.linspace(0,year,year_res+1)[:-1] # Avoid repeating last point as first point

phi_steps = 201
phi_span = np.linspace(-np.pi/2,np.pi/2,phi_steps)
t_steps = (tmax-tmin)*year_res
phi_delta = phi_span[1] - phi_span[0]
t_span = np.linspace(tmin*year2sec,tmax*year2sec,t_steps) #years
delta = t_span[1] - t_span[0]
frame_refr = year_res//40

dd1 = np.eye(phi_steps,k=1) - np.eye(phi_steps,k=-1)
dd1[0,1]=0
dd1[-1,-2]=0
dd2 = -2*np.eye(phi_steps) + np.eye(phi_steps,k=1) + np.eye(phi_steps,k=-1)
dd2[0,1]=2
dd2[-1,-2]=2
diffuse_mat1 = np.diag(-np.tan(phi_span))@dd1*1/(2*phi_delta)
diffuse_mat2 = dd2*1/phi_delta**2
diffuse_mat = csr_matrix(diffuse_mat1+diffuse_mat2)

def run_long_term(tmin=tmin, tmax=tmax, jump=500):
    model = Budyko(tmin=tmin, tmax=tmax)
    years = np.arange(tmin, tmax+1, jump)
    Ts = np.empty((len(years),year_res,phi_steps))
    etas = np.zeros((len(years),year_res,2))
    last_eta_ave = np.array(etas0)
    for i,year in enumerate(years):
        Ts[i,...], etas[i,...] = model.get_year_temp(year,30,last_eta_ave)
        last_eta_ave = np.mean(etas[i,...],0)
        print(f'{year:8}',end='\r')
    return Ts, etas

def main():
    model = Budyko()
    list(model.iter_func())
    model.eta_record[model.eta_record==0] = np.nan
    print(np.nanmean(model.eta_record[:,1]))
    np.savetxt('budyko_numerical_day_diff_phi_eta.csv',model.eta_record,delimiter=',')
    #np.savetxt('budyko_numerical_day_diff_phi_T.csv',model.T_record,delimiter=',')

def anim_main():
    model = Budyko()
    model.animate()
    print('Max: ',np.amax(model.eta_record))
    model.eta_record[model.eta_record==0] = np.nan
    print(np.nanmean(model.eta_record[:,1]))
    #print(max(model.max_T))
    plt.plot(model.eta_record)
    plt.show()

class Budyko:
    def __init__(self, etas0=etas0, tmin=tmin, tmax=tmax):
        self.eta_record = np.zeros((t_steps//frame_refr,2)) 
        #self.T_record = np.zeros((t_steps//frame_refr,phi_steps)) 
        #self.max_T = np.zeros(t_steps//frame_refr)
        milanko_t, milanko_ecc, milanko_obliq, milanko_l_peri = milanko_params.load_milanko('backward')
        self.eps_func = interp1d(milanko_t, milanko_ecc, 'cubic')
        self.beta_func = interp1d(milanko_t, milanko_obliq, 'cubic')
        self.l_peri_func = interp1d(milanko_t, milanko_l_peri, 'cubic')
        self.etas = np.array(etas0) #n initial
        self.T = np.zeros(phi_steps)

    def a_eta(self, y):
        etaS, etaN = self.etas
        M = 200
        arg = (y>0)*(y - etaN) + (y<=0)*(-y + etaS)
        return (alpha_1+alpha_2)/2 + (alpha_2-alpha_1)/2 * (np.tanh(M*arg))

    def transport(self, T):
        return -C*(T - np.trapz(T,np.sin(phi_span))/2)

    def dX_dt(self, T, eta):
        dT = self.Qs*(1-self.a_eta(phi_span)) - (A + B*T) + D*diffuse_mat.dot(T) #self.transport(T)#
        T_eta = self.T_at_phi(T, eta)
        deta = np.array([-1,1])*T_eta - T_ice_shift
        #deta/=np.cos(eta)
        #deta += (deta>0)*deta*2
        return dT/R, deta/S

    def T_at_phi(self, T, phi):
        top_ind = ((phi+np.pi/2)/np.pi*phi_steps+1*(phi<0)).astype(int)
        top_ind = np.minimum(top_ind, phi_steps-1)
        bot_phi, top_phi = phi_span[top_ind-1], phi_span[top_ind]
        top_weight = (phi-bot_phi)/(top_phi-bot_phi)
        T_approx = top_weight*T[top_ind] + (1-top_weight)*T[top_ind-1]
        return T_approx

    def milanko_update(self, t):
        eps = float(self.eps_func(t/1000))
        beta = float(self.beta_func(t/1000))
        l_peri = float(self.l_peri_func(t/1000))
        rho = (3/2*np.pi - l_peri)%(2*np.pi)
        return eps, beta, rho

    def RK4(self, T, eta, ddt, h):
        "Apply Runge Kutta Formulas to find next values of T and eta"
        k1T, k1eta = ddt(T, eta)
        k2T, k2eta = ddt(T + h*k1T/2, eta + h*k1eta/2)
        k3T, k3eta = ddt(T + h*k2T/2, eta + h*k2eta/2)
        k4T, k4eta = ddt(T+ h*k3T, eta + h*k3eta)
        dT = h/6*(k1T + 2*k2T + 2*k3T + k4T)
        deta = h/6*(k1eta + 2*k2eta + 2*k3eta + k4eta)
        return T+dT, eta+deta

    def euler(self, T, eta, ddt, h):
        dT, deta = ddt(T,eta)
        return T+h*dT, eta+h*deta

    def iter_func(self):
        #Iter over all time points
        Q_year = np.empty((year_res, phi_steps))
        for frame, t in enumerate(t_span):
            if frame%(t_steps//100)==0:print(frame/t_steps,end='\r')
            if frame%(100*year_res) == 0:
                Q_year = self.get_Q_year(t//year2sec)    
            self.Qs = Q_year[frame%year_res]
            self.T, etas = self.euler(self.T, self.etas, self.dX_dt, delta)
            etas[etas>np.pi/2]=np.pi/2
            etas[etas<-np.pi/2]=-np.pi/2
            self.etas = etas
            if frame%frame_refr==0:
                self.eta_record[frame//frame_refr,:] = self.etas
                yield t

    def get_year_temp(self, end_year, run_time, last_etas_ave):
        etas_year = np.empty((year_res,2))
        T_year = np.empty((year_res, phi_steps))
        Q_year = self.get_Q_year(end_year)
        self.etas = last_etas_ave
        for i in range(year_res*run_time):
            self.Qs = Q_year[i%year_res,:]
            self.T, etas = self.euler(self.T, self.etas, self.dX_dt, delta)
            etas[etas>np.pi/2]=np.pi/2
            etas[etas<-np.pi/2]=-np.pi/2
            self.etas = etas
            if i >= (year_res*(run_time-1)):
                etas_year[i%year_res,:] = self.etas
                T_year[i%year_res,:] = self.T
        return T_year, etas_year

    def get_Q_year(self, year):
        f_name = f'.Q_year_cache/{int(np.ptp(phi_span))}_{year_res}_{phi_steps}_{int(year)}'
        try:
            with open(f_name,'rb') as f:
                Q_year = np.fromfile(f).reshape(year_res,phi_steps)
        except FileNotFoundError:
            with open(f_name,'wb') as f:
                eps, beta, rho = self.milanko_update(year)
                Q_year = Q_day(year_span, beta, rho, eps, phi_span).T
                Q_year.tofile(f)
        return Q_year


    def update(self, t):
        self.ax.set_ylabel('$T(y$, {:.1e} years) (°C)'.format(t/year2sec))
        self.temp.set_xdata(phi_span)
        self.temp.set_ydata(self.T)
        self.insol.set_xdata(phi_span)
        self.insol.set_ydata(self.Qs/7-40)
        self.iceN.set_xdata(self.etas[1])
        self.iceS.set_xdata(self.etas[0])
        self.ax.set_ylim(-80,50)
        self.iceN.set_ydata(self.ax.get_ylim())
        self.iceS.set_ydata(self.ax.get_ylim())

    def animate(self):
        self.fig, self.ax = plt.subplots()
        plt.plot([-np.pi/2,np.pi/2],2*[T_ice])
        self.ydata, self.Tdata = [], []
        self.temp, = self.ax.plot([], [], 'r-', label='Temperature Profile',linewidth=3)
        self.insol, = self.ax.plot([], [], 'orange', label='Insol',linewidth=2)
        self.iceN, = self.ax.plot([], [], 'b-', label='Iceline',linewidth=3)
        self.iceS, = self.ax.plot([], [], 'b-',linewidth=3)
        self.ax.set_xlim(phi_span[0], phi_span[-1])
        self.ax.set_xlabel('$\\varphi$')
        self.fig.legend(framealpha=1)
        ani = FuncAnimation(self.fig, self.update, frames=self.iter_func, interval=1, repeat=False)
        plt.show()

if __name__ == '__main__':
    #main()
    anim_main()
    #run_long_term(-100000,0)
    """
    import cProfile, pstats
    profiler = cProfile.Profile()
    profiler.enable()
    run_long_term(-1000,0)
    profiler.disable()
    stats = pstats.Stats(profiler).sort_stats('cumulative')
    stats.print_stats()
   """ 
