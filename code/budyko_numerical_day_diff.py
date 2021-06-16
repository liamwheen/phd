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
from scipy.sparse import csr_matrix
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
T_ice_shift = np.array([-T_ice,T_ice])
R = 4.2*10**7 #J m^-2 K^-1 (Temp damping)
S = 1.2*10**9 # degC s (Ice line damping)
Q_0 = 340.327 #Wm^-2
        
etas0 = [-0.841,0.841] # Initial Icelines
tmin = -100
tmax = 0 # Years
year_res = 15000 # Points per year
year_span = np.linspace(0,year,year_res+1)[:-1] # Avoid repeating last point as first point

y_steps = 201
y_span = np.linspace(-1,1,y_steps)
t_steps = (tmax-tmin)*year_res
y_delta = y_span[1] - y_span[0]
t_span = np.linspace(tmin*year2sec,tmax*year2sec,t_steps) #years
delta = t_span[1] - t_span[0]
frame_refr = 140

dd1 =  np.eye(y_steps,k=1) + -np.eye(y_steps,k=-1)
dd1[0,1]=0
dd1[-1,-2]=0
dd2 = -2*np.eye(y_steps) + np.diag(np.ones(y_steps-1),1) + np.diag(np.ones(y_steps-1),-1)
diffuse_mat1 = np.diag(-2*y_span)@dd1*1/(2*y_delta)
diffuse_mat2 = np.diag(1-y_span**2)@dd2*1/y_delta**2
diffuse_mat = csr_matrix(diffuse_mat1+diffuse_mat2)

def run_long_term(tmin=tmin, tmax=tmax, jump=500):
    model = Budyko(tmin=tmin, tmax=tmax)
    years = np.arange(tmin, tmax+1, jump)
    Ts = np.empty((len(years),y_steps))
    for i,year in enumerate(years):
        print(f'{year:8}',end='\r')
        Ts[i,...] = np.mean(model.get_year_temp(year,10),0)
    return Ts

def main():
    model = Budyko()
    list(model.iter_func())
    np.savetxt('budyko_numerical_day_diff_eta.csv',model.eta_record,delimiter=',')
    np.savetxt('budyko_numerical_day_diff_T.csv',model.T_record,delimiter=',')

def anim_main():
    model = Budyko()
    model.animate()
    model.eta_record[model.eta_record==0] = np.nan
    print(np.nanmean(model.eta_record[:,1]))
    #print(max(model.max_T))
    #plt.plot(model.eta_record)
    #plt.show()

class Budyko:
    def __init__(self, etas0=etas0, tmin=tmin, tmax=tmax):
        self.eta_record = np.zeros((t_steps//frame_refr,2)) 
        self.T_record = np.zeros((t_steps//frame_refr,y_steps)) 
        #self.max_T = np.zeros(t_steps//frame_refr)
        milanko_t, milanko_ecc, milanko_obliq, milanko_l_peri = milanko_params.load_milanko('backward')
        krange = range(int(tmin//1000),2+max(1,int(tmax//1000)))
        self.eps_func = interp1d(milanko_t[krange], milanko_ecc[krange])
        self.beta_func = interp1d(milanko_t[krange], milanko_obliq[krange])
        self.l_peri_func = interp1d(milanko_t[krange], milanko_l_peri[krange])
        self.etas = np.array(etas0) #n initial
        self.T = np.zeros(y_steps)


    def a_eta(self, y):
        etaS, etaN = self.etas
        M = 200
        arg = (y>0)*(y - etaN) + (y<=0)*(-y + etaS)
        return (alpha_1+alpha_2)/2 + (alpha_2-alpha_1)/2 * (np.tanh(M*arg))

    def transport(self, T):
        return -C*(T - np.trapz(T,y_span)/2)

    def dX_dt(self, T, eta):
        dT = self.Qs*(1-self.a_eta(y_span)) - (A + B*T) + D*diffuse_mat.dot(T) #self.transport(T)#
        T_eta = T[self.y_ind(eta)]
        deta = np.array([-1,1])*T_eta - T_ice_shift
        #deta += (deta>0)*deta*2
        return dT/R, deta/S

    def y_ind(self, y):
        y = (y+1)/2
        y_ind = (y*y_steps).astype(int)
        return np.minimum(y_steps-1,y_ind)

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
        Q_year = np.empty((year_res, y_steps))
        for frame, t in enumerate(t_span):
            if frame%(t_steps//100)==0:print(frame/t_steps,end='\r')
            if frame%(100*year_res) == 0:
                Q_year = self.get_Q_year(t//year2sec)    
            self.Qs = Q_year[frame%year_res]
            self.T, etas = self.euler(self.T, self.etas, self.dX_dt, delta)
            etas[etas>1]=1
            etas[etas<-1]=-1
            self.etas = etas
            if frame%frame_refr==0:
                #day = int((frame%year_res/year_res*year+185)%year)
                #input(dt.datetime.strptime(f'{day}', '%j').strftime('%d %B'))
                self.eta_record[frame//frame_refr,:] = self.etas
                self.T_record[frame//frame_refr,:] = self.T
                #self.max_T[frame//frame_refr] = max(self.T)
                yield t

    def get_year_temp(self, end_year, run_time=10):
        T_year = np.empty((year_res, y_steps))
        Q_year = self.get_Q_year(end_year)
        for i in range(year_res*run_time):
            self.Qs = Q_year[i%year_res,:]
            self.T, etas = self.euler(self.T, self.etas, self.dX_dt, delta)
            etas[etas>1]=1
            etas[etas<-1]=-1
            self.etas = etas
            if i >= (year_res*(run_time-1)):
                T_year[i%year_res,:] = self.T
        return T_year

    def get_Q_year(self, year):
        f_name = f'.Q_year_cache/{int(np.ptp(y_span))}_{year_res}_{y_steps}_{int(year)}'
        try:
            with open(f_name,'rb') as f:
                Q_year = np.fromfile(f).reshape(year_res,y_steps)
        except FileNotFoundError:
            with open(f_name,'wb') as f:
                eps, beta, rho = self.milanko_update(year)
                Q_year = Q_day(year_span, beta, rho, eps, np.arcsin(y_span)).T
                Q_year.tofile(f)
        return Q_year


    def update(self, t):
        self.ax.set_ylabel('$T(y$, {:.1e} years) (°C)'.format(t/year2sec))
        self.temp.set_xdata(y_span)
        self.temp.set_ydata(self.T)
        self.insol.set_xdata(y_span)
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
        self.ax.set_xlim(y_span[0], y_span[-1])
        self.ax.set_xlabel('y = sin(latitude)')
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
