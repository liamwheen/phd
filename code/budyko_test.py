import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

sec2year = 3.154e+7 #Translate time dependent units to 'per year' instead of 'per second'

alpha_1 = 0.32
alpha_2 = 0.62
A = 202*sec2year**3 #Wm^-2
B = 1.9*sec2year**3 #Wm^-2
C = 3.04*sec2year**3 #Wm^-2 K^-1
T_ice = -10 #degC
R = 4*10**9*sec2year**2 #J m^-2 K^-1
S = 2.5*10**12
Q_0 = 343*sec2year**3 #Wm^-2
eps = 0.0167 #(eccentricity)
Q_e = Q_0/(np.sqrt(1-eps**2))
beta = 0.4091 #(obliquity) radians
c_b = (5/16)*(3*np.sin(beta)**2 - 2)
        
class Budyko:
    def __init__(self):
        self.y_span = np.linspace(0,1,1001)
        self.y_delta = self.y_span[1]
        self.tspan = np.linspace(0,10,101) #years
        self.delta = self.tspan[1]
        T0 = 0 # Initial temperature all over
        n0 = 0.9 # Initial iceline
        #self.Tj = T0*np.ones(self.y_span.size) #T initial
        self.Tj = 34 - 44*self.y_span**2 #T initial (quadratic distribution)
        self.n = n0 #n initial
        #self.Tjp1 = np.zeros(self.y_span.size) #T next iter

    def s_b(self, y):
        return 1 + 0.5*c_b*(3*y**2 - 1)

    def a_n(self, y):
        return ((y!=self.n)*0.5+0.5)*((y<=self.n)*alpha_1+(y>=self.n)*alpha_2)
        # Using Widiasih's smooth a_n
        #M = 25
        #return 0.47 + 0.15 * (np.tanh(M*(y - self.n)))

    def int_T(self):
        return np.sum(self.Tj)*self.y_delta

    def dT_dt(self, y):
        f = Q_e*self.s_b(y)*(1 - self.a_n(y)) - (A + B*self.Tj) - C*(self.Tj - self.int_T())
        f[[0,-1]] = f[[1,-2]] # Force dT/dy = 0 at ends, as should be the case for a cymmetric earth
        return f/R

    def dn_dt(self):
        return T_y(self.n) - T_ice

    def T_y(self, y):
        rad = np.arcsin(y)
        ratio = 2*rad/np.pi
        ind = np.round(ratio*self.Tj.size).astype(int)-1
        return self.Tj[ind]
     
    def iter_func(self):
        #Iter over all time points
        for t in range(1, len(self.tspan)+1):
            #Iter over all inner latitude points
            #for lat in range(1, len(self.y_span)):
            #    self.Tj[lat] += self.delta*self.dT_dt(self.y_span[lat])
            self.Tj += self.delta*self.dT_dt(self.y_span)
            self.n += self.delta*(self.T_y(self.n) - T_ice)/S
            yield self.tspan[t-1]

    def update(self, t):
        self.ax.set_ylabel('T(y, {:.1e})'.format(t))
        self.ax.set_xlabel('y = sin(Latitude)')
        self.temp.set_xdata(self.y_span)
        self.temp.set_ydata(self.Tj)
        self.ice.set_xdata(self.n)
        #self.ax.set_ylim(min(self.Tj),max(self.Tj))
        self.ax.set_ylim(-50,50)
        self.ice.set_ydata(self.ax.get_ylim())

    def animate(self):
        self.fig, self.ax = plt.subplots()
        self.ydata, self.Tdata = [], []
        self.temp, = self.ax.plot([], [], 'g-', label='Temperature Profile')
        self.ice, = self.ax.plot([], [], 'b-', label='sin(Iceline Position)')
        self.ax.set_xlim(self.y_span[0], self.y_span[-1])
        ani = FuncAnimation(self.fig, self.update, frames=self.iter_func, interval=50, repeat=False)
        plt.show()

model = Budyko()
model.animate()


#plt.plot(model.y_span, model.s_b(model.y_span))
#plt.plot(model.y, model.y.size*[1])
#plt.plot(model.y, np.cumsum(model.s_b(model.y))*model.y_delta)
plt.show()
