import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

Q_0 = 343 #Wm^-2
alpha_1 = 0.32
alpha_2 = 0.62
A = 202 #Wm^-2
B = 1.9 #Wm^-2
C = 3.04 #Wm^-2 K^-1
T_ice = -10 #degC
R = 4*10**8 #J m^-2 K^-1
S = 2.5*10**12
eps = 0.0167 #(eccentricity)
beta = 0.40910518 #(obliquity) radians
c_b = (5/16)*(3*np.sin(beta)**2 - 2)
        
class Budyko:
    def __init__(self):
        self.y = np.linspace(0,1,1001)
        self.y_delta = self.y[1]
        self.tspan = np.linspace(0,10**8,1001) #years
        self.delta = self.tspan[1]
        T0 = 10 # Initial temperature all over
        n0 = 0.9 # Initial iceline
        self.Tj = T0*np.ones(self.y.size) #T initial
        self.Tj = np.linspace(T0+10, T0-10, self.y.size) #T initial
        self.n = n0 #n initial
        self.Tjp1 = np.zeros(self.y.size) #T next iter

    def s_b(self, y):
        return 1 + 0.5*c_b*(3*y**2 - 1)

    def a_n(self, y):
        return ((y!=self.n)*0.5+0.5)*((y<=self.n)*alpha_1+(y>=self.n)*alpha_2)

    def int_T(self):
        return np.sum(self.T_y(np.arange(0,1,self.y_delta)))*self.y_delta

    def dT_dt(self, y):
        f = Q_0*self.s_b(y)*(1 - self.a_n(y)) - (A + B*self.T_y(y)) - C*(self.T_y(y) - self.int_T())
        return f/R

    def dn_dt(self):
        return T_y(self.n) - T_ice

    def T_y(self, y):
        rad = np.arcsin(y)
        ratio = 2*rad/np.pi
        inds = np.round(ratio*self.Tj.size).astype(int)-1
        return self.Tj[inds]
     
    def iter_func(self):
        #Iter over all time points
        for t in range(1, len(self.tspan)+1):
            #Iter over all inner latitude points
            #for lat in range(1, len(y)):
            self.Tj += self.delta*self.dT_dt(self.y)
            self.n += (self.T_y(self.n) - T_ice)/S
            yield self.tspan[t-1]

    def update(self, t):
        self.ax.set_ylabel('T(y, {:.1e})'.format(t))
        self.ax.set_xlabel('y = sin(Latitude)')
        self.temp.set_xdata(self.y)
        self.temp.set_ydata(self.Tj)
        self.ice.set_xdata(self.n)
        self.ice.set_ydata([-50,50])

    def animate(self):
        self.fig, self.ax = plt.subplots()
        self.ydata, self.Tdata = [], []
        self.temp, = self.ax.plot([], [], 'g-', label='Temperature Profile')
        self.ice, = self.ax.plot([], [], 'b-', label='sin(Iceline Position)')
        self.ax.set_xlim(self.y[0], self.y[-1])
        self.ax.set_ylim(-20, 30)
        ani = FuncAnimation(self.fig, self.update, frames=self.iter_func, interval=50, repeat=False)
        plt.show()

model = Budyko()
model.animate()


#plt.plot(model.y, model.s_b(model.y))
#plt.plot(model.y, model.y.size*[1])
#plt.plot(model.y, np.cumsum(model.s_b(model.y))*model.y_delta)
plt.show()
