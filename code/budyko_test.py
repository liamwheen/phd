import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

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
eps = 0.0167 #(eccentricity)
Q_e = Q_0/(np.sqrt(1-eps**2))
beta = 0.4091 #(obliquity) radians
c_b = (5/16)*(3*np.sin(beta)**2 - 2)

num_steps = 1*10**4
frame_refr = num_steps//(1*10**2)
        
class Budyko:
    def __init__(self):
        self.y_span = np.linspace(0,1,10000)
        self.y_delta = self.y_span[1]
        self.tspan = np.linspace(0,10*year2sec,num_steps) #years
        self.delta = self.tspan[1]
        #n0 = 0.24879494 # Initial iceline
        n0 = 0.8
        self.n = n0 #n initial
        #self.Tj = self.T_star(self.y_span) #set temp profile to eq profile
        self.Tj = np.zeros(self.y_span.size)
        self.Tnj = self.T_y(self.T_star(self.y_span), self.n)

    def s_b(self, y):
        return 1 + 0.5*c_b*(3*y**2 - 1)

    def a_n(self, y):
        # Account for fp errors in ice line positions
        round_n = self.y_delta*np.round(self.n/self.y_delta) 
        round_y = self.y_delta*np.round(y/self.y_delta) 
        return ((round_y!=round_n)*0.5+0.5)*((round_y<=round_n)*alpha_1+(round_y>=round_n)*alpha_2)
        # Using Widiasih's smooth a_n
        #M = 25
        #return 0.47 + 0.15 * (np.tanh(M*(y - self.n)))

    def int_T(self):
        #Currently always uses actual T rather than the equilibrium T, not sure
        #if thats right...
        return np.sum(self.Tj)*self.y_delta

    def T_star(self, y):
        In = np.sum(self.s_b(y)*(1-self.a_n(y)))*self.y_delta
        T_bar = (Q_e*In - A)/B
        T_star = (Q_e*self.s_b(y)*(1 - self.a_n(y)) - A + C*T_bar)/(B + C)
        return T_star

    def dT_dt(self, T, y):
        f = Q_e*self.s_b(y)*(1 - self.a_n(y)) - (A + B*T) - C*(T - self.int_T())
        #if not isinstance(T,float):
            #f[[0,-1]] = f[[1,-2]]
        return f/R

    def T_y(self, T, y):
        ind = np.clip(np.round(y*T.size),0,T.size-1).astype(int)
        return T[ind]
     
    def iter_func(self):
        #Iter over all time points
        for t in range(len(self.tspan)):
            self.Tj += self.delta*self.dT_dt(self.Tj, self.y_span)
            self.Tnj += self.delta*self.dT_dt(self.Tnj, self.n)
            self.n = np.clip(self.n + self.delta*(self.Tnj - T_ice)/S,0,1)
            if t%frame_refr==0: 
                yield self.tspan[t]

    def update(self, t):
        self.ax.set_ylabel('T(y, {:.1e} years) (°)'.format(t/year2sec))
        self.ax.set_xlabel('y = sin(Latitude)')
        self.temp.set_xdata(self.y_span)
        self.temp.set_ydata(self.Tj)
        self.ice.set_xdata(self.n)
        self.equil.set_xdata(self.n)
        self.equil.set_ydata(self.Tnj)
        #self.ax.set_ylim(min(self.Tj),max(self.Tj))
        self.ax.set_ylim(-50,50)
        self.ice.set_ydata(self.ax.get_ylim())
        self.grad.set_xdata(self.y_span[1:])
        self.grad.set_ydata(10000*np.diff(self.Tj))

    def animate(self):
        self.fig, self.ax = plt.subplots()
        self.ydata, self.Tdata = [], []
        self.temp, = self.ax.plot([], [], 'g-', label='Temperature Profile')
        self.ice, = self.ax.plot([], [], 'b-', label='sin(Iceline Position)')
        self.equil, = self.ax.plot([], [], 'ro', label='Equilibrium Temperature at Iceline')
        self.grad, = self.ax.plot([], [], 'k', linewidth=0.5, label='Gradient (scaled)')
        self.ax.set_xlim(self.y_span[0], self.y_span[-1])
        self.fig.legend(framealpha=1)
        ani = FuncAnimation(self.fig, self.update, frames=self.iter_func, interval=1, repeat=False)
        plt.show()

model = Budyko()
model.animate()


#plt.plot(model.y_span, model.s_b(model.y_span))
#plt.plot(model.y, model.y.size*[1])
#plt.plot(model.y, np.cumsum(model.s_b(model.y))*model.y_delta)
plt.show()
