import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
import insolation

frame_refr = 5
k2day = 365250
tmin = 0 # days
tmax = tmin + 500
num_steps = 501
t_span = np.linspace(tmin,tmax,num_steps)

class Animate:
    def __init__(self):
        self.sim = insolation.Insolation(tmin, tmax, milanko_direction='forward')
        self.insol_vals = np.array([[None]*1]*num_steps)

    def iter_func(self):
        for frame, t in enumerate(t_span):
            self.insol_vals[frame,:] = self.sim.update(t)
            if frame%frame_refr==0 or t==t_span[-1]:
                yield t

    def update(self, t):
        self.ax.set_title('Days Since Aphelion: {:.1f}'.format(t%365.25))
        self.ax.set_ylabel('Semi-minor axis')
        self.ax.set_xlabel('Semi-major axis')
        theta = np.linspace(0, 2*np.pi, 100)
        a,b = self.sim.ellipse_axes(self.sim.eps)
        sun = np.sqrt(a**2-b**2)
        self.ellipse.set_xdata(sun+a*np.cos(theta))
        self.ellipse.set_ydata(b*np.sin(theta))
        earthx, earthy, _ = self.sim.pol2cart(*self.sim.polar_pos(t))
        self.earth.set_xdata(earthx)
        self.earth.set_ydata(earthy)
        latlon_unit = self.sim.latlon2unit(0,0)
        lon0x, lon0y, _ = self.sim.rotate_mat(self.sim.beta, self.sim.rho).dot(latlon_unit)
        self.latlon0.set_xdata([earthx,earthx+1e11*lon0x])
        self.latlon0.set_ydata([earthy,earthy+1e11*lon0y])
        self.insol_plot.set_xdata(np.linspace(tmin/365,tmax/365,len(self.insol_vals)))
        self.insol_plot.set_ydata(self.insol_vals[:,0])
        #self.insol_plot2.set_xdata(np.linspace(tmin/365,tmax/356,len(self.sim.insol_vals)))
        #self.insol_plot2.set_ydata(self.sim.insol_vals[:,1])
        self.insol_ax.set_xlim([tmin/365,max(t/365,tmin/365+1)])
        #self.insol_ax.set_ylim([400,600])

    def init(self):
        self.ellipse, = self.ax.plot([],[],'m--',linewidth=0.5)
        self.earth, = self.ax.plot([],[],'co')
        self.latlon0, = self.ax.plot([],[],'b')
        self.insol_ax = self.fig.add_subplot(333)
        self.insol_ax.set_xlabel('Years')
        self.insol_ax.set_ylabel('Ave Insol')
        self.insol_ax.yaxis.tick_right()
        self.insol_plot, = self.insol_ax.plot([],[],'b')
        #self.insol_plot2, = self.insol_ax.plot([],[],'r')
        self.ax.plot([0],[0],'yo',linewidth=4)
        au = insolation.au
        self.ax.set_xlim([-1.5*au,1.5*au])
        self.ax.set_ylim([-1.5*au,1.5*au])
        self.ax.set_aspect('equal')
        self.insol_ax.set_ylim([0,600])
        return self.ellipse, self.earth, self.latlon0, self.insol_plot

    def animate(self):
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111)
        ani = FuncAnimation(self.fig, self.update,
                frames=self.iter_func, init_func=self.init, interval=1, repeat=False)
        plt.show()

anim = Animate()
anim.animate()
 
