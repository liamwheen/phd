import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib import rc
rc('text', usetex=True)
rc('font', family='serif',size=18)
"""
#3D plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
insol = np.loadtxt('../data/ave_insolation.csv',delimiter=',')
e58insol = np.loadtxt('../data/E_0.058_ave_insolation.csv',delimiter=',')
xx, yy = np.meshgrid(np.linspace(-90,90,insol.shape[1]),np.linspace(0,366,insol.shape[0]))
ax.plot_surface(xx,yy,e58insol-insol,cmap=cm.coolwarm)
ax.set_xlabel('Latitude (deg)')
ax.set_ylabel('Days Since Aphelion')
ax.set_zlabel('Insolation (W/m$^2$)')
plt.show()
# Single 2D plot
fig = plt.figure(figsize=(7,6))
ax = fig.add_subplot(111)
insol = np.loadtxt('../data/ave_insolation.csv',delimiter=',')
yy, xx = np.meshgrid(np.linspace(-90,90,insol.shape[1]),np.linspace(0,366,insol.shape[0]))
cont = ax.contourf(xx,yy,insol,cmap=cm.coolwarm)
ax.set_xlabel('Days Since Aphelion')
ax.set_ylabel('Latitude (Deg)')
fig.colorbar(cont, ax=ax)
plt.tight_layout()
#plt.savefig('../daily_ave_insolation_all_lats.pdf')
"""
fig, (now_ax, e6_ax, cbar1, space, delta_ax, cbar2) = plt.subplots(1,6,constrained_layout=True,
        gridspec_kw={'width_ratios':[8,8,0.5,0.5,8,0.5]},figsize=(15,4))
space.remove()
insol = np.loadtxt('../data/ave_insolation.csv',delimiter=',')
e6insol = np.loadtxt('../data/e0.06_ave_insolation.csv',delimiter=',')
yy, xx = np.meshgrid(np.linspace(-90,90,insol.shape[1]),np.linspace(0,366,insol.shape[0]))
cont = now_ax.contourf(xx,yy,insol,cmap=cm.coolwarm,vmin=0,vmax=640)
e6cont = e6_ax.contourf(xx,yy,np.roll(e6insol,1,axis=0),cmap=cm.coolwarm)
deltacont = delta_ax.contourf(xx,yy,np.roll(e6insol,1,axis=0)-insol,cmap=cm.RdBu_r)
cbar = fig.colorbar(e6cont, cax=cbar1)
cbar.set_label('Insolation (W/m$^2$)',rotation=270,labelpad=30)
cbar = fig.colorbar(deltacont, cax=cbar2)
cbar.set_label('$\Delta$ Insolation (W/m$^2$)',rotation=270,labelpad=30)
now_ax.set_yticks([-90,-60,-30,0,30,60,90])
e6_ax.set_yticks([-90,-60,-30,0,30,60,90])
delta_ax.set_yticks([-90,-60,-30,0,30,60,90])
now_ax.yaxis.set_major_formatter(FormatStrFormatter('%g$^\circ$'))
e6_ax.yaxis.set_major_formatter(FormatStrFormatter('%g$^\circ$'))
delta_ax.yaxis.set_major_formatter(FormatStrFormatter('%g$^\circ$'))
now_ax.set_ylabel('Latitude')
now_ax.set_xlabel('Days Since Aphelion')
e6_ax.set_xlabel('Days Since Aphelion')
delta_ax.set_xlabel('Days Since Aphelion')
#plt.savefig('../both_daily_ave_insolation_all_lats.pdf')
plt.show()
