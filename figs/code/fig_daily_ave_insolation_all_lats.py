import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib import rc
rc('text', usetex=True)
rc('font', family='serif',size=17)
"""
#3D plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
insol = np.loadtxt('../data/ave_insolation.csv',delimiter=',')
xx, yy = np.meshgrid(np.linspace(-90,90,insol.shape[1]),np.linspace(0,366,insol.shape[0]))
ax.plot_surface(xx,yy,insol,cmap=cm.coolwarm)
ax.set_xlabel('Latitude (deg)')
ax.set_ylabel('Days Since Aphelion')
ax.set_zlabel('Insolation (W/m^2)')
"""
"""
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
plt.savefig('../daily_ave_insolation_all_lats.pdf')
"""
fig, axes = plt.subplots(1,2,figsize=(15,6))
fig.subplots_adjust(right=0.8)
insol = np.loadtxt('../data/ave_insolation.csv',delimiter=',')
e58insol = np.loadtxt('../data/E_0.058_ave_insolation.csv',delimiter=',')
yy, xx = np.meshgrid(np.linspace(-90,90,insol.shape[1]),np.linspace(0,366,insol.shape[0]))
cont = axes[0].contourf(xx,yy,insol,cmap=cm.coolwarm)
e58cont = axes[1].contourf(xx,yy,e58insol,cmap=cm.coolwarm)
fig.colorbar(e58cont)
axes[0].set_aspect(2.022099447513812)
axes[1].set_aspect(2.022099447513812)
axes[1].set_yticks([])
#ax = fig.add_subplot(111)
#ax.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
#ax.set_xlabel('Days Since Aphelion')
axes[0].set_ylabel('Latitude (Deg)')
axes[0].set_xlabel('Days Since Aphelion')
axes[1].set_xlabel('Days Since Aphelion')
plt.tight_layout()
plt.savefig('../both_daily_ave_insolation_all_lats.pdf')
