import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
from matplotlib import rc
rc('text', usetex=True)
rc('font', family='serif',size=17)
#3D version
fig = plt.figure(figsize=(7,6))
ax = fig.add_subplot(111, projection='3d')
insol = np.loadtxt('../data/yearly_insol_vals.csv',delimiter=',')
yy, xx = np.meshgrid(np.linspace(-90,90,insol.shape[1]),np.linspace(-150,0,insol.shape[0]))
cont = ax.plot_surface(xx,yy,insol,cmap=cm.coolwarm)
ax.set_xlabel('K Years Ago')
ax.set_ylabel('Latitude (Deg)')
fig.colorbar(cont, ax=ax)
plt.tight_layout()
plt.show()
"""
fig = plt.figure(figsize=(7,6))
ax = fig.add_subplot(111)
insol = np.loadtxt('../data/yearly_insol_vals.csv',delimiter=',')
yy, xx = np.meshgrid(np.linspace(-90,90,insol.shape[1]),np.linspace(-150,0,insol.shape[0]))
cont = ax.contourf(xx,yy,insol,cmap=cm.coolwarm)
ax.set_xlabel('Thousand Years Since Present')
ax.set_ylabel('Latitude (Deg)')
fig.colorbar(cont, ax=ax)
plt.tight_layout()
#plt.savefig('../yearly_average_insol_150kN.pdf')
plt.show()
"""
