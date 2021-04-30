import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from matplotlib import cm
from matplotlib import rc
rc('text', usetex=True)
rc('font', family='serif',size=15)

T = np.loadtxt('../data/budyko_milanko_T.csv', delimiter=',')
fig, ax = plt.subplots(figsize=(6.4/1.2,4.8/1.2))
cont = ax.imshow(np.flipud(T.T),cmap=cm.coolwarm)#,vmin=0,vmax=640)
cbar = fig.colorbar(cont, ax=ax)
cbar.set_label('Temperature ($^\circ$C)',rotation=270,labelpad=20)
ax.set_yticks(T.shape[0]*np.sin(np.array([0,10,22,36,54,90][::-1])*np.pi/180))
ax.set_yticklabels(['{}$^\circ$'.format(i) for i in [0,10,22,38,54,90]])
ax.set_xticks(np.linspace(0,T.shape[1],5))
ax.set_xticklabels(np.linspace(800,0,5,dtype=int))
ax.set_ylabel('Latitude')
ax.set_xlabel('Time (kya)')
plt.tight_layout()
#plt.show()
plt.savefig('../budyko_milanko_T.pdf')

ice = np.loadtxt('../data/budyko_milanko_eta.csv',delimiter=',')
f, p = signal.periodogram(ice[20:],10/8) #Cut off the transient
fig, ax = plt.subplots(figsize=(9,2.5),tight_layout=True)
plt.plot(f,p,'C0',linewidth=3)
peaks = signal.find_peaks(p, prominence=max(p)/100)[0]
print(f[peaks],p[peaks])
#for i, peak in zip(['r','c','m','y'],peaks):
#    ax.plot(f[peak], p[peak],'{}o'.format(i),label='{:.1f}\,kyr'.format(1/f[peak]))
ax.set_xlim([0,0.08])
ax.set_xticks([1/41.3])
ax.set_xticklabels([41.3])
ax.plot(2*[1/41.3],ax.get_ylim(),'k--')
ax.margins(y=0)
ax.set_yticks([])
ax.set_xlabel('Period (kyr)')
ax.set_ylabel('Power')
ax.set_xlim([0,0.08])
#plt.show()
plt.savefig('../budyko_milanko_eta.pdf')
