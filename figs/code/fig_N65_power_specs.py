import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
from matplotlib import rc
rc('text', usetex=True)
rc('font', family='serif',size=16)

fig, ax = plt.subplots(figsize=(9,2.31),tight_layout=True)
insol = np.loadtxt('../data/N65_insol.csv',delimiter=',')
#insol = np.loadtxt('../../code/insol_vals.csv')
ax.plot(np.linspace(-3000,0,insol.size),insol,'C3',linewidth=2)
ax.set_xlabel('Time (kya)')
ax.set_ylabel('Insolation (W/m$^2$)')
fig.savefig('../N65_time_series.pdf')

f, p = signal.welch(insol,10/3,nperseg=insol.size)
fig, ax = plt.subplots(figsize=(9,2.7),tight_layout=True)
plt.plot(f,p,'C3',linewidth=3)

peaks = signal.find_peaks(p, prominence=max(p)/4)[0]
print(f[peaks],p[peaks])
for i, peak in zip(['r','c','m','y'],peaks):
    ax.plot(f[peak], p[peak],'{}o'.format(i),label='{:.1f}\,kyr'.format(1/f[peak]))
ax.set_xlim([0,0.08])
ax.set_yticks([])
ax.legend(loc='upper right')
ax.set_xlabel('Frequency (1/kyr)')
ax.set_ylabel('Power')
ax.set_xlim([0,0.08])
fig.savefig('../N65_power_spec.pdf')
#plt.show()
