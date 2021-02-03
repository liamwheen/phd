import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
from scipy.interpolate import interp1d
from matplotlib import rc
rc('text', usetex=True)
rc('font', family='serif',size=16)

fig, ax = plt.subplots(figsize=(9,2.31),tight_layout=True)
co2 = np.loadtxt('../data/co2.csv',delimiter=',')
carbon_func = interp1d(co2[:,0],co2[:,1])
t_span = np.linspace(-800000, 0,1000)
ax.plot(-t_span/1000,carbon_func(t_span),linewidth=2)
ax.invert_xaxis()
ax.set_xlabel('Time (kya)')
ax.set_ylabel('CO$_2$ (ppmv)')
ax.set_yticks([180,230,280])
fig.savefig('../co2_time_series.pdf')

f, p = signal.periodogram(carbon_func(t_span),len(t_span)/(np.ptp(t_span)/1000))
fig, ax = plt.subplots(figsize=(9,2.7),tight_layout=True)
plt.plot(f,p,linewidth=3)

peaks = signal.find_peaks(p, prominence=max(p)/15)[0]
print(f[peaks],p[peaks])
for i, peak in zip(['r','c','m','y'],peaks):
    ax.plot(f[peak], p[peak],'{}o'.format(i),label='{:.1f}\,kyr'.format(1/f[peak]))
ax.set_xlim([0,0.08])
ax.set_yticks([])
ax.legend(loc='upper right')
ax.set_xlabel('Frequency (1/kyr)')
ax.set_ylabel('Power')
ax.set_xlim([0,0.08])
fig.savefig('../co2_power_spec.pdf')
#plt.show()
