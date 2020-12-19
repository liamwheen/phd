import matplotlib.pyplot as plt
import sys
sys.path.append('../../code/data/')
from milanko_params import load_milanko
import numpy as np
from scipy import signal
from matplotlib import rc
rc('text', usetex=True)
rc('font', family='serif',size=16)

t, ecc, obliq, l_peri = load_milanko('backward')
prec= (((1/2)*np.pi - l_peri)%(2*np.pi) - np.pi)[-3001:] # Transform to used definition of rho
prec = l_peri[-3001:]
t = t[-3001:]
ecc = ecc[-3001:]
obliq = obliq[-3001:]

time_fig, time_ax = plt.subplots(3,1,tight_layout=True,figsize=(10,5))
time_ax[0].plot(t,obliq,linewidth=2)
time_ax[1].plot(t,prec,'C1',linewidth=2)
time_ax[2].plot(-t,ecc,'C2',linewidth=2)
time_ax[2].invert_xaxis()
time_ax[0].set_xticks([])
time_ax[1].set_xticks([])
time_ax[0].set_ylabel('Obliquity')
time_ax[1].set_ylabel('Precession')
time_ax[1].set_yticks([-3,0,3])
time_ax[1].set_ylim([-4,4])
time_ax[2].set_ylabel('Eccentricity')
time_ax[2].set_xlabel('Time (kya)')
#time_fig.savefig('../orbital_param_time_series.pdf')

ob_f, ob_p = signal.welch(obliq,nperseg=3000)
prec_f, prec_p = signal.welch(prec,nperseg=3000)
ecc_f, ecc_p = signal.welch(ecc,window='boxcar',nperseg=3000)
ob_fig, ob_ax = plt.subplots(figsize=(9,2.5))
prec_fig, prec_ax = plt.subplots(figsize=(9,2.5))
ecc_fig, ecc_ax = plt.subplots(figsize=(9,2.5))

ob_ax.plot(ob_f,ob_p,linewidth=3)
prec_ax.plot(prec_f,prec_p,'C1',linewidth=3)
ecc_ax.plot(ecc_f,ecc_p,'C2',linewidth=3)
for ax, powspec, freq in zip([ob_ax,prec_ax,ecc_ax],[ob_p, prec_p, ecc_p],[ob_f, prec_f, ecc_f]):
    peaks = signal.find_peaks(powspec, prominence=max(powspec)/2)[0]
    print(freq[peaks],powspec[peaks])
    for i, peak in zip(['r','c','m'],peaks):
        ax.plot(freq[peak], powspec[peak],'{}o'.format(i),label='{:.1f}\,kyr'.format(1/freq[peak]))
    ax.set_xlim([0,0.08])
    ax.set_yticks([])
    ax.legend(loc='upper right')
    ax.set_xlabel('Frequency (1/kyr)')
    ax.set_ylabel('Power')

for name, fig in zip(['beta', 'rho', 'ecc'],[ob_fig, prec_fig, ecc_fig]):
    fig.tight_layout()
    #fig.savefig('../{}_power_spec.pdf'.format(name))
plt.show()
