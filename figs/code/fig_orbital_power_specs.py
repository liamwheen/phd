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
t = t[-3001:]
ecc = ecc[-3001:]
obliq = obliq[-3001:]

ob_f, ob_p = signal.welch(obliq,nperseg=3000)
prec_f, prec_p = signal.welch(prec,nperseg=3000)
ecc_f, ecc_p = signal.welch(ecc,window='boxcar',nperseg=3000)
ob_fig, ob_ax = plt.subplots(figsize=(9,3))
prec_fig, prec_ax = plt.subplots(figsize=(9,3))
ecc_fig, ecc_ax = plt.subplots(figsize=(9,3))

ob_ax.plot(ob_f,ob_p)
prec_ax.plot(prec_f,prec_p,'C1')
ecc_ax.plot(ecc_f,ecc_p,'C2')
for ax, powspec, freq in zip([ob_ax,prec_ax,ecc_ax],[ob_p, prec_p, ecc_p],[ob_f, prec_f, ecc_f]):
    peaks = signal.find_peaks(powspec, prominence=max(powspec)/2)[0]
    for i, peak in zip(['r','c','m'],peaks):
        ax.plot(freq[peak], powspec[peak],'{}o'.format(i),label='{:.1f}\,kyr'.format(1/freq[peak]))
    ax.set_xlim([0,0.08])
    ax.set_yticks([])
    ax.legend(loc='upper right')
    ax.set_xlabel('Frequency (1/kyr)')
    ax.set_ylabel('Power')

for name, fig in zip(['beta', 'rho', 'ecc'],[ob_fig, prec_fig, ecc_fig]):
    fig.tight_layout()
    fig.savefig('../{}_power_spec.pdf'.format(name))
