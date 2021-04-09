import matplotlib.pyplot as plt
import sys
sys.path.append('../../code/data/')
from milanko_params import load_milanko
import numpy as np
from scipy.signal import welch, find_peaks
from scipy.interpolate import interp1d
from matplotlib import rc
rc('text', usetex=True)
rc('font', family='serif',size=16)
rc('text.latex', preamble=r'\usepackage{wasysym}')

benth = np.loadtxt('../data/benthic.csv',delimiter=',')

fig, ax = plt.subplots(1,1,tight_layout=True,figsize=(10,3))
ax.plot(-benth[:,0],benth[:,1])
ax.set_ylabel('$\delta^{18}$O ($\permil$)')
ax.set_xlabel('Time (kya)')
ax.set_ylim([1.8,5.2])
ax.invert_yaxis()
ax.invert_xaxis()
ax.arrow(1794,2.8,754,0,width=.1,head_length=40,color='k')
ax.arrow(1794,2.8,-629,0,width=.1,head_length=40,color='k')
ax.arrow(500,2.8,160,0,width=.1,head_length=40,color='k')
ax.arrow(500,2.8,-460,0,width=.1,head_length=40,color='k')
ax.text(350,2.5,'100\,kyr',ha='center')
ax.text(1919,2.5,'41.1\,kyr',ha='center')
benth2 = ax.twinx()
T = lambda b:16.1 - 4.76*(b-0.2)
benth2.yaxis.set_major_locator(plt.MaxNLocator(5))
benth2.set_ylim(-7,7.5)
benth2.plot(-benth[:,0],T(benth[:,1]),linewidth=1)
benth2.set_ylabel('Temperature ($^\circ$C)',rotation=-90,labelpad=20)
fig.savefig('../benthic_3m_years.pdf')

fig, axs = plt.subplots(3,1,tight_layout=True,figsize=(10,5))
ben_fun = interp1d(benth[:,0],benth[:,1])
quat_f, quat_p = welch(ben_fun(np.linspace(-2500,0,2500)),1,nperseg=790)
pre_f, pre_p = welch(ben_fun(np.linspace(-2500,-1250,1250)),1,nperseg=810)
post_f, post_p = welch(ben_fun(np.linspace(-700, 0,700)),1,nperseg=700)
quat, = axs[0].plot(quat_f,quat_p,linewidth=3,label='Quaternary')
space = axs[0].plot([0],[0],color="w",label=' ')
pre, = axs[1].plot(pre_f,pre_p,'C1',linewidth=3,label='Before MPT')
post, = axs[2].plot(post_f,post_p,'C2',linewidth=3,label='After MPT')
for ax, powspec, freq in zip(axs,[quat_p, pre_p, post_p],[quat_f, pre_f, post_f]):
    peaks = find_peaks(powspec, prominence=max(powspec)/2)[0]
    for i, peak in zip(['r','c','m'],peaks):
        ax.plot(freq[peak], powspec[peak],'{}o'.format(i),label='{:.1f}\,kyr'.format(1/freq[peak]))
    ax.set_xlim([0,0.08])
    ax.set_yticks([])
    ax.legend(ncol=2,loc='upper right')
    ax.set_ylabel('Power')
axs[0].set_xticks([])
axs[1].set_xticks([])
axs[2].set_xlabel('Frequency (1/kyr)')
fig.savefig('../mpt_power_specs.pdf')

