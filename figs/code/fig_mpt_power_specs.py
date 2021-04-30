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

fig, (arw,ax) = plt.subplots(2,1,tight_layout=True,figsize=(10,3),gridspec_kw={'height_ratios':[1,7]})
ax.plot(-benth[:,0],benth[:,1],linewidth=1)
ax.set_ylabel('$\delta^{18}$O ($\permil$)')
ax.set_xlabel('Time (kya)')
ax.set_ylim([2.8082758,5.24689655])
ax.invert_yaxis()
ax.invert_xaxis()
arw.set_xlim(3150,-150)
arw.arrow(1794,1,754,0,width=.1,head_length=40,color='k')
arw.arrow(1794,1,-629,0,width=.1,head_length=40,color='k')
arw.arrow(500,1,160,0,width=.1,head_length=40,color='k')
arw.arrow(500,1,-460,0,width=.1,head_length=40,color='k')
arw.text(350,1.2,'100\,kyr',ha='center')
arw.text(1919,1.2,'41.1\,kyr',ha='center')
arw.axis('off')
benth2 = ax.twinx()
T = lambda b:16.1 - 4.76*(b-0.2)
benth2.yaxis.set_major_locator(plt.MaxNLocator(5))
benth2.set_ylim(-7.2,3.2)
benth2.plot(-benth[:,0],T(benth[:,1]),linewidth=0)
benth2.set_ylabel('Temperature ($^\circ$C)',rotation=-90,labelpad=20)
fig.savefig('../benthic_3m_years.pdf')

fig, axs = plt.subplots(1,1,tight_layout=True,figsize=(10,3))
ben_fun = interp1d(benth[:,0],benth[:,1])
quat_f, quat_p = welch(ben_fun(np.linspace(-2500,0,2500)),1,nperseg=790)
pre_f, pre_p = welch(ben_fun(np.linspace(-2500,-1250,1250)),1,nperseg=810)
post_f, post_p = welch(ben_fun(np.linspace(-700, 0,700)),1,nperseg=700)
quat, = axs.plot(quat_f,quat_p/np.mean(quat_p),linewidth=3,label='Quaternary',zorder=2)
pre, = axs.plot(pre_f,pre_p/np.mean(pre_p),'C1',linewidth=3,label='Before MPT',zorder=1)
post, = axs.plot(post_f,post_p/np.mean(post_p),'C2',linewidth=3,label='After MPT',zorder=3)
#for ax, powspec, freq in zip(axs,[quat_p, pre_p, post_p],[quat_f, pre_f, post_f]):
    #peaks = find_peaks(powspec, prominence=max(powspec)/2)[0]
    #for i, peak in zip(['r','c','m'],peaks):
        #ax.plot(freq[peak], powspec[peak],'{}o'.format(i),format(1/freq[peak]))
ylim = axs.get_ylim()[1]
axs.plot([1/100]*2,[-ylim*0.05,ylim*1.05],'k--')
axs.plot([1/41]*2,[-ylim*0.05,ylim*1.05],'k--')
axs.margins(y=0)
axs.set_xlim([0.001,0.05])
axs.set_xticks([1/100,1/41])
axs.set_xticklabels([100,41])
axs.set_yticks([])
axs.legend(ncol=1,loc='upper right')
axs.set_ylabel('Power')
axs.set_xlabel('Period (kyr)')
fig.savefig('../mpt_power_specs.pdf')

