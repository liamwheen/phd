import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append('../../code/data/')
from milanko_params import load_milanko
from scipy.interpolate import interp1d
from matplotlib import rc
rc('text', usetex=True)
rc('font', family='serif',size=16)
rc('text.latex', preamble=r'\usepackage{wasysym}')

Q65 = np.loadtxt('../data/N65_insol.csv',delimiter=',')
t, ecc, obliq, l_peri = load_milanko('backward')
ecc_fun = interp1d(t,ecc)
benth = np.loadtxt('../data/benthic.csv',delimiter=',')
ben_fun = interp1d(benth[:,0],benth[:,1])

fig, ax = plt.subplots(3,1,tight_layout=True,figsize=(10,5))
t = np.linspace(-1000,0,3333)
ax[1].plot(t,Q65[-3333:],linewidth=2)
ax[0].plot(t,ecc_fun(t),'C1',linewidth=2)
ax[2].plot(t[::-1],ben_fun(t[::-1]),'C2',linewidth=2)
ax[2].invert_yaxis()
ax[0].set_xticks([])
ax[1].set_xticks([])
ax[2].set_xticks(-np.arange(1000,-1,-200))
ax[2].set_xticklabels(np.arange(1000,-1,-200))
ax[0].set_yticks([0,0.05])
ax[2].set_yticks([5,4,3])
ax[2].set_ylim(5.18,2.9)
ax[1].set_ylabel('$Q^{65^\circ\mathrm{N}}$ (W/m$^2$)')
ax[0].set_ylabel('Eccentricity')
ax[2].set_ylabel('$\delta^{18}$O ($\permil$)')
ax[2].set_xlabel('Time (kya)')
peaks = [-20,-140,-251,-226,-342,-433,-539,-512,-585,
        -630,-719,-745,-796,-874,-920,-964]
spans = [15,15,10,10,15,25,15,20,10,20,22,13,16,15,10,10]
for a in ax:
    for peak, span in zip(peaks,spans):
        a.axvspan(peak, peak+span, color='grey', alpha=0.5)
#plt.show()
plt.savefig('../Q65_benth.pdf')
