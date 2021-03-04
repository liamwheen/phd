"""Probably not worth including in report"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib import rc
rc('text', usetex=True)
rc('font', family='serif',size=16)

sza_T = np.loadtxt('../data/budyko_sza_T_400.csv', delimiter=',')
orig_T = np.loadtxt('../data/budyko_milanko_T_400.csv', delimiter=',') 
fig, (orig_ax, space, sza_ax, cbar) = plt.subplots(1,4,constrained_layout=True,
        gridspec_kw={'width_ratios':[8,0.5,8,0.5]},figsize=(10,4))
space.remove()
min_val, max_val = np.amin((np.amin(sza_T),np.amin(orig_T))), np.amax((np.amax(sza_T),np.amax(orig_T)))
sza_cont = sza_ax.imshow(np.flipud(sza_T.T),cmap=cm.coolwarm,
        interpolation='bicubic', vmin=min_val, vmax=max_val)
orig_cont = orig_ax.imshow(np.flipud(orig_T.T),cmap=cm.coolwarm,
        interpolation='bicubic', vmin=min_val, vmax=max_val)
cbar = fig.colorbar(sza_cont, cax=cbar)
cbar.set_label('Temperature ($^\circ$C)',rotation=270,labelpad=20)
for ax in [sza_ax, orig_ax]:
    ax.set_yticks(orig_T.shape[0]*np.sin(np.array([0,10,22,36,54,90][::-1])*np.pi/180))
    ax.set_yticklabels(['{}$^\circ$'.format(i) for i in [0,10,22,38,54,90]])
    ax.set_xticks(np.linspace(0,orig_T.shape[1],5))
    ax.set_xticklabels(np.linspace(400,0,5,dtype=int))
    ax.set_xlabel('Time (kya)')
orig_ax.set_ylabel('Latitude')
plt.show()
#plt.savefig('../compare_numeric_budyko.pdf')

