import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib import rc
rc('text', usetex=True)
rc('font', family='serif',size=18)

orig_T = np.loadtxt('../data/budyko_milanko_400k_T.csv', delimiter=',')
num_year_T = np.loadtxt('../data/budyko_num_year_400k_T.csv', delimiter=',')
num_day_T = np.loadtxt('../data/budyko_num_day_R_4.2_S_1_C_4.2.csv', delimiter=',') 
vmax = np.amax(np.concatenate((orig_T,num_year_T,num_day_T)))
vmin = np.amin(np.concatenate((orig_T,num_year_T,num_day_T)))

fig, (orig_ax, num_year_ax, num_day_ax, cbar_ax) = plt.subplots(1,4,constrained_layout=True,
        gridspec_kw={'width_ratios':[8,8,8,0.5]},figsize=(13.5,4))
orig_cont = orig_ax.imshow(np.flipud(orig_T.T),cmap=cm.coolwarm,
        interpolation='bicubic',vmin=vmin,vmax=vmax)
num_year_cont = num_year_ax.imshow(np.flipud(num_year_T.T),cmap=cm.coolwarm,
        interpolation='bicubic', vmin=vmin,vmax=vmax)
num_day_cont = num_day_ax.imshow(np.flipud(num_day_T.T),cmap=cm.coolwarm,
        interpolation='bicubic', vmin=vmin,vmax=vmax)
cbar = fig.colorbar(orig_cont, cax=cbar_ax)
cbar.set_label('Temperature ($^\circ$C)',rotation=270,labelpad=20)
for ax in (orig_ax, num_year_ax, num_day_ax):
    ax.set_yticks(orig_T.shape[0]*np.sin(np.array([0,10,22,36,54,90][::-1])*np.pi/180))
    ax.set_yticklabels(['{}$^\circ$'.format(i) for i in [0,10,22,38,54,90]])
    ax.set_xticks(np.linspace(0,orig_T.shape[1],6))
    ax.set_xticklabels(np.linspace(400,0,6,dtype=int))
    ax.set_xlabel('Time (kya)')
orig_ax.set_ylabel('Latitude')
fig.savefig('../compare_budyko_400k_sim.pdf')


