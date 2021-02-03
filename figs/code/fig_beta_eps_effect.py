import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rc
rc('text', usetex=True)
rc('font', family='serif',size=19)

betas = np.loadtxt('../data/beta_effect.csv',delimiter=',')
print(np.trapz(betas[1:,:],np.sin(np.linspace(-np.pi/2, np.pi/2, 200))))
epss = np.loadtxt('../data/eps_effect.csv',delimiter=',')
fig, axs = plt.subplots(1,3,constrained_layout=True,figsize=(12,5),
        gridspec_kw={'width_ratios':[1,0.1,1]})
axs[1].remove()
betas_plot = axs[0].plot(90*np.sin(np.linspace(-np.pi/2,np.pi/2,betas.shape[1])),betas.T,linewidth=3)
epss_plot = axs[2].plot(90*np.sin(np.linspace(-np.pi/2,np.pi/2,epss.shape[1])),epss.T,linewidth=3)
axs[0].legend(betas_plot,['$\\beta = {}$'.format(i) for i in [0,0.4,0.8,1.2]])
axs[2].legend(epss_plot,['$\\varepsilon = {}$'.format(i) for i in
    [0,0.06,0.12,0.18]],loc='lower center')
axs[0].set_ylabel('Average Yearly Insolation (W/m$^2$)')
axs[0].set_xlabel('Latitude')
axs[2].set_xlabel('Latitude')
axs[0].set_xticks(90*np.sin(np.array([-90,-45,-20,0,20,45,90])*np.pi/180))
axs[2].set_xticks(90*np.sin(np.array([-90,-45,-20,0,20,45,90])*np.pi/180))
axs[0].set_xticklabels(['{}$^\circ$'.format(i) for i in [-90,-45,-20,0,20,45,90]])
axs[2].set_xticklabels(['{}$^\circ$'.format(i) for i in [-90,-45,-20,0,20,45,90]])
axs[2].set_ylim(*axs[0].get_ylim())
plt.show()
#plt.savefig('../beta_eps_effect.pdf')


