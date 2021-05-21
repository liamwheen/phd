import numpy as np
from scipy.interpolate import interp1d
import calendar
import copy
import matplotlib.pyplot as plt
from matplotlib import rc,cm
from matplotlib.lines import Line2D
rc('text', usetex=True)
rc('font', family='serif',size=15)

imfig, (march_ax, sept_ax, cbar_ax) = plt.subplots(1,3,constrained_layout=True,
        gridspec_kw={'width_ratios':[8,8,0.3]},figsize=(12.5,3))
meanfig, mean_ax = plt.subplots(1,1,constrained_layout=True)
monthly = np.loadtxt('../data/monthly_eta_1870_2003.csv', delimiter=',')
march = np.loadtxt('../data/sea_ice_march.csv',delimiter=',')
sept = np.loadtxt('../data/sea_ice_sept.csv',delimiter=',')

cmap = copy.copy(cm.Blues)
cmap.set_bad('gainsboro')
march_plot = march_ax.imshow(march,cmap=cmap,interpolation='nearest')
sept_plot = sept_ax.imshow(sept,cmap=cmap,interpolation='nearest')
cbar = imfig.colorbar(march_plot, cax=cbar_ax)
cbar.set_label('\% Ice Concentration',rotation=270,labelpad=14)
march_ax.set_yticks(np.linspace(0,179,7))
sept_ax.set_yticks(np.linspace(0,179,7))
march_ax.set_xticks(np.linspace(0,359,7))
sept_ax.set_xticks(np.linspace(0,359,7))
march_ax.set_yticklabels(['{}$^\circ$'.format(i) for i in np.linspace(90,-90,7)])
sept_ax.set_yticklabels(['{}$^\circ$'.format(i) for i in np.linspace(90,-90,7)])
march_ax.set_xticklabels(['{}$^\circ$'.format(i) for i in np.linspace(-180,180,7)])
sept_ax.set_xticklabels(['{}$^\circ$'.format(i) for i in np.linspace(-180,180,7)])
march_ax.set_ylabel('Latitude')
march_ax.set_xlabel('Longitude')
sept_ax.set_xlabel('Longitude')

cm = plt.get_cmap('autumn_r')
mean_ax.set_prop_cycle(color=[cm(1.*i/len(monthly)) for i in range(len(monthly))])
mean_ax.plot(np.arange(1,13),monthly.T,alpha=0.4)
#[mean_ax.scatter(np.arange(1,13),monthly[i,:], s=25, c='C2', linewidths=0, alpha=0.3) for i in range(monthly.shape[0])]
mean = np.mean(monthly, 0)
#fun = interp1d(np.arange(1,13), mean)
#x = np.linspace(1,12,200)
mean_ax.set_xticks(np.arange(1,13))
mean_ax.set_xticklabels(calendar.month_abbr[1:],rotation=45,ha='right')
mean_ax.plot(np.arange(1,13),mean,'k',linewidth=3)
mean_ax.set_yticks(np.linspace(70,85,7))
mean_ax.set_yticklabels(['{}$^\circ$'.format(i) for i in np.linspace(70,85,7)])
mean_ax.set_ylim(69,86)
#mean_ax.legend(['Mean','Single Month'])
custom_lines = [Line2D([0], [0], color='k', lw=3),
                Line2D([0], [0], color=cm(1.), alpha=0.4, lw=2),
                Line2D([0], [0], color=cm(0.), alpha=0.4, lw=2)]
mean_ax.legend(custom_lines,['Mean','2003','1870'],loc='upper left')

#plt.show()
imfig.savefig('../sea_ice_march_sept.pdf')
meanfig.savefig('../monthly_eta_1870_2003.pdf')
