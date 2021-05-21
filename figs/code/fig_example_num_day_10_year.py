import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib import rc
rc('text', usetex=True)
rc('font', family='serif', size=16)

"""
split this into two figures, one with just temp imshow and another with T0,
T90, and ice line (which still needs turning into data), showing how close the
model gets to the last 10 years of temp/ice (1990-1999), then maybe split up
the overlay figure to look like the 150k insol plot, with the insolation above
(maybe, try it anyway), then finish writing up this stuff, explaining the
reason its equator (less effect of obliquity due to high C) and the reason its
south (long winter effect), maybe then connect to Rubincam stuff. Also need to
read the rest of his report and look at the other papers recomended by Ian on
the other tabs. Also look at overlay fig code for other things to do.
"""

T = np.loadtxt('../data/example_num_day_T_10_years.csv',delimiter=',')
#T = np.roll(T,28,0)
eta = np.loadtxt('../data/example_num_day_eta_10_years.csv',delimiter=',')
T_0_90 = np.loadtxt('../data/T_0_T_90_10_years.csv',delimiter=',')
real_eta = np.loadtxt('../data/real_eta_90_99.csv',delimiter=',')
imfig, imax = plt.subplots(constrained_layout=True,figsize=(6.29,5))
compfig, compaxs = plt.subplots(2,1,gridspec_kw={'height_ratios':[3,1]},constrained_layout=True)

cont = imax.imshow(np.flipud(T.T),cmap=cm.coolwarm, interpolation='bicubic')
eta_ax = imax.twinx()
eta_ax.plot(eta,'k',label='Ice Line')
eta_ax.set_yticks([])
eta_ax.set_ylim(0,1)
eta_ax.legend()
cbar = imfig.colorbar(cont, ax=imax)
cbar.set_label('Temperature ($^\circ$C)',rotation=270,labelpad=20)
imax.set_ylabel('Latitude')
imax.set_yticks(T.shape[0]*np.sin(np.array([0,10,22,36,54,90][::-1])*np.pi/180))
imax.set_yticklabels(['{}$^\circ$'.format(i) for i in [0,10,22,38,54,90]])
imax.set_xticks(np.linspace(0,T.shape[1]-1,6))
imax.set_xticklabels(np.linspace(10,0,6,dtype=int))
imax.set_xlabel('Time (years)')

T = np.roll(T,28,0)
eta = np.roll(eta,22,0)
compaxs[0].plot(T[:,0],'C1',linewidth=2,label='$T^{0^\circ}$')
compaxs[0].plot(np.linspace(0,len(T),len(T_0_90)),T_0_90[:,1],'C1--',linewidth=2)
compaxs[0].plot(T[:,-1],'C0',linewidth=2,label='$T^{90^\circ}$')
compaxs[0].plot(np.linspace(0,len(T),len(T_0_90)),T_0_90[:,0],'C0--',linewidth=2)
compaxs[1].plot(np.linspace(0,T.shape[0]-1,len(eta)),eta,'C2',linewidth=2,label='Ice Line')
compaxs[1].plot(np.linspace(0,T.shape[0]-1,len(real_eta)),real_eta,'C2--',linewidth=2)

compaxs[1].set_ylim(0.92,1.008)
compaxs[0].set_ylabel('Temperature ($^\circ$C)')
compaxs[1].set_ylabel('Latitude')
#compaxs[1].set_ylabel('$T^{0^\circ}$ ($^\circ$C)')
compaxs[0].set_yticks([-25,0,25])
#compaxs[1].set_yticks([20,25])
#compaxs[1].set_ylim(18,27)
compaxs[1].set_yticks(np.sin(np.array([70,76,90])*np.pi/180))
compaxs[1].set_yticklabels(['{}$^\circ$'.format(i) for i in [70,76,90]])
compaxs[1].set_xticks(np.linspace(0,T.shape[1]-1,6))
compaxs[1].set_xticklabels(np.linspace(10,0,6,dtype=int))
compaxs[1].set_xlabel('Time (years)')
compaxs[0].set_xticks([])
#compaxs[0].margins(x=0)
#compaxs[2].margins(x=0)
compfig.legend()
plt.show()
#imfig.savefig('../example_num_day_10_year_T.pdf')
#compfig.savefig('../example_num_day_10_year_compare.pdf')
