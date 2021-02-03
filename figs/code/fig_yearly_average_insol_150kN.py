import matplotlib.pyplot as plt
import sys
sys.path.append('../../code/data/')
import milanko_params
from matplotlib import cm
import numpy as np
from matplotlib import rc
rc('text', usetex=True)
rc('font', family='serif',size=17)
"""
#3D version
fig = plt.figure(figsize=(7,6))
ax = fig.add_subplot(111, projection='3d')
insol = np.loadtxt('../data/yearly_insol_vals.csv',delimiter=',')
insol_ave = np.mean(insol,0)
insol_smooth=np.mean(insol[:,50:-50],1)
yy, xx = np.meshgrid(np.linspace(-90,90,insol.shape[1]),np.linspace(-150,0,insol.shape[0]))
cont = ax.plot_surface(xx,yy,insol,cmap=cm.coolwarm)
ax.set_xlabel('K Years Ago')
ax.set_ylabel('Latitude (Deg)')
fig.colorbar(cont, ax=ax)
plt.tight_layout()
plt.figure()
plt.plot(np.linspace(-150,0,insol.shape[0]), insol_smooth)
plt.show()
"""
fig, (ecc_ax, ax, ob_ax) = plt.subplots(3,1,constrained_layout=True,
        gridspec_kw={'height_ratios':[1,6,1]},figsize=(7,7))
t, ecc, obliq, _ =  milanko_params.load_milanko('backaward')
insol = np.loadtxt('../data/yearly_insol_vals.csv',delimiter=',')
yy, xx = np.meshgrid(90*np.sin(np.linspace(-np.pi/2,np.pi/2,insol.shape[1])),np.linspace(-150,0,insol.shape[0]))
insol_ave = np.mean(insol,0)
diff_insol = insol-insol_ave
#removing the few values <-8 so cbar range stays [-8:8]
diff_insol[np.where(diff_insol<-8)]=-8
cont = ax.contourf(xx,yy,np.sign(diff_insol)*np.log(abs(diff_insol)+1),cmap=cm.coolwarm)#[-8,-3,-2,-1,0,1,2,3,8],cmap=cm.coolwarm)
ob_ax.plot(t[-151:],obliq[-151:], linewidth=2)
ecc_ax.plot(t[-151:],ecc[-151:], linewidth=2)
ob_ax.set_xlabel('Thousand Years Ago')
ax.set_ylabel('Latitude')
ax.set_xticks([])
ecc_ax.set_xticks([])
ax.set_yticks(90*np.sin(np.array([-90,-45,-20,0,20,45,90])*np.pi/180))
ax.set_yticklabels(['{}$^\circ$'.format(i) for i in [-90,-45,-20,0,20,45,90]])
ob_ax.xaxis.set_major_locator(plt.MaxNLocator(6))
ob_ax.margins(x=0)
ecc_ax.margins(x=0)
ob_ax.set_xticks([-150,-125,-100,-75,-50,-25,0])
ob_ax.set_xticklabels([150,125,100,75,50,25,0])
ob_ax.yaxis.set_label_coords(-0.05, 0.25)
ecc_ax.yaxis.set_label_coords(-0.05, 0.35)
ob_ax.set_ylabel('$\\beta$', rotation=0, labelpad=10)
ecc_ax.set_ylabel('$\\varepsilon$', rotation=0, labelpad=10)
ecc_ax.yaxis.label.set_fontsize(25)
ob_ax.yaxis.label.set_fontsize(20)
ecc_ax.yaxis.tick_right()
ob_ax.yaxis.tick_right()
cbar = fig.colorbar(cont, ax=ax)
cbar.set_ticks([-2.4,-1.8,-1.2,-0.6,0,0.6,1.2,1.8,2.4])
cbar.set_ticklabels([np.round(np.sign(i)*(np.exp(abs(i))-1),1) for i in [-2.4,-1.8,-1.2,-0.6,0,0.6,1.2,1.8,2.4]])
cbar.set_label('$\Delta$ Insolation (W/m$^2$)',rotation=270,labelpad=25)
#plt.show()
plt.savefig('../yearly_average_insol_150kN.pdf')
