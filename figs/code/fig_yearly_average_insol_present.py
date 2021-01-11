import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
from matplotlib import rc
rc('text', usetex=True)
rc('font', family='serif',size=18)
import sys
sys.path.append('../../code/')
sys.path.append('../../code/data')
from insol_sympy import calc_yearly_average
from milanko_params import load_milanko

t, ecc, obliq, _ = load_milanko('forward')
lats = np.linspace(-np.pi/2,np.pi/2,500)
sinlats = np.sin(np.linspace(-np.pi/2,np.pi/2,500))
y = np.sin(lats) 
insol = calc_yearly_average(obliq[0],lats,ecc[0])
c_b = (5/16)*(3*np.sin(0.401)**2 - 2)
budyko_insol = 340*(1 + 0.5*c_b*(3*y**2 - 1))
fig, ax = plt.subplots()
ax.plot(sinlats*90,insol,linewidth=4,label='$Q_{\mathrm{year}}$')
ax.plot(sinlats*90,budyko_insol,'k--',linewidth=3,label='$Q_\\varepsilon s_\\beta(y)$')
ax.legend()
ax.xaxis.set_major_formatter(FormatStrFormatter('%g$^\circ$'))
ax.set_xticks(90*np.sin(np.array([-90,-45,-20,0,20,45,90])*np.pi/180))
ax.set_xticklabels(['{}$^\circ$'.format(i) for i in [-90,-45,-20,0,20,45,90]])
ax.set_xlabel('Latitude')
ax.set_ylabel('Average Yearly Insolation (W/m$^2$)')
plt.tight_layout()
#plt.show()
plt.savefig('../yearly_average_insol_present.pdf')
