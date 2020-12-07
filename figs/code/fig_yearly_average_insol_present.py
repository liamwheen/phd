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
y = np.sin(lats) 
insol = calc_yearly_average(obliq[0],lats,ecc[0])
c_b = (5/16)*(3*np.sin(0.401)**2 - 2)
budyko_insol = 340*(1 + 0.5*c_b*(3*y**2 - 1))
fig, ax = plt.subplots()
ax.plot(lats*180/np.pi,insol,linewidth=4,label='$Q_{\mathrm{year}}$')
ax.plot(lats*180/np.pi,budyko_insol,'k--',linewidth=2,label='$Q_\\varepsilon s(y)$')
ax.legend()
ax.xaxis.set_major_formatter(FormatStrFormatter('%g$^\circ$'))
ax.set_xticks([-90,-60,-30,0,30,60,90])
ax.set_xlabel('Latitude')
ax.set_ylabel('Average Yearly Insolation (W/m$^2$)')
plt.tight_layout()
#plt.show()
plt.savefig('../yearly_average_insol_present.pdf')
