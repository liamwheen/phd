import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
from matplotlib import rc
rc('text', usetex=True)
rc('font', family='serif',size=15)
import warnings
warnings.filterwarnings('error')

Q = 340.375
A = 202.1
B = 1.9
C = 3.04
beta = 0.4090928
cb = 5/16*(3*np.sin(beta)**2 - 2)
Tice = -10
alpha1 = 0.32
alpha2 = 0.62
alpha0 = (alpha1+alpha2)/2

a0 = Q*(1-1/2*cb)*(1-alpha0) - A*(1+C/B) + C*Q/B*(1-alpha2) - (B+C)*Tice
a1 = C*Q/B*(alpha2-alpha1)*(1-1/2*cb)
a2 = 3/2*Q*cb*(1-alpha0)
a3 = C*Q*cb/(2*B)*(alpha2-alpha1)

ice_eq = lambda n:a3*n**3 + a2*n**2 + a1*n + a0
n_vals = np.linspace(-5,2,500)
tight_n_vals = np.linspace(0,1.1,500)
fig,axs = plt.subplots(1,3, gridspec_kw=({'width_ratios':(6,1,6)}),
        constrained_layout=True, figsize=(11,3.5))
axs[1].remove()
axs[0].plot(n_vals,ice_eq(n_vals),linewidth=3)
axs[2].plot(tight_n_vals,ice_eq(tight_n_vals),linewidth=3)
xlim0 = axs[0].get_xlim()
axs[0].plot([-10,10],2*[0],'k--')
axs[2].plot([-10,10],2*[0],'k--')
axs[0].set_xlim(xlim0)
axs[2].set_xlim([-0.05,1.05])
axs[2].set_ylim([-41,32])
axs[0].set_xlabel('$\eta$')
axs[2].set_xlabel('$\eta$')
axs[0].xaxis.label.set_fontsize(17)
axs[2].xaxis.label.set_fontsize(17)
axs[0].set_ylabel('$f(\eta)$',rotation=0,labelpad=10)
axs[2].set_ylabel('$f(\eta)$',rotation=0,labelpad=12)
#plt.show()
plt.savefig('../ice_line_equilibrium.pdf')
