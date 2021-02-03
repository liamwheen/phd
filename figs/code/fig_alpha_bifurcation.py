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

def cubic(alpha1, alpha2):
    alpha0 = (alpha1+alpha2)/2
    a0 = Q*(1-1/2*cb)*(1-alpha0) - A*(1+C/B) + C*Q/B*(1-alpha2) - (B+C)*Tice
    a1 = C*Q/B*(alpha2-alpha1)*(1-1/2*cb)
    a2 = 3/2*Q*cb*(1-alpha0)
    a3 = C*Q*cb/(2*B)*(alpha2-alpha1)

    return lambda n:a3*n**3 + a2*n**2 + a1*n + a0

alphas = np.linspace(0.1,0.45,1000)
nsols = np.empty((len(alphas),2))
for i,alph in enumerate(alphas):
    nstar = cubic(alph,0.94-alph)
    for x0 in [0,0.1,0.2,0.3,0.4]:
        try:
            nsol = fsolve(nstar,np.array([x0,1-x0]))
            break
        except Warning:
            if x0==0.4: nsol = np.empty(2)
            print('failed')

    nsol[(nsol<0)|(nsol>1)] = np.nan
    nsols[i,:] = nsol

nsols[:len(nsols)//2:,1]=np.nan
nsols[9*len(nsols)//10:,0]=np.nan
fig,axs = plt.subplots(1,3, gridspec_kw=({'width_ratios':(6,1,6)}),
        constrained_layout=True, figsize=(11,4.5))
axs[1].remove()
axs[0].plot(alphas, nsols[:,1], 'C0',linewidth=3, label='Stable')
axs[0].plot(alphas, nsols[:,0], '--C1',linewidth=3, label='Unstable')
axs[0].legend()
axs[0].set_xlabel('$\\alpha_1$')
axs[0].xaxis.label.set_fontsize(17)
axs[0].set_ylabel('$\eta^*$',rotation=0,labelpad=12)
secax = axs[0].secondary_xaxis('top',functions=(lambda x:0.94-x,lambda x:0.94-x))
secax.set_xlabel('$\\alpha_2$',labelpad=10)
secax.xaxis.label.set_fontsize(17)
ylims = axs[0].get_ylim()
axs[0].plot(2*[0.32],[-2,2],'--C2')
axs[0].set_ylim(ylims)
alphas = np.linspace(0.1,0.43,4)
#[plt.scatter(nsols[max(0,i*100),:],2*[0],c='C%d'%(9-i)) for i in range(10)]
for i,alph in enumerate(alphas):
    nstar = cubic(alph,0.94-alph)
    n = np.linspace(0,1,10000)
    f = nstar(n)
    axs[2].plot(n,f,zorder=0,label='$\\alpha_1 = {0:.2f},\,\\alpha_2 = {1:.2f}$'.format(alph,0.94-alph))
    axs[2].plot(n[np.where(abs(f)<0.03)],len(np.where(abs(f)<0.03)[0])*[0],'C%do'%i,zorder=1)
    print(n[np.where(abs(f)<0.03)])

axs[2].legend()
axs[2].plot([0,1],2*[0],'k',linewidth=1,zorder=0.5)
axs[2].set_ylabel('$f(\eta)$',rotation=0,labelpad=5)
axs[2].set_xlabel('$\eta$')
"""
nsols = np.empty((len(alphas),2))
for i,alph in enumerate(alphas):
    sols = cubic(0.32,alph)
    sols[(sols<0)|(sols>1)] = np.nan
    nsols[i,:] = sols

plt.plot(alphas, nsols)
"""
#plt.savefig('../alpha_bifurcation.pdf')
