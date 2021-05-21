import matplotlib.pyplot as plt
import sys
sys.path.append('../../code/data/')
from milanko_params import load_milanko
import numpy as np
from matplotlib import cm
from matplotlib import rc
rc('text', usetex=True)
rc('font', family='serif',size=14)

t, ecc, obliq, l_peri = load_milanko('backward')
rho = (3/2*np.pi - l_peri)%(2*np.pi)
south_insol_eq = np.loadtxt('../data/south_summer_insol.csv',delimiter=',')
T = np.loadtxt('../data/budyko_num_day_R_4.2_S_1_C_4.2.csv',delimiter=',')

fig, (T_ax, cbar_ax) = plt.subplots(1,2,constrained_layout=True,
        gridspec_kw={'width_ratios':[8,0.5]},figsize=(5.3,4))
#insol_ax = T_ax.twinx()
num_day_cont = T_ax.imshow(np.flipud(T.T),cmap=cm.coolwarm,
        interpolation='bicubic')
#note this plot flips due to imshow, so pi/2 is used instead of 3pi/2
#T_ax.plot(np.linspace(0,T.shape[0]-1,400),80+300*ecc[-400:]*np.sin(np.pi/2-rho[-400:]),'C2',linewidth=2,alpha=0.7)
#T_ax.legend(['$\\varepsilon\sin\left(\\frac{3\pi}{2}-\\rho\\right)$'])
T_ax.plot(np.linspace(0,T.shape[0]-1,400),230-0.3*south_insol_eq[:,5],'k',linewidth=2,alpha=0.7)
T_ax.legend(['$Q^{\mathrm{0^\circ S}}$'])

cbar = fig.colorbar(num_day_cont, cax=cbar_ax)
cbar.set_label('Temperature ($^\circ$C)',rotation=270,labelpad=15)
#T_ax.set_ylim(-2.2,0.16)
#T_ax.set_ylabel('$\epsilon\sin\left(\\frac{3\pi}{2}-\\rho\\right)$')
T_ax.set_ylabel('Latitude')
T_ax.set_yticks((T.shape[0]-1)*np.sin(np.array([0,10,22,36,54,90][::-1])*np.pi/180))
T_ax.set_yticklabels(['{}$^\circ$'.format(i) for i in [0,10,22,38,54,90]])
T_ax.set_xticks(np.linspace(0,T.shape[1]-1,6))
T_ax.set_xticklabels(np.linspace(400,0,6,dtype=int))
T_ax.set_xlabel('Time (kya)')
fig.savefig('../budyko_num_day_insol_overlay.pdf')
"""
why does eg insol at equator during south summer solstice match eps*sin(rho-pi/2) almost
exactly? even though insolation uses 1/sqrt(1-e^2) as the scaling term. look at
I_day and try and get an idea. This also governs the iceline/temperature in fig
26, (or minus northern if duration has the effect not the southern peak temp).
read the rubincam paper.

put this figure in the report.
Look at the meaning of 3pi/2-rho and confirm thats the opposite to the
'expected' q65 dependence, explain this, and C tweaking. Then look at levels of
variation within a year vs over 400k, which has larger variations? if year has
higher than 400k, is that reasonable, i think rubincam was who said that
glacial maximum was only 4C colder than now? Read the rubincam paper.

add ecc*sin(l_peri) plot on top with second axis for insolation, then explain
in report how this is different to q65 in that it has less dependence on
obliquity, which is because of the increased C value, maybe then compare with
the C=3.04 case when its equal to the q65 insolation curve. explain how C
needed changing to achieve the correct intra-year dynamics as well as changing
R and S. Then look into how this is still oppositely dependent on northern
insolation, depending instead on winter length than summer intensity. It also
stil fails to fluctuate much more (if even more at all) than the yearly
movement. If yearly movement is larger than this, then why would that not
trigger glacial movement rather than peaks/troffs in q65?
"""
