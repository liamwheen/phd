import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib import rc
rc('text', usetex=True)
rc('font', family='serif',size=18)

orig_T = np.loadtxt('../data/original_budyko_T.csv', delimiter=',')
numeric_T = np.loadtxt('../data/numeric_budyko_T.csv', delimiter=',')
analytic_T = np.loadtxt('../data/analytical_budyko_T.csv', delimiter=',') 
fig, (analytic_ax, cbar0_ax, space, orig_diff_ax, numeric_diff_ax, cbar1_ax) = plt.subplots(1,6,constrained_layout=True,
        gridspec_kw={'width_ratios':[8,0.5,0.5,8,8,0.5]},figsize=(15,4))
space.remove()
analytic_cont = analytic_ax.imshow(np.flipud(analytic_T.T),cmap=cm.coolwarm,interpolation='bicubic')
orig_diff_cont = orig_diff_ax.imshow(np.flipud((orig_T-analytic_T).T),cmap=cm.RdBu_r,
        interpolation='bicubic', vmin=np.amin(orig_T-analytic_T), vmax=-np.amin(orig_T-analytic_T))
numeric_diff_cont = numeric_diff_ax.imshow(np.flipud((numeric_T-analytic_T).T),cmap=cm.RdBu_r,
        interpolation='bicubic', vmin=np.amin(orig_T-analytic_T), vmax=-np.amin(orig_T-analytic_T))
cbar0 = fig.colorbar(analytic_cont, cax=cbar0_ax)
cbar1 = fig.colorbar(orig_diff_cont, cax=cbar1_ax)
cbar0.set_label('Temperature ($^\circ$C)',rotation=270,labelpad=20)
cbar1.set_label('$\Delta$ Temperature ($^\circ$C)',rotation=270,labelpad=20)
for ax in [analytic_ax, orig_diff_ax, numeric_diff_ax]:
    ax.set_yticks(orig_T.shape[0]*np.sin(np.array([0,10,22,36,54,90][::-1])*np.pi/180))
    ax.set_yticklabels(['{}$^\circ$'.format(i) for i in [0,10,22,38,54,90]])
    ax.set_xticks(np.linspace(0,orig_T.shape[1],6))
    ax.set_xticklabels(np.linspace(10,0,6,dtype=int))
    ax.set_xlabel('Time (kya)')
analytic_ax.set_ylabel('Latitude')
print('Orig Rel Err: ', np.mean(abs(analytic_T-orig_T)))
print('Numeric Rel Err: ', np.mean(abs(analytic_T-numeric_T)))
#plt.show()
plt.savefig('../compare_numeric_budyko.pdf')

