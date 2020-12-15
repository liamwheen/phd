import numpy as np
import sys
sys.path.append('../../code/')
from insol_sympy import calc_yearly_average

betas = [calc_yearly_average(i,np.linspace(-np.pi/2, np.pi/2, 200),0.01670236225492288)
            for i in [0,0.4,0.8,1.2]]
epss = [calc_yearly_average(0.4090928042223415,np.linspace(-np.pi/2, np.pi/2, 200),i)
            for i in [0,0.06,0.12,0.18]]

np.savetxt('../data/beta_effect.csv',np.array(betas),delimiter=',')
np.savetxt('../data/eps_effect.csv',np.array(epss),delimiter=',')

