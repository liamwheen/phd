import numpy as np
import os
wd = os.path.dirname(os.path.realpath(__file__))

def load_milanko(direction):
    if direction == 'forward':
        data_file = wd+'/bin_milanko_20P'
    else:
        data_file = wd+'/bin_milanko_50N'

    with open(data_file,'rb') as f:
        t, ecc, obliq, l_peri = np.fromfile(f).reshape(4,-1)

    return t, ecc, obliq, l_peri
#import matplotlib.pyplot as plt
#tf,_,_,pf= load_milanko('forward')
#tb,e,_,pb= load_milanko('backward')
#print(min(e[-3000:]),max(e[-3000:]))
#t = np.concatenate([tb,tf])
#p = np.concatenate([pb,pf])
#plt.plot(t[51001-180:51160],p[51001-180:51160])
#plt.show()
