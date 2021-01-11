import numpy as np
import csv
import os
wd = os.path.dirname(os.path.realpath(__file__))

def load_milanko(direction):
    if direction == 'forward':
        data_file = wd+'/milanko_20P.csv'
    else:
        data_file = wd+'/milanko_50N.csv'

    with open(data_file) as f:
        csv_data = csv.reader(f)
        next(csv_data)
        t, ecc, obliq, l_peri = [],[],[],[]
        for line in csv_data:
            t.append(int(line[0]))
            ecc.append(float(line[1]))
            obliq.append(float(line[2]))
            l_peri.append(float(line[3]))

    t = np.array(t)
    ecc = np.array(ecc)
    obliq = np.array(obliq)
    l_peri = np.array(l_peri)

    if direction == 'forward':
        #Make l_peri continuous to allow for interpolation later
        inds = np.where(np.diff(l_peri)<0)[0]+1
        for ind in inds:
            l_peri[ind:]+=2*np.pi
        return t, ecc, obliq, l_peri

    #Make l_peri continuous to allow for interpolation later
    inds = np.where(np.diff(l_peri)>0)[0]+1
    for ind in inds:
        l_peri[ind:]-=2*np.pi

    return np.flip(t), np.flip(ecc), np.flip(obliq), np.flip(l_peri)


#import matplotlib.pyplot as plt
#tf,_,_,pf= load_milanko('forward')
#tb,e,_,pb= load_milanko('backward')
#print(min(e[-3000:]),max(e[-3000:]))
#t = np.concatenate([tb,tf])
#p = np.concatenate([pb,pf])
#plt.plot(t[51001-180:51160],p[51001-180:51160])
#plt.show()
