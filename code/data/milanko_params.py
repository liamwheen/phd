import numpy as np
import csv
import os
wd = os.path.dirname(os.path.realpath(__file__))

def load_forward_milanko():
    with open(wd+'/milanko_20P.csv') as f:
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
    return t, ecc, obliq, l_peri

def load_backward_milanko():
    with open(wd+'/milanko_50N.csv') as f:
        csv_data = csv.reader(f)
        next(csv_data)
        t, ecc, obliq, l_peri = [],[],[],[]
        for line in csv_data:
            t.append(int(line[0]))
            ecc.append(float(line[1]))
            obliq.append(float(line[2]))
            l_peri.append(float(line[3]))

    t = np.flip(np.array(t))
    ecc = np.flip(np.array(ecc))
    obliq = np.flip(np.array(obliq))
    l_peri = np.flip(np.array(l_peri))
    return t, ecc, obliq, l_peri
