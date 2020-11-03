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
        return t, ecc, obliq, l_peri

    return np.flip(t), np.flip(ecc), np.flip(obliq), np.flip(l_peri)
