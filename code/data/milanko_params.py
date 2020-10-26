import matplotlib.pyplot as plt
import numpy as np
import csv

with open('data/milanko_20P.csv') as f:
    csv_data = csv.reader(f)
    next(csv_data)
    t, ecc, obliq, l_peri = [],[],[],[]
    for line in csv_data:
        t.append(int(line[0]))
        ecc.append(float(line[1]))
        obliq.append(float(line[2]))
        l_peri.append(float(line[3]))
        
#plt.plot(t[:28], l_peri[:28])
#plt.show()
