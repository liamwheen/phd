import matplotlib.pyplot as plt
import numpy as np
import csv

with open('milanko_20P.csv') as f:
    csv_data = csv.reader(f)
    next(csv_data)
    t, ecc, obliq = [],[],[]
    for line in csv_data:
        t.append(int(line[0]))
        ecc.append(float(line[1]))
        obliq.append(float(line[2]))
        
#plt.plot(t[:1000], ecc[:1000])
#plt.plot(t[:1000], np.array(obliq[:1000])-0.34)
#plt.show()
