import numpy as np
import cmath as cm
import matplotlib.pyplot as plt
import csv
"""This method is probably not worth while as the data is already available
from -250M to +20M"""

with open('milanko_data.csv') as f:
    csv_data = csv.reader(f)
    n, u_k, b_k, phi_k = [],[],[],[]
    next(csv_data) # Skip first row
    for line in csv_data:
        n.append(int(line[0]))
        u_k.append(float(line[1]))
        b_k.append(float(line[2]))
        phi_k.append(float(line[3]))

z_t = lambda t,num: sum(cm.rect(b_k[i],u_k[i]*t+phi_k[i]) for i in range(num)).real
t_span = np.linspace(-11,1,1000)

plt.plot(t_span, [z_t(t,len(n)) for t in t_span])
#plt.plot(t_span, [z_t(t,3) for t in t_span])
plt.show()
