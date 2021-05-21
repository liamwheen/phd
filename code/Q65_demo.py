import numpy as np
import matplotlib.pyplot as plt
from data import milanko_params
from scipy.interpolate import interp1d

def norm(x):
    y = x - np.mean(x)
    return y/np.std(y)

""" 
https://www.ncdc.noaa.gov/abrupt-climate-change/Glacial-Interglacial%20Cycles

This shows how the Q65 insolation, which has period 23k (as precession),
correlates with the jump in co2/benthic when it coincides with increasing
eccentricity. This might mean the budyko model requires sub-year resolution as
the temperature in summer is relating to the initiation of the inter-glacial."""

N_insol = np.loadtxt('data/NQ65.csv',delimiter=',')
S_insol = np.loadtxt('data/SQ65.csv',delimiter=',')
t, ecc, obliq, l_peri = milanko_params.load_milanko('backward')
rho = (3/2*np.pi - l_peri)%(2*np.pi)
benth = np.loadtxt('data/benthic.csv',delimiter=',') #https://lorraine-lisiecki.com/LisieckiRaymo2005.pdf
temp = np.loadtxt('data/delta_temp.csv',delimiter=',') #https://www.nature.com/articles/nature06015#rightslink
carbon = np.loadtxt('data/co2.csv',delimiter=',') #https://www.nature.com/articles/nature06949
ecc_fun = interp1d(t,ecc)
rho_fun = interp1d(t,rho)
obliq_fun = interp1d(t,obliq)
ben_fun = interp1d(benth[:,0],benth[:,1])
carbon_func = interp1d(carbon[:,0],carbon[:,1])

plt.figure(figsize=(9,3))
plt.plot(np.linspace(-3e3,0,3000),norm(N_insol))
plt.plot(np.linspace(-3e3,0,3000),np.roll(norm(ecc_fun(np.linspace(-3e3,0,3000))),0))
plt.plot(np.linspace(-3e3,0,3000),norm(-ben_fun(np.linspace(-3e3,0,3000))),linewidth=1)
plt.plot(np.linspace(-8e2,0,1000),norm(carbon_func(np.linspace(-8e5,0,1000))),linewidth=1)
plt.xlim([-1000,0])
plt.ylim([-7,7])

"""
This shows how the summer soltices in each hemisphere are essentially mirrored
due to how precession places them around the orbit, they both then trend
according to obliquity when taking the average"""
plt.figure(figsize=(9,3))
plt.plot(norm(N_insol),linewidth=1)
plt.plot(norm(S_insol),linewidth=1)
plt.plot(norm((N_insol+S_insol)/2))
plt.plot(norm(obliq_fun(np.linspace(-3e3,0,3000))),'--')

"""
This shows how the difference between the summer solstices in the two
hemispheres trend with eccentricity, this only really applies when the summers
occur at peri/aphelion. If ecc is low, then they be similar,
despite their summers occuring at opposite sides of the orbit, if it is high,
then they are more dissimilar"""
plt.figure(figsize=(9,3))
plt.plot(norm(ecc_fun(np.linspace(-3e3,0,3000))))
plt.plot(norm(abs(norm(N_insol)-norm(S_insol))),linewidth=1)

plt.show()
