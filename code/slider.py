import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, RadioButtons

fig, ax = plt.subplots(figsize=(5,7))
plt.subplots_adjust(bottom=0.45)
n = np.arange(-5, 2, 0.001)
Q = 340.375
A = 202.1
B = 1.9
C = 3.04
beta = 0.4090928
cb = 5/16*(3*np.sin(beta)**2 - 2)
Tice = -10
alpha1 = 0.32
alpha2 = 0.62
#alpha0 = (alpha1+alpha2)/2

a0 = Q*(1-1/2*cb)*(1-(alpha1+alpha2)/2) - A*(1+C/B) + C*Q/B*(1-alpha2) - (B+C)*Tice
a1 = C*Q/B*(alpha2-alpha1)*(1-1/2*cb)
a2 = 3/2*Q*cb*(1-(alpha1+alpha2)/2)
a3 = C*Q*cb/(2*B)*(alpha2-alpha1)
ice_eq = lambda n:a3*n**3 + a2*n**2 + a1*n + a0
fn = ice_eq(n) 
l, = plt.plot(n, fn, lw=2)
plt.plot([-5,2],[0,0],'k--',linewidth=1)
ax.margins(x=0)

axcolor = 'lightgoldenrodyellow'
axA = plt.axes([0.15, 0.06, 0.65, 0.03], facecolor=axcolor)
axC = plt.axes([0.15, 0.12, 0.65, 0.03], facecolor=axcolor)
axcb = plt.axes([0.15, 0.18, 0.65, 0.03], facecolor=axcolor)
axTice = plt.axes([0.15, 0.24, 0.65, 0.03], facecolor=axcolor)
axalpha1 = plt.axes([0.15, 0.30, 0.65, 0.03], facecolor=axcolor)
axalpha2 = plt.axes([0.15, 0.36, 0.65, 0.03], facecolor=axcolor)

sA = Slider(axA, 'A', 150, 300, valinit=A)
sC = Slider(axC, 'C', 1, 5, valinit=C)
scb = Slider(axcb, 'cb', -0.5, -0.46, valinit=cb)
sTice= Slider(axTice, 'Tice', -20, 0, valinit=Tice)
salpha1= Slider(axalpha1, 'alpha1', 0, 1, valinit=alpha1)
salpha2= Slider(axalpha2, 'alpha2', 0, 1, valinit=alpha2)

def update(val):
    A = sA.val 
    C = sC.val 
    cb= scb.val 
    Tice = sTice.val 
    alpha1 = salpha1.val
    alpha2 = salpha2.val
    a0 = Q*(1-1/2*cb)*(1-(alpha1+alpha2)/2) - A*(1+C/B) + C*Q/B*(1-alpha2) - (B+C)*Tice
    a1 = C*Q/B*(alpha2-alpha1)*(1-1/2*cb)
    a2 = 3/2*Q*cb*(1-(alpha1+alpha2)/2)
    a3 = C*Q*cb/(2*B)*(alpha2-alpha1)
    ice_eq = lambda n:a3*n**3 + a2*n**2 + a1*n + a0
    fn = ice_eq(n) 
    l.set_ydata(fn)
    fig.canvas.draw_idle()


sA.on_changed(update)
sC.on_changed(update)
scb.on_changed(update)
sTice.on_changed(update)
salpha1.on_changed(update)
salpha2.on_changed(update)

resetax = plt.axes([0.8, 0.9, 0.1, 0.04])
button = Button(resetax, 'Reset', color=axcolor, hovercolor='0.975')


def reset(event):
    sA.reset()
    sC.reset()
    scb.reset()
    sTice.reset()
    salpha1.reset()
    salpha2.reset()
button.on_clicked(reset)

#rax = plt.axes([0.025, 0.5, 0.15, 0.15], facecolor=axcolor)
#radio = RadioButtons(rax, ('red', 'blue', 'green'), active=0)


#def colorfunc(label):
#    l.set_color(label)
#    fig.canvas.draw_idle()
#radio.on_clicked(colorfunc)
ax.set_xlim([0,1])
ax.set_ylim([-50,50])
plt.show()
