import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib import image
from matplotlib import rc
rc('text', usetex=True)
rc('font', family='serif',size=13)

earth_map = image.imread('../gall.png')
earth_map[(earth_map<1)] = 0
fig, axs = plt.subplots(1,2, gridspec_kw={'width_ratios':(1,6)},figsize=(7,4))
axs[1].imshow(1-earth_map,'binary',alpha=0.4)
sums = np.sum(earth_map,1)/earth_map.shape[1]
sums[-2:] = sums[-3] #fix antarctic tailing off at end
axs[0].plot(sums[::-1,0],np.linspace(-90,90,len(sums[:,0])),linewidth=3)
axs[0].set_yticks(90*np.sin(np.array([-90,-45,-20,0,20,45,90])*np.pi/180))
axs[0].set_yticklabels(['{}$^\circ$'.format(i) for i in [-90,-45,-20,0,20,45,90]])
axs[0].set_ylim((-90,90))
axs[1].set_ylim([earth_map.shape[0],0])
axs[0].margins(x=0)
axs[0].set_xticks([0,1])
axs[1].set_yticks([])
axs[1].set_xticks([])
axs[0].set_xlabel('Land Proportion')
axs[0].set_ylabel('Latitude')
plt.tight_layout()
plt.savefig('../land_distribution.pdf',dpi=1000)
#plt.show()
