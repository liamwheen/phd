import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib import image
from matplotlib import rc
rc('text', usetex=True)
rc('font', family='serif',size=14.7)

earth_map = image.imread('../earth_map_clean.png')
earth_map = 1-earth_map[:,200:]
fig, axs = plt.subplots(1,2, gridspec_kw={'width_ratios':(1,6)},figsize=(13/1.5,4))
axs[1].imshow(1-earth_map,'binary',alpha=0.4)
sums = np.sum(earth_map,1)/earth_map.shape[1]
axs[0].plot(sums[:,0],np.linspace(0,len(np.sum(earth_map,1)),len(np.sum(earth_map,1))),linewidth=3)
axs[0].set_yticks(np.linspace(0,earth_map.shape[0],7))
axs[0].set_ylim([earth_map.shape[0],0])
axs[1].set_ylim([earth_map.shape[0],0])
axs[0].set_yticklabels(['{}$^\circ$'.format(i) for i in np.arange(90,-100,-30)])
axs[0].margins(x=0)
axs[0].set_xticks([])
axs[1].set_yticks([])
axs[1].set_xticks([])
axs[0].set_xlabel('Land Density')
axs[0].set_ylabel('Latitude')
plt.tight_layout()
plt.savefig('../land_distribution.pdf',dpi=1000)
#plt.show()
