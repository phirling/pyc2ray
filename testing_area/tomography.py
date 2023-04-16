import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

class zTomography:
    def __init__(self,datacube,zi,fs=6):
        self.data = datacube
        self.N = datacube.shape[0]
        self.fig, self.ax = plt.subplots(figsize=(fs,fs))
        self.zz = zi
        self.ax.set_title(f"Ionization Rate",fontsize=12)
        self.cmap = mpl.colormaps['inferno']
        self.cmap.set_bad('white')
        self.im = self.ax.imshow(self.data[:,:,zi],origin='lower',cmap=self.cmap,vmax=np.nanmax(datacube),vmin=np.nanmin(datacube))
        self.cb = plt.colorbar(self.im,ax=self.ax)
        self.cb.set_label(label=r"$\log \Gamma$ [s$^{-1}$]",size=15)
        self.fig.canvas.mpl_connect('key_press_event',self.switch)
        self.fig.tight_layout()

    def switch(self,event):
        up = event.key == 'up'
        down = event.key == 'down'
        zz = self.zz
        if up:
            zz += 10
        elif down:      
            zz -= 10
        if up or down:
            if zz in range(self.N):
                self.im.set_data(self.data[:,:,zz])
                self.zz = zz
                self.fig.canvas.draw()