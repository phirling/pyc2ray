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


class zTomography_3panels:
    def __init__(self,datacube1,datacube2,datacube3,zi,fs=6):
        self.data1 = datacube1
        self.data2 = datacube2
        self.data3 = datacube3
        self.N = datacube1.shape[0]
        self.zz = zi
        self.cmap = mpl.colormaps['inferno']
        self.cmap.set_bad('white')

        vmax = max(np.nanmax(datacube1),np.nanmax(datacube2))
        vmin = max(np.nanmin(datacube1),np.nanmin(datacube2))

        vmax_res = np.nanmax(datacube3)
        vmin_res = np.nanmin(datacube3)

        self.fig, (self.ax1,self.ax2,self.ax3) = plt.subplots(1,3,figsize=(3*fs,fs))

        # Panel 1
        self.ax1.set_title(f"Ionization Rate, C2Ray",fontsize=12)
        self.im1 = self.ax1.imshow(self.data1[:,:,zi],origin='lower',cmap=self.cmap,vmax=vmax,vmin=vmin)
        self.cb1 = plt.colorbar(self.im1,ax=self.ax1)
        self.cb1.set_label(label=r"$\log \Gamma_1$ [s$^{-1}$]",size=15)

        # Panel 2
        self.ax2.set_title(f"Ionization Rate, OCTA GPU",fontsize=12)
        self.im2 = self.ax2.imshow(self.data2[:,:,zi],origin='lower',cmap=self.cmap,vmax=vmax,vmin=vmin)
        self.cb2 = plt.colorbar(self.im2,ax=self.ax2)
        self.cb2.set_label(label=r"$\log \Gamma_2$ [s$^{-1}$]",size=15)

        # Panel 3
        self.ax3.set_title(f"Residual",fontsize=12)
        #self.im3 = self.ax3.imshow(self.data3[:,:,zi],origin='lower',cmap='bwr',vmax=1,vmin=-1)
        self.im3 = self.ax3.imshow(self.data3[:,:,zi],origin='lower',cmap='bwr',vmax=vmax_res,vmin=vmin_res,norm='symlog')
        self.cb3 = plt.colorbar(self.im3,ax=self.ax3)
        self.cb3.set_label(label=r"$\Gamma_1 / \Gamma_2 - 1$",size=15)

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
                self.im1.set_data(self.data1[:,:,zz])
                self.im2.set_data(self.data2[:,:,zz])
                self.im3.set_data(self.data3[:,:,zz])
                self.zz = zz
                self.fig.canvas.draw()

class zTomography_xfrac:
    def __init__(self,datacube,zi,incr=10,xmin=None,cmap='jet',fs=6):
        self.data = datacube
        self.N = datacube.shape[0]
        self.fig, self.ax = plt.subplots(figsize=(fs,fs))
        self.zz = zi
        self.incr = incr
        self.ax.set_title(f"Neutral Hydrogen Fraction",fontsize=12)
        self.cmap = mpl.colormaps[cmap]
        self.cmap.set_bad('white')
        if xmin is None:
            vmin = np.min(1.0-datacube)
        else: vmin = xmin
        self.im = self.ax.imshow(1.0-self.data[:,:,zi],origin='lower',norm='log',cmap=self.cmap,vmax=1.0,vmin=vmin)
        self.cb = plt.colorbar(self.im,ax=self.ax)
        self.cb.set_label(label=r"$1-x_{HI}$",size=15)
        self.fig.canvas.mpl_connect('key_press_event',self.switch)
        self.fig.tight_layout()

    def switch(self,event):
        up = event.key == 'up'
        down = event.key == 'down'
        zz = self.zz
        if up:
            zz += self.incr
        elif down:      
            zz -= self.incr
        if up or down:
            if zz in range(self.N):
                self.im.set_data(1.0-self.data[:,:,zz])
                self.zz = zz
                self.fig.canvas.draw()