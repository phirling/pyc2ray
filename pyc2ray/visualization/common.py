import matplotlib.pyplot as plt

def xfrac_plot(data,ax,xmin=None,xmax=1.0,cmap='jet',fs=10,boxsize=None,time=None):
    if boxsize is None:
        unitstr = "[Grid Coordinates]"
        ext = None
    else:
        unitstr = "[kpc]"
        ext = (0,boxsize,0,boxsize)
    im = ax.imshow(data,origin='lower',norm='log',cmap=cmap,vmin=xmin,vmax=xmax,extent=ext)
    cb = plt.colorbar(im,ax=ax)
    cb.set_label(label=r"$x_{HI}$",size=1.5*fs)
    ax.set_xlabel("$x$ " + unitstr,fontsize=fs)
    ax.set_ylabel("$y$ " + unitstr,fontsize=fs)
    if time is not None:
        ttl = ax.text(0.01, 0.99,"$t={:.2f}$ Myr".format(time),
        horizontalalignment='left',
        verticalalignment='top',
        color = 'white',
        transform = ax.transAxes)

    return im, cb