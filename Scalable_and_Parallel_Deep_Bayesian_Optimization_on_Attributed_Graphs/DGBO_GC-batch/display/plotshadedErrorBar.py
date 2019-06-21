import numpy as np
import scipy.io
import scipy.stats
import pylab as pl
from matplotlib.patches import Polygon
from matplotlib import colors

def plotshadedErrorBar(x,mean_,std_,ax,linecolor,linestyle):
    
    mainline,=pl.plot(x[0], mean_[0], '%s%s'%(linecolor,linestyle), linewidth=1)
    
    # Make the shaded region
    ix = np.array(x[0])
    main = np.array(mean_[0])
    upstd = np.array(mean_[0])+np.array(std_[0])
    downstd = np.array(mean_[0])-np.array(std_[0])
    #   print ix,main,upstd,downstd
    verts = list(zip(ix, main)) + list(zip(ix, upstd))[::-1]
    poly = Polygon(verts, facecolor=np.array(list(colors.to_rgb('%s'%linecolor)))+(1-np.array(list(colors.to_rgb('%s'%linecolor))))*0.65, edgecolor='none',alpha=0.7)
    ax.add_patch(poly)
    verts = list(zip(ix, main)) + list(zip(ix, downstd))[::-1]
    poly = Polygon(verts, facecolor=np.array(list(colors.to_rgb('%s'%linecolor)))+(1-np.array(list(colors.to_rgb('%s'%linecolor))))*0.65, edgecolor='none',alpha=0.7)
    ax.add_patch(poly)
    # pl.show()
    return mainline

def plotshadedErrorBar2(x,mean_,up,down,ax,linecolor,linestyle):
    
    mainline,=pl.plot(x[0], mean_[0], '%s%s'%(linecolor,linestyle), linewidth=1)
    
    # Make the shaded region
    ix = np.array(x[0])
    main = np.array(mean_[0])
    upstd = np.array(up[0])
    downstd = np.array(down[0])
    #   print ix,main,upstd,downstd
    verts = list(zip(ix, main)) + list(zip(ix, upstd))[::-1]
    poly = Polygon(verts, facecolor=np.array(list(colors.to_rgb('%s'%linecolor)))+(1-np.array(list(colors.to_rgb('%s'%linecolor))))*0.65, edgecolor='none',alpha=0.7)
    ax.add_patch(poly)
    verts = list(zip(ix, main)) + list(zip(ix, downstd))[::-1]
    poly = Polygon(verts, facecolor=np.array(list(colors.to_rgb('%s'%linecolor)))+(1-np.array(list(colors.to_rgb('%s'%linecolor))))*0.65, edgecolor='none',alpha=0.7)
    ax.add_patch(poly)
    # pl.show()
    return mainline
