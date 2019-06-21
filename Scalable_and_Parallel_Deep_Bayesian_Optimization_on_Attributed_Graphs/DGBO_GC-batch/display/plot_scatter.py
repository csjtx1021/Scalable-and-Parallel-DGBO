#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 20 12:06:21 2017

@author: cuijiaxu
"""

import numpy as np
import pylab as pl

def plot_scatter(data,ax,color,marker,alpha=0.2):
    x=data[:,0]
    y=data[:,1]
    return ax.scatter(x,y,c = color,marker = marker,edgecolors='none',alpha=alpha)

def plot_scatter_with_trendline(data,ax,color1,marker,alpha=0.2,color2='k'):
    x=data[:,0]
    y=data[:,1]
    a1=ax.scatter(x,y,c = color1,marker = marker,edgecolors='none',alpha=alpha)
    #with the trendline (it is simply a linear fitting)
    z = np.polyfit(x, y, 1)
    p = np.poly1d(z)
    a2,=ax.plot(x,p(x),color=color2,linestyle='--')
    return a1,a2

