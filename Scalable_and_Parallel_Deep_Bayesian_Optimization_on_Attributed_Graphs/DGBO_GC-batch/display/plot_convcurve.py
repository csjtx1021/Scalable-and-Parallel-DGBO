#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 20 12:06:21 2017

@author: cuijiaxu
"""

import numpy as np
import scipy.io
import scipy.stats
import pylab as pl
from plotshadedErrorBar import plotshadedErrorBar2
from matplotlib.ticker import FuncFormatter

def getconvcurve(data):
    curve=np.zeros(data.shape[0])
    for i in range(len(curve)):
        curve[i]=max(data[0:i+1,1])
    return curve

def plot_one_convcurve(data,ax,color):
    a,=ax.plot(data[:,0],getconvcurve(data),color = color)
    return a

def plot_multi_convcurves(data_list,max_x,ax,color,linestyle,max_y=None):
    data_list_truncated=[]
    for data in data_list:
        data1=np.zeros(2*max_x).reshape(max_x,2)
        if data.shape[0]<max_x:
            data1[:,1]=np.array(list(data[:,1])+[max(data[:,1]) for i in range(max_x-len(data[:,1]))])
        else:
            data1[:,1]=data[0:max_x,1]
        data_list_truncated.append(data1)
    curves=[]
    for data_ in data_list_truncated:
        #print data_.shape
        curves.append(getconvcurve(data_))
    curves=np.array(curves)
    dis_mean=np.mean(curves,axis=0)
    dis_var=2.0*np.std(curves,axis=0)
    #print dis_mean.shape,dis_var.shape
    if max_y==None:
        return plotshadedErrorBar(np.linspace(1,len(dis_mean),len(dis_mean)).reshape(1,len(dis_mean)),dis_mean.reshape(1,len(dis_mean)),dis_var.reshape(1,len(dis_mean)),ax,color,linestyle)
    else:
        dis_up=dis_mean+dis_var
        dis_up[dis_up>max_y]=max_y
        dis_down=dis_mean-dis_var
        
        return plotshadedErrorBar2(np.linspace(1,len(dis_mean),len(dis_mean)).reshape(1,len(dis_mean)),dis_mean.reshape(1,len(dis_mean)),dis_up.reshape(1,len(dis_mean)),dis_down.reshape(1,len(dis_mean)),ax,color,linestyle)


