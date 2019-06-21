#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 20 12:06:21 2017

@author: cuijiaxu
"""

import numpy as np

def load(filename=""):
    data=np.loadtxt(filename)
    data[:,0]=np.linspace(1,data.shape[0],data.shape[0])
    return data

