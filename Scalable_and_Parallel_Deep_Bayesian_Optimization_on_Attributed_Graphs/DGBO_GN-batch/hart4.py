#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 18 12:01:30 2017

@author: cuijiaxu
"""
import numpy as np

def hart4(xx):
##########################################################################
# HARTMANN 4-DIMENSIONAL FUNCTION
# xx = [x1, x2, x3, x4]
##########################################################################
    alpha = np.array((1.0, 1.2, 3.0, 3.2)).reshape(4,1)
    A = np.array((10, 3, 17, 3.5, 1.7, 8,0.05, 10, 17, 0.1, 8, 14,3, 3.5, 1.7, 10, 17, 8,17, 8, 0.05, 10, 0.1, 14)).reshape(4,6)
    P = np.array((1312, 1696, 5569, 124, 8283, 5886,2329, 4135, 8307, 3736, 1004, 9991,2348, 1451, 3522, 2883, 3047, 6650,4047, 8828, 8732, 5743, 1091, 381)).reshape(4,6)
    P = P*1e-4
    outer = 0
    for ii in range(4):
        inner = 0
        for jj in range(4):
            xj = xx[jj]
            Aij = A[ii, jj]
            Pij = P[ii, jj]
            inner = inner + Aij*np.square(xj-Pij)
        new = alpha[ii] * np.exp(-inner)   
        outer = outer + new
    y = (1.1 - outer) / 0.839
    return y
            
            
            