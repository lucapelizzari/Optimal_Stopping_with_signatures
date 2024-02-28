#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 27 22:35:11 2024
Test rBergomi American option pricing
@author: lucapelizzari
"""

import numpy as np
import scipy as sc
from American_Option_Pricing_rBergomi import LongstaffSchwartzrBergomi

M = 50000
M2 = 50000
M3 = 500
#K_primal =3
K_dual = 3
N = [48]
N1 = 12
T = 1
strike = 80
X0 = 100
#put-option
def phi1(x):
    return np.maximum(70-x,0)
def phi2(x):
    return np.maximum(80-x,0)
def phi3(x):
    return np.maximum(90-x,0)
def phi4(x):
    return np.maximum(100-x,0)
def phi5(x):
    return np.maximum(110-x,0)
def phi6(x):
    return np.maximum(120-x,0)
#phi = [phi1,phi2,phi3,phi4,phi5,phi6]
phi = [phi3]
KK_dual = 3
rho = -0.9
#depth for Laguerre-polynomials of stat
depth_primal = 2
depth_dual =6
sig_depth = 3
h = 0.07
K_primal = 3
KK_primal = 6
xi = 0.09
eta = 1.9
r =0.05



#%%
y0 = np.zeros((len(phi),len(N)))
STD = np.zeros((len(phi),len(N)))
timee = np.zeros(len(N))
for KK in range(len(N)):
    y0[:,KK],STD[:,KK],timee[KK] = LongstaffSchwartzrBergomi(M,M2,N[KK],N1,T,phi,rho,K_primal,KK_primal,X0,h,xi,eta,r)







