#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 27 22:35:11 2024
Test rBergomi American option pricing
@author: lucapelizzari
"""

import numpy as np
import scipy as sc
from American_Option_Pricing_rBergomi import LongstaffSchwartz_signature_rBergomi
from American_Option_Pricing_rHeston import LongstaffSchwartz_signature_rHeston

M = 50000
M2 = 50000
N = [64]
N1 = 4
lam = 0.3
nu = 0.3
theta = 0.02
V_0 = 0.02
rho = -0.7
S_0 = 1
T=1
r = 0.06
X0= 100
strike = 105
def phi1(x):
    return np.maximum(strike-x,0)

phi = [phi1]
K = 3
KK_primal = 4
H = 0.01
#%%
y0 = np.zeros((len(phi),len(N)))
STD = np.zeros((len(phi),len(N)))
timee = np.zeros(len(N))
for KK in range(len(N)):
    y0[:,KK],STD[:,KK],timee[KK] = LongstaffSchwartz_signature_rHeston(M,M2,N[KK],N1,T,phi,rho,K,KK_primal,X0,H,lam,nu,theta,V_0,r,3)







