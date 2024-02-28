#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 27 22:35:11 2024
Test rBergomi American option pricing
@author: lucapelizzari
"""

import numpy as np
import scipy as sc
from American_Option_Pricing_rBergomi import LongstaffSchwartz_signature_rBergomi, DualSAA_signature_rBergomi

M = 5000
M2 = 5000
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
h = 0.07

#Signature and polynomial feature level for primal
K_signature = 3
K_polynomial = 3

#Signature and polynomial feature level for dual
KD_signature = 3
KD_polynomial = 3


xi = 0.09
eta = 1.9
r =0.05



#%%

#lower bound outputs
y0_lo = np.zeros((len(phi),len(N)))
STD_lo = np.zeros((len(phi),len(N)))
timee_lo = np.zeros(len(N))

#upper bound outputs
y0_up = np.zeros((len(phi),len(N)))
STD_up = np.zeros((len(phi),len(N)))
timee_up = np.zeros(len(N))
for KK in range(len(N)):
    y0_lo[:,KK],STD_lo[:,KK],timee_lo[KK] = LongstaffSchwartz_signature_rBergomi(M,M2,N[KK],N1,T,phi,rho,K_signature,K_polynomial,X0,h,xi,eta,r)
    y0_up[:,KK],STD_up[:,KK],timee_up[KK] = DualSAA_signature_rBergomi(M,M2,N[KK],N1,T,phi,rho,KD_signature,KD_polynomial,X0,h,xi,eta,r)







