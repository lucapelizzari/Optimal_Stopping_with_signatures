#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 27 19:15:45 2024

Simulation of rBergomi model, see "Pricing under rough volatility"
We use the simulation package https://github.com/ryanmccrickerd/rough_bergomi

@author: lucapelizzari
"""
import numpy as np
from FBM_package import FBM
import scipy as sc
from rBergomi import rBergomi



def SimulationofrBergomi(M,N,T,phi,rho,K,X0,H,xi,eta,r):
    """Simulate paths of rBergomi price, volatility, Brownian motions and I
    M = Number of samples in first simulation, used for LS-regression
    M2 = Number of samples in Resimulation, typically much larger than M
    N = Number of discretization points for the grid [0,T]
    N1 = Number of exercise points in [0,T] in the sense of Bermudda-options
    T = maturiy
    phi = payoff functions (Put or Call)
    rho = correlation coefficient
    K = depth of Signature
    X0 = log(S0) initial value of log-price
    H = Hurst-parameter for fBm in rBergomi model
    xi,eta = specifications for rBergomi volatility process
    r = interest-rate 
    
    Output:
        X = stock-price in rBergomi model
    """
    tt = np.linspace(0,T,N+1)
    #Using rBergomi-Package for volatility and Brownian motions
    rB = rBergomi(N, M, T, -0.5+H)
    #two independent Brownian motion increments
    dW1 = rB.dW1()
    dW2 = rB.dW2()
    #volatility process V,array of Mx(N+1)
    Y = rB.Y(dW1)
    V = rB.V(Y, xi, eta)
    #price-process in rBergomi
    dB = rB.dB(dW1, dW2, rho)
    X = rB.S(V, dB) #array of Mx(N+1)
    X = X0*X*np.exp(r*tt)
    #print('European:',np.mean(np.exp(-r)*phi(X[:,-1])), 'MC-Error:', np.std(np.exp(-r)*phi(X[:,-1]))/np.sqrt(M))
    I = np.zeros(shape = (M,int(T*N)+1))
    for n in range(int(T*N)):
        I[:,n+1] = I[:,n] + np.sqrt(V[:,n])*dW1[:,n,0]
    dI = I[:,1:int(T*N)+1]-I[:,0:int(T*N)]
    dI = dI.reshape(M,int(T*N),1)
    return X,V,I,dI,dW1,dW2,dB