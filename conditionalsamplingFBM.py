#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 18 10:25:52 2023

@author: lucapelizzari

conditional sampling using formulas from Nuzmann&Poor for estimation of fBm, and translate it to Louiville fBM later
"""
import numpy as np
from FBM_LP_1 import FBM
import scipy as sc
from eulermaruyama import eulermaruyama
from sig_regression_terminal_LP import signature,signatureQV
from regression import Regression
import matplotlib.pyplot as plt
import time


def linearpredictionFBM(t,s,dF,T,n,H):
    """Compute linear prediction of fractional Brownian motion,E_t[W^H_s] using explicit representation of Nuzmann and Poor
    t,s = time points in [0,T] with t<s
    dfBm = increments of fractional Brownian motion
    N = discretization
    n = number of weight for quadrature
    Output: W_t^H + integral, where the integral is computed using Gauss-Jacobbi quadrature"""
    #compute fBm and interpolate in time
    M,N = dF.shape
    fBm = np.zeros((M,N+1))
    fBm[:,1:N+1]= np.cumsum(dF,axis=1)
    tt = np.linspace(0,T,N+1)
    fBM = sc.interpolate.interp1d(tt,fBm)
    c= s/t-1
    
    if H <0.5:
        #weights for gauss-jacobbi quadrature
        [w,u] = sc.special.roots_jacobi(n, -(H+0.5), -(H+0.5))
        ss = t/2+t/2*w
        #print(ss)
        a = H+0.5
        b = 1-2*H
        def f(u):
            #return t**(2*H)*((0.5-H)*sc.special.betainc(a,b,c/(c+1))*sc.special.gamma(a)*sc.special.gamma(b)/sc.special.gamma(a+b)+c**(H+0.5)*(1+c)**(H-0.5)*(1-u/t)/(c+u/t))
            return (fBM(t-u))*((0.5-H)*beta(a,b,c/(c+1))+(c**(H+0.5)*(1+c)**(H-0.5)*(1-u/t)/(c+u/t)))
            #return (fBM(t-u)-fBM(t))*((0.5-H)*beta(a,b,c/(c+1))+(c**(H+0.5)*(1+c)**(H-0.5)*(1-u/t)/(c+u/t)))
        I = 0
        for k in range(len(ss)):
            #print(f(ss[k]))
            I = I + f(ss[k])*u[k]
        
        #I = np.cos(np.pi*H)/np.pi*(s-t)**(H+1/2)*(t/2)**(1/2-H)*I
        #return FBM(t)+I
        #return fBM(t)+np.cos(np.pi*H)/np.pi*I*(0.5)**(-2*H)
        return np.cos(np.pi*H)/np.pi*I*(0.5)**(-2*H)
    if H == 0.5:
        return fBM(t)
    if H>0.5:
        def g(u,w):
            return (u+1)**(H-0.5)/(u+w/t)
        def I(r):
            J=0
            for j in range(len(w)):
                J = J + g(c/2*w[j]+c/2,r)*u[k]
            return J
        def M_tc(r):
            return np.cos(np.pi*H)/np.pi*t**(2*H-1)*I(r)
def kernelfBM(s,t,H):
    return (t-s)**(H-0.5)/sc.special.gamma(H+1/2)*sc.special.hyp2f1(H-0.5,0.5-H,H+0.5,1-t/s)
    
def conditionalcovariance(t,tt,i,j,H):
    """Compute E_t[(X_tt[j]-E_t[XX_tt[j])(X_tt[j]-E_t[X_tt[j]])]"""
    def g(u):
        return kernelfBM(u,tt[j],H)*kernelfBM(u,tt[i],H)
    [I,u] = sc.integrate.quadrature(g, t, min(tt[j],tt[i]))
    return I
def discretelinearpredictionFBM(k,j,FBM,T,H,ttt,M):
    """Discrete formula for E_k[X^H_k] with 1 \leq k < j \leq N1"""
    #construct covariance matrices
    G = np.zeros((k,k))
    for u in range(k):
        for w in range(k):
            G[u,w] = 0.5*(ttt[u+1]**(2*H)+ttt[w+1]**(2*H)+np.abs(ttt[u+1]-ttt[w+1])**(2*H))
    L = np.zeros((M,k))
    for m in range(M):
        L[m,:] = 0.5*(ttt[int(j[m])]**(2*H)+ttt[1:k+1]**(2*H)+(ttt[int(j[m])]-ttt[1:k+1])**(2*H))
    Sigma = np.einsum('ml,ll -> ml',L,np.linalg.inv(G))
    return np.einsum('ml,ml->m',Sigma,FBM[:,0:k])

def beta(a,b,x):
    return sc.special.betainc(a,b,x)*sc.special.gamma(a)*sc.special.gamma(b)/sc.special.gamma(a+b)