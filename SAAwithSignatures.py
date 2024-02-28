#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 22 16:20:30 2023

@author: pelizzari
Dual optimal stopping with Signatures, SAA method
"""

import numpy as np
from AmericanOptionsWithSignaturesDualSIgnature import SimulationofrBergomi
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
import scipy as sc
from sig_regression_terminal_LP import full_log_signature,signature,signatureQV
from fBmaugmented import dualLPsparse1, dualLPsparsePY
import time
#Longstaff-Schwarz with Signatures

def DualSAASignature(M,M2,N,N1,T,phi,rho,K,KK_dual,X0,H,xi,eta,r,e):
    """Compute upper bounds for Bermuddan option price with N1 equally spased exercise dates between 0 and T.
    M,M2 = number of paths for LP, respectively resimulation for upper-bounds
    N = time-discretization for Signature
    N1 = exercise dates, N1 <= N
    T = maturity
    e = estimate of price for randomization
    phi = array of payoff functions (i.e. different strikes)
    K = level of Signature (tensor)
    KK = number of state-polynomials added to basis-function
    X0,rho,xi,eta,r,H = parameters for rBergomi
    Output: array of (true) upper bounds for each payoff-function"""
    s = time.time()
    #number of strikes
    N_strikes = len(phi)
    #Step 1: First simulation of rBergomi.
    X,V,I,dI,dW1,dW2,dB = SimulationofrBergomi(M,N,T,phi,rho,K,X0,H,xi,eta,r)
    #exercise-dates with and without zero
    subindex = [int((j+1)*N/N1) for j in range(N1)]
    subindex2 = [int((j)*N/N1) for j in range(N1+1)]
    tt = np.linspace(0,T,N+1)
    #Payoff-process on finer grid
    Z = np.zeros((M,N+1,N_strikes))
    for k in range((N_strikes)):
        Z[:,:,k]= np.exp(-r*tt)*phi[k](X)
    #compute signature of (QV,X)
    D = int((1-(2+1)**(K+1))/(-2) -1)
    #Signatures for each payoff
    SIG = np.zeros((M,N+1,D,N_strikes))
    dX=X[:,1:N+1]-X[:,0:N]
    QV = np.zeros(shape = (M,N+1))
    for n in range(N):
        QV[:,n+1] = QV[:,n]+V[:,n]*1/(N+1)
    for k in range(N_strikes):
        dY = Z[:,1:N+1,k]-Z[:,0:N,k]
        dXX = np.zeros((M,N,2))
        dXX[:,:,0]=dY
        dXX[:,:,1] = dX
        SIG[:,1:N+1,:,k] = signatureQV(tt,dXX,QV,K)
    DD_dual = int((KK_dual+1)*(KK_dual+2)/2) #Number of polynomials 2 dim
    P_dual= np.zeros((M,N+1,DD_dual))
    for k in range(KK_dual+1):
        for j in range(0,k+1):
            C = np.zeros((KK_dual+1,KK_dual+1))
            C[k,j] = 1
            P_dual[:,:,int(k*(k+1)/2+j)] = np.polynomial.laguerre.lagval2d(X, np.sqrt(V), C)
    xx = np.zeros((2*(DD_dual+D),N_strikes))
    yy = np.zeros((2*(DD_dual+D),N_strikes))
    for st in range(N_strikes):
        Basis_dual = np.ones((M,N+1,DD_dual+D))
        Basis_dual[:,:,0:D]=SIG[:,:,:,st]
        Basis_dual[:,:,D:DD_dual+D] = P_dual
        MG = np.zeros((M,N+1,2*(D+DD_dual)))
        U = np.random.uniform(0,2*e[st],M)
        L = np.zeros((M,N+1))
        L[:,1:N+1] = Z[:,1:N+1,st]
        L[:,0] = np.abs(U)
        print(D+DD_dual)
        for dd in range(D+DD_dual):
            for k in range(N):
                MG[:,k+1,dd] = MG[:,k,dd] + rho*X[:,k]*np.sqrt(V[:,k])*Basis_dual[:,k,dd]*dW1[:,k,0]
                MG[:,k+1,dd+DD_dual+D] = MG[:,k,dd+DD_dual+D] + np.sqrt(1-rho**2)*X[:,k]*np.sqrt(V[:,k])*Basis_dual[:,k,dd]*dW2[:,k]
        #xx[:,st] = dualLPsparse1(L,tt,N1,N,D,M,MG[:,subindex2,:],subindex2)
        xx[:,st] = dualLPsparse1(Z[:,:,st],tt,N1,N,D,M,MG[:,subindex2,:],subindex2)
        print('Estimator dual for Payoff number',st,'is', np.mean(np.max(Z[:,subindex,st]-np.dot(MG,xx[:,st])[:,subindex],axis=1)),'with standard deviation',np.std(np.max(Z[:,subindex,st]-np.dot(MG,xx[:,st])[:,subindex],axis=1)))
        #print('Estimator dual for Payoff number with PYTHON',st,'is', np.mean(np.max(Z[:,subindex,st]-np.dot(MG,yy[:,st])[:,subindex],axis=1)),'with standard deviation',np.std(np.max(Z[:,subindex,st]-np.dot(MG,yy[:,st])[:,subindex],axis=1)))
    del X,V,I,dI,dW1,dW2,dB,Basis_dual,P_dual,MG,SIG,QV,dX
    #Resimulation
    X,V,I,dI,dW1,dW2,dB = SimulationofrBergomi(M2,N,T,phi,rho,K,X0,H,xi,eta,r)
    
    Z = np.zeros((M2,N+1,N_strikes))
    for k in range((N_strikes)):
        Z[:,:,k]= np.exp(-r*tt)*phi[k](X)
    #compute signature of (QV,X)
    D = int((1-(2+1)**(K+1))/(-2) -1)
    #Signatures for each payoff
    SIG = np.zeros((M2,N+1,D,N_strikes))
    dX=X[:,1:N+1]-X[:,0:N]
    QV = np.zeros(shape = (M2,N+1))
    for n in range(N):
        QV[:,n+1] = QV[:,n]+V[:,n]*1/(N+1)
    for k in range(N_strikes):
        dY = Z[:,1:N+1,k]-Z[:,0:N,k]
        dXX = np.zeros((M2,N,2))
        dXX[:,:,0]=dY
        dXX[:,:,1] = dX
        SIG[:,1:N+1,:,k] = signatureQV(tt,dXX,QV,K)
    DD_dual = int((KK_dual+1)*(KK_dual+2)/2) #Number of polynomials 2 dim
    P_dual= np.zeros((M2,N+1,DD_dual))
    for k in range(KK_dual+1):
        for j in range(0,k+1):
            C = np.zeros((KK_dual+1,KK_dual+1))
            C[k,j] = 1
            P_dual[:,:,int(k*(k+1)/2+j)] = np.polynomial.laguerre.lagval2d(X, np.sqrt(V), C)
    y0 = np.zeros((N_strikes))
    MC = np.zeros((N_strikes))
    for st in range(N_strikes):
        Basis_dual = np.ones((M2,N+1,DD_dual+D))
        Basis_dual[:,:,0:D]=SIG[:,:,:,st]
        Basis_dual[:,:,D:DD_dual+D] = P_dual
        MG = np.zeros((M2,N+1,2*(D+DD_dual)))
        
        for dd in range(D+DD_dual):
            for k in range(N):
                MG[:,k+1,dd] = MG[:,k,dd] + rho*X[:,k]*np.sqrt(V[:,k])*Basis_dual[:,k,dd]*dW1[:,k,0]
                MG[:,k+1,dd+DD_dual+D] = MG[:,k,dd+DD_dual+D] + np.sqrt(1-rho**2)*X[:,k]*np.sqrt(V[:,k])*Basis_dual[:,k,dd]*dW2[:,k]
        y0[st] = np.mean(np.max(Z[:,subindex,st]-np.dot(MG,xx[:,st])[:,subindex],axis=1))
        MC[st] = np.std(np.max(Z[:,subindex,st]-np.dot(MG,xx[:,st])[:,subindex],axis=1))
        print('Upper-biased price for Payoff number',st,'is',y0[st],'with standard deviation',MC[st])
        
    ss = time.time()
    timee = ss-s
    return y0, MC,timee
    

    