#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 22 11:08:27 2023

@author: lucapelizzari


"""

import numpy as np
from AmericanOptionsWithSignaturesDualSIgnature import SimulationofrBergomi
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
import scipy as sc
from sig_regression_terminal_LP import full_log_signature,signature,signatureQV
import time
#Longstaff-Schwarz with Signatures

def LongstaffSchwartzrBergomi(M,M2,N,N1,T,phi,rho,K,KK_primal,X0,H,xi,eta,r):
    """Compute lower bounds for Bermuddan option price with N1 equally spased exercise dates between 0 and T.
    M,M2 = number of paths for Regression, respectively resimulation for lower bounds
    N = time-discretization for Signature
    N1 = exercise dates, N1 <= N
    T = maturity
    phi = array of payoff functions (i.e. different strikes)
    K = level of Signature (tensor)
    KK = number of state-polynomials added to basis-function
    X0,rho,xi,eta,r,H = parameters for rBergomi
    
    Output: array of (true) lower bounds for each payoff-function
    """
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
    dX = X[:,1:N+1]-X[:,0:N]
    dX = dX.reshape(M,N,1)
    QV = np.zeros(shape = (M,N+1))
    for n in range(N):
        QV[:,n+1] = QV[:,n]+V[:,n]*1/(N+1)
        
    S = signatureQV(tt,dX,QV,K)
    D = len(S[0,0,:])
    SS = np.zeros((M,N+1,D))
    SS[:,1:N+1] = S
    
    #Compute Basis-functions (for each payoff)
    DD_primal = int((KK_primal+1)*(KK_primal+2)/2) #Number of polynomials 2 dim
    P_primal = np.zeros((M,N+1,DD_primal))
    for k in range(KK_primal+1):
        for j in range(0,k+1):
            C = np.zeros((KK_primal+1,KK_primal+1))
            C[k,j] = 1
            P_primal[:,:,int(k*(k+1)/2+j)] = np.polynomial.laguerre.lagval2d(X,np.sqrt(V), C)
    Basis_primal = np.ones((M,N+1,DD_primal+D+1,N_strikes))
    for k in range(N_strikes):
        Basis_primal[:,:,0:D,k]=SS
        Basis_primal[:,:,D:DD_primal+D,k] = P_primal
        Basis_primal[:,:,DD_primal+D,k] = phi[k](X)
    #discouting over one period
    dtt = np.exp(-r*T/(N1+1))
    regr = [0]*N_strikes
    #Perform Longstaff-Schwartz Regression for each payoff
    for k in range(N_strikes):
        Basis = Basis_primal[:,:,:,k]
        Basis_Reg = Basis[:,subindex,:]

        YY = phi[k](X[:,subindex])
        value = YY[:,-1]
        regr[k] = [0]*(N1-1)
        for j in reversed(range(1,N1)):
            ITM = [0]
            value = value*dtt
            for m1 in range(M):
                if YY[m1,j-1]>0:
                    ITM.append(m1)
            if len(ITM) == 0:
                continue
            else:

                regr[k][j-1] = LinearRegression().fit(Basis_Reg[ITM,j-1,:],value[ITM])
                print(regr[k][j-1].score(Basis_Reg[ITM,j-1,:],value[ITM]))
                reg = regr[k][j-1].predict(Basis_Reg[ITM,j-1,:])
                #print('RMSE',np.mean((reg - value[ITM])**2))
                for m in range(len(ITM)):
                    if reg[m] > YY[ITM[m],j-1]:
                        continue
                    else:
                        value[ITM[m]] = YY[ITM[m],j-1]
        print('Estimater LS for Payoff Number',k,'is',np.mean(value),'with standard-deviation',np.std(value))
    del X,V,I,dI,dW1,dW2,dB,Basis,Basis_Reg,Basis_primal,P_primal,S,SS,QV,dX
    #Start Resimulation
    X,V,I,dI,dW1,dW2,dB = SimulationofrBergomi(M2,N,T,phi,rho,K,X0,H,xi,eta,r)
    #exercise-dates with and without zero
    #Payoff-process on finer grid
    Z = np.zeros((M2,N+1,N_strikes))
    for k in range((N_strikes)):
        Z[:,:,k]= np.exp(-r*tt)*phi[k](X)
    #compute signature of (QV,X)
    dX = X[:,1:N+1]-X[:,0:N]
    dX = dX.reshape(M2,N,1)
    QV = np.zeros(shape = (M2,N+1))
    for n in range(N):
        QV[:,n+1] = QV[:,n]+V[:,n]*1/(N+1)
    S = signatureQV(tt,dX,QV,K)
    D = len(S[0,0,:])
    SS = np.zeros((M2,N+1,D))
    SS[:,1:N+1] = S
    ttt = np.linspace(0,T,N1+1)
    
    #Compute Basis-functions (for each payoff)
    DD_primal = int((KK_primal+1)*(KK_primal+2)/2) #Number of polynomials 2 dim
    P_primal = np.zeros((M2,N+1,DD_primal))
    for k in range(KK_primal+1):
        for j in range(0,k+1):
            C = np.zeros((KK_primal+1,KK_primal+1))
            C[k,j] = 1
            P_primal[:,:,int(k*(k+1)/2+j)] = np.polynomial.laguerre.lagval2d(X,np.sqrt(V), C)
    Basis_primal = np.ones((M2,N+1,DD_primal+D+1,N_strikes))
    for k in range(N_strikes):
        Basis_primal[:,:,0:D,k]=SS
        Basis_primal[:,:,D:DD_primal+D,k] = P_primal
        Basis_primal[:,:,DD_primal+D,k] = phi[k](X)
        #Basis_primal[:,:,DD_primal+D+1,k] = phi[k](X)**2
    #discouting over one period
    dtt = np.exp(-r*T/N1)
    #Perform Longstaff-Schwartz Regression for each payoff
    y0 = np.zeros(N_strikes)
    MC = np.zeros(N_strikes)
    for k in range(N_strikes):
        Basis = Basis_primal[:,:,:,k]
        Basis_Reg = Basis[:,subindex,:]
        YY = phi[k](X[:,subindex])
        value = YY[:,-1]
        
        reg = np.zeros((M2,N1-1))
        for n in range(N1-1):
            reg[:,n] = regr[k][n].predict(Basis_Reg[:,n,:])
        for m in range(M2):
            i = 0
            while YY[m,i]==0 or reg[m,i] > YY[m,i]:
                i = i+1
                if i == N1-1:
                    break
            value[m] = YY[m,i]*np.exp(-r*T*(ttt[i+1]-ttt[1]))
        print('Lower-biased price for Payoff Number',k,'is',np.mean(value),'with standard-deviation',np.std(value))
        y0[k] = np.mean(value)
        MC[k] = np.std(value)/np.sqrt(M2)
    ss = time.time()
    timee = ss-s
    return y0,MC,timee
        



    
    
    
    
    
    
    