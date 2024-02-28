#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 19 15:02:27 2023

Tests: American option pricing with signatures using partial markovianity in B to reduce the dimension of the signature.
lift I_t=(t,\int vdW_t).

@author: lucapelizzari
"""


import numpy as np
from FBM_LP_1 import FBM
import scipy as sc
from eulermaruyama import eulermaruyama
from sig_regression_terminal_LP import full_log_signature,signature,signatureQV
from regression import Regression
import matplotlib.pyplot as plt
import time
from fBmaugmented import SI1,fbmdualA,MC2A,dualLPsparse1
from TensorNormalization import polynomial, robustsignature
from rBergomi import rBergomi
import gurobipy as gu
from gurobipy import *
from helpfunctions import valuesatoptimal, optimalstopping, alpha, conditionaldeltas, conditionalprice, RomanoTouzi, valuefunction,alpha1, RomanoTouziRegression,alphainterpol,alphanew
from AndersonBroadie import runAndersonBroadi
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
#Longstaff-Schwarz with Signatures
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
    dB = rB.dB(dW1, dW2, rho = -0.9)
    X = rB.S(V, dB) #array of Mx(N+1)
    X = X0*X*np.exp(r*tt)
    #print('European:',np.mean(np.exp(-r)*phi(X[:,-1])), 'MC-Error:', np.std(np.exp(-r)*phi(X[:,-1]))/np.sqrt(M))
    I = np.zeros(shape = (M,int(T*N)+1))
    for n in range(int(T*N)):
        I[:,n+1] = I[:,n] + np.sqrt(V[:,n])*dW1[:,n,0]
    dI = I[:,1:int(T*N)+1]-I[:,0:int(T*N)]
    dI = dI.reshape(M,int(T*N),1)
    return X,V,I,dI,dW1,dW2,dB
def AmericanOptionPricerBergomiND(M1,M2,M3,N,N1,T,phi,rho,K,KK,strike,X0,H,xi,eta,r,modeprimal,modedual):
    """Pricing American options (Bermuddan options) in rBergomi using Longstaff-Schwarz with signatures.
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
    """
    #Simulate everything from rBergomi
    X,V,I,dI,dW1,dW2,dB = SimulationofrBergomi(M1,N,T,phi,rho,K,X0,H,xi,eta,r)
    print('European 2:',np.mean(np.exp(-r)*phi(X[:,-1])), 'MC-Error:', np.std(np.exp(-r)*phi(X[:,-1]))/np.sqrt(M1))
    #laguerre polynomial for regression
    
    F = [0]*2*K 
    for k in range(2*K):
        F[k] = sc.special.genlaguerre(k+1, 0)
    regr = PrimalRegression(T,phi,rho,V,X,dW1[:,:,0],dW2,dB,dI,N1,K,KK,strike,X0,r,F,modeprimal)

    del dW1,dW2,V,X,I,dI,dB
    X,V,I,dI,dW1,dW2,dB = SimulationofrBergomi(M2,N,T,phi,rho,K,X0,H,xi,eta,r)
  
    #derive lower bound and regression coefficients for Dual approach
    y0_lo,beta = resimluationPrimalDual(regr,T,phi,rho,V,X,dW1[:,:,0],dW2,dB,dI,N1,K,KK,X0,M2,r,F,modeprimal,modedual)
    #
    #Final Resimulation for upper biased values with Anderson-Broadie
    
    del dW1,dW2,V,X,I,dI,dB
    X,V,I,dI,dW1,dW2,dB = SimulationofrBergomi(M2,N,T,phi,rho,K,X0,H,xi,eta,r)
    y0_up,Doob = upperbiasedprice(y0_lo,beta,T,phi,rho,V,X,dW1[:,:,0],dW2,dB,dI,N1,K,X0,M2,r,F,modedual)
    #print(np.mean(DoobMG,axis=0))
    return y0_lo,y0_up,Doob
       
def PrimalRegression(T,phi,rho,v,X,dW,dB,dBB,dI,N1,K,KK,strike,X0,r,F,modeprimal):
    """ Computing Regression coefficient for Longstaff-Schwarz with signatures.
    T = maturity
    rho = correlation factor in [-1,1]
    v = Samples of volatility process (in rBergomi), Mx(N+1) (M=simulations, N = discretization of [0,T])
    dW,dB = increments of two independent Brownian motions, arrays of the form MxN
    dI = increments of the martingale \intv_sdW_s, array of the form MxN
    mode = process we lift to signature
    K = depth of Signature
    X0 = starting value for log-price
    """
    M,N = dW.shape
    subindex = [int((j+1)*N/N1) for j in range(N1)]
    subindex2 = [int((j)*N/N1) for j in range(N1+1)]
    subindex3 = [int((j+1)*N/N1-1) for j in range(N1)]
    tt = np.linspace(0,T,N+1)
    #discounted payoff on whole grid
    Z = np.exp(-r*tt)*phi(X)
    #compute signature depending on lifts we choose in mode
    if modeprimal == "(QV,X,XX,XXX)-Lift":
        D = int((1-(3+1)**(K+1))/(-3) -1)
        II = np.zeros((M,N+1))
        II[:,1:N+1] = np.cumsum(dI[:,:,0],axis=1)
        U = Z
        UU = F[0](X)
        UUU = F[1](X)
        QV = np.zeros(shape = (M,N+1))
        for n in range(N):
            QV[:,n+1] = QV[:,n]+v[:,n]*1/(N+1)
        dZ = np.zeros((M,N,3))
        dZ[:,:,0] = U[:,1:N+1]-U[:,0:N]
        dZ[:,:,1] = UU[:,1:N+1]-UU[:,0:N]
        dZ[:,:,2] =  UUU[:,1:N+1]-UUU[:,0:N]
        S = signature(tt,dZ,K)
    
    if modeprimal == "(I,QV,X)-Lift":
        D = int((1-(2+1)**(K+1))/(-2) -1)
        QV = np.zeros(shape = (M,N+1))
        dX = X[:,1:N+1]-X[:,0:N]
        for n in range(N):
            QV[:,n+1] = QV[:,n]+v[:,n]*1/(N+1)
        dZ = np.zeros((M,N,2))
        dZ[:,:,0] = dI[:,:,0]
        dZ[:,:,1] = dX
        S = signatureQV(tt,dZ,QV,K)
    if modeprimal == "(t,I,B)-Lift":
        D = int((1-(2+1)**(K+1))/(-2) -1)
        dZ = np.zeros((M,N,2))
        dZ[:,:,0] = dI[:,:,0]
        dZ[:,:,1] = dB
        S = signature(tt,dZ,K)
        SD = np.zeros((M,N+1,D))
        SD[:,1:N+1] = S
    if modeprimal == "([I],I)-Lift":
        D = int((1-(1+1)**(K+1))/(-1) -1)
        QV = np.zeros(shape = (M,N+1))
        for n in range(N):
            QV[:,n+1] = QV[:,n]+v[:,n]*1/(N+1)
        S = signatureQV(tt,dI,QV,K)
    if modeprimal == "(t,I)-Lift":
        D = int((1-(1+1)**(K+1))/(-1) -1)
        S = signature(tt,dI,K)
        SD = np.zeros((M,N+1,D))
        SD[:,1:N+1] = S
    if modeprimal == '(QV,X)-Lift':
        D = int((1-(1+1)**(K+1))/(-1) -1)
        dX = X[:,1:N+1]-X[:,0:N]
        dX = dX.reshape(M,N,1)
        QV = np.zeros(shape = (M,N+1))
        for n in range(N):
            QV[:,n+1] = QV[:,n]+v[:,n]*1/(N+1)
        S = signatureQV(tt,dX,QV,K)
        SD = np.zeros((M,N+1,D))
        SD[:,1:N+1] = S
    if modeprimal == '(QV,W,B)-Lift':
        D = int((1-(1+2)**(K+1))/(-1-1) -1)
        dZZ = np.zeros((M,N,2))
        dZZ[:,:,0] = dW
        dZZ[:,:,1] = dB
        QV = np.zeros(shape = (M,N+1))
        for n in range(N):
            QV[:,n+1] = QV[:,n]+v[:,n]*1/(N+1)
        S = signatureQV(tt,dZZ,QV,K)
        SD = np.zeros((M,N+1,D))
        SD[:,1:N+1] = S
    if modeprimal == '(QV,X,Y)-Lift':
        D = int((1-(2+1)**(K+1))/(-2) -1)
        dX = X[:,1:N+1]-X[:,0:N]
        dZ = Z[:,1:N+1]-Z[:,0:N]
        dZZ = np.zeros((M,N,2))
        dZZ[:,:,0] = dX
        dZZ[:,:,1] = dZ
        QV = np.zeros(shape = (M,N+1))
        for n in range(N):
            QV[:,n+1] = QV[:,n]+v[:,n]*1/(N+1)
        S = signatureQV(tt,dZZ,QV,K)
        SD = np.zeros((M,N+1,D))
        SD[:,1:N+1] = S
    if modeprimal == '(QV,I,Y)-Lift':
        D = int((1-(2+1)**(K+1))/(-2) -1)
        dZ = Z[:,1:N+1]-Z[:,0:N]
        dZZ = np.zeros((M,N,2))
        dZZ[:,:,0] = dI[:,:,0]
        dZZ[:,:,1] = dZ
        QV = np.zeros(shape = (M,N+1))
        for n in range(N):
            QV[:,n+1] = QV[:,n]+v[:,n]*1/(N+1)
        S = signatureQV(tt,dZZ,QV,K)
    if modeprimal == '(QV,I,X)-Lift':
        D = int((1-(2+1)**(K+1))/(-2) -1)
        dX = X[:,1:N+1]-X[:,0:N]
        dZZ = np.zeros((M,N,2))
        dZZ[:,:,0] = dI[:,:,0]
        dZZ[:,:,1] = dX
        QV = np.zeros(shape = (M,N+1))
        for n in range(N):
            QV[:,n+1] = QV[:,n]+v[:,n]*1/(N+1)
        S = signatureQV(tt,dZZ,QV,K)
    if modeprimal == '(t,B,X)-Lift':
         D = int((1-(2+1)**(K+1))/(-2) -1)
         dX = X[:,1:N+1]-X[:,0:N]
         dZZ = np.zeros((M,N,2))
         dZZ[:,:,0] = dB
         dZZ[:,:,1] = dX
         S = signature(tt,dZZ,K)
    if modeprimal == '(t,I,X)-Lift':
        D = int((1-(2+1)**(K+1))/(-2) -1)
        if K == 0:
            i = 1
        
        else:
    
            dX = X[:,1:N+1]-X[:,0:N]
            dZZ = np.zeros((M,N,2))
            dZZ[:,:,0] = dI[:,:,0]
            dZZ[:,:,1] = dX
            S = signature(tt,dZZ,K)
            SD = np.zeros((M,N+1,D))
            SD[:,1:N+1] = S
    if modeprimal == '(t,V)-Lift':
         D = int((1-(1+1)**(K+1))/(-1) -1)
         dX = np.sqrt(v[:,1:N+1])-np.sqrt(v[:,0:N])
         dX = dX.reshape(M,N,1)
         S = signature(tt,dX,K)
         SD = np.zeros((M,N+1,D))
         SD[:,1:N+1] = S
    if modeprimal == '(QV,V)-Lift':
         D = int((1-(1+1)**(K+1))/(-1) -1)
         dX = np.sqrt(v[:,1:N+1])-np.sqrt(v[:,0:N])
         dX = dX.reshape(M,N,1)
         QV = np.zeros(shape = (M,N+1))
         for n in range(N):
             QV[:,n+1] = QV[:,n]+v[:,n]*1/(N+1)
         S = signatureQV(tt,dX,QV,K)
         SD = np.zeros((M,N+1,D))
         SD[:,1:N+1] = S
    if modeprimal == '(t,W)-Lift':
         D = int((1-(1+1)**(K+1))/(-1) -1)
         dX = dW[:,:].reshape(M,N,1)
         S = signature(tt,dX,K)
    if modeprimal == '(t,V,X)-Lift':
        D = int((1-(2+1)**(K+1))/(-2) -1)
        dZ = np.sqrt(v[:,1:N+1])-np.sqrt(v[:,0:N])
        dX = X[:,1:N+1]-X[:,0:N]
        dZZ = np.zeros((M,N,2))
        dZZ[:,:,0] = dZ
        dZZ[:,:,1] = dX
        S = signature(tt,dZZ,K)
    if modeprimal == '(t,W,X)-Lift':
        D = int((1-(2+1)**(K+1))/(-2) -1)
        if D == 0:
            ooo = 1
        else:
            dX = X[:,1:N+1]-X[:,0:N]
            dZZ = np.zeros((M,N,2))
            dZZ[:,:,0] = dW
            dZZ[:,:,1] = dX
            S = signature(tt,dZZ,K)
            SD = np.zeros((M,N+1,D))
            SD[:,1:N+1] = S
            #SD = np.zeros((M,N+1,D))
            #SD[:,1:N+1] = S
            """W = np.zeros((M,N+1))
            W[:,1:N+1] = np.cumsum(dW,axis=1)
            XX = np.zeros((M,N+1,3))
            XX[:,:,0] = tt
            XX[:,:,1] = X
            XX[:,:,2] = W
            S = full_log_signature(XX, K)
            D = len(S[0,0,:])
            SD = S"""
    
    #computing price-process
    YY = phi(X[:,subindex])
    value = YY[:,-1]
    #integrand for Doob-Martingale
    regr = [0]*(N1-1)
    #interest
    dtt = np.exp(-r*(T/(N1+1)))
    DD = int((KK+1)*(KK+2)/2)
    P = np.zeros((M,N+1,DD))
    for k in range(KK+1):
        for j in range(0,k+1):
            C = np.zeros((KK+1,KK+1))
            C[k,j] = 1
            P[:,:,int(k*(k+1)/2+j)] = np.polynomial.laguerre.lagval2d(X, np.sqrt(v), C)
            #S[:,:,int(k*(k+1)/2+j)] = np.sqrt(V)**j*X**(k-j)
    if KK == 0:
        S1 = np.ones(shape=(M,N1,D+1))
        S1[:,:,1:D+1] = SD[:,subindex,:] #adding Signature value on first level (=1)
        Sig = S1
    if K == 0:
        Sig = P[:,subindex,:]
    if K >0 and KK > 0:
        S1 = np.ones(shape=(M,N1,D+1))
        S1[:,:,1:D+1] = SD[:,subindex,:] #adding Signature value on first level (=1)
        Sig = np.zeros((M,N1,DD+D))
        Sig[:,:,0:DD] = P[:,subindex,:]
        Sig[:,:,DD:DD+D+1] = SD[:,subindex,:]
    
    
    for j in reversed(range(1,N1)):
        #only in the money paths
        ITM = []
        value = value*dtt
        for k in range(M):
            if YY[k,j-1]>0:
                ITM.append(k)
        
        if len(ITM)==0:
            continue
        else:
            #regr[j-1] = Regression(S2[ITM,j-1,:],value[ITM],mode = 'linear')
            #regr[j-1] = Ridge(alpha=100,fit_intercept=False).fit(S1[ITM,j-1,:],value[ITM])
            #print(regr[j-1].score(S1[ITM,j-1,:],value[ITM]))
            regr[j-1] = LinearRegression().fit(Sig[ITM,j-1,:],value[ITM])
            print(regr[j-1].score(Sig[ITM,j-1,:],value[ITM]))
            reg = regr[j-1].predict(Sig[ITM,j-1,:])
            print('RMSE',np.mean((reg - value[ITM])**2))
            if j == 30:
                print(regr[j-1].coef_)
            for m in range(len(ITM)):
                if reg[m] > YY[ITM[m],j-1]:
                    continue
                else:
                    value[ITM[m]] = YY[ITM[m],j-1]
    print('Primal 1 unbiased:',np.mean(value[:]))
    print('Primal, unbiased:', np.mean(valuesatoptimal(0,Sig,YY,regr,T,r)), 'MC-Error:', np.std(valuesatoptimal(0,Sig,YY,regr,T,r))/np.sqrt(M))
    #print('Primal after 1, unbiased:', np.mean(u(1,S2,YY,regr,T)), 'MC-Error:', np.std(u(0,S2,YY,regr,T))/np.sqrt(M))
    #print('Primal at the ende, unbiased:', np.mean(u(N1-1,S2,YY,regr,T)), 'MC-Error:', np.std(u(0,S2,YY,regr,T))/np.sqrt(M))
    #print('Sanity für funktion u',[np.mean(u(j,S2,YY,regr,T)) for j in range(N1)])
    
   
    """
    #Romano-TOuzi European prices by Regression
    IV = np.zeros(shape = (M,N+1))
    I = np.zeros((M,N+1))
    I[:,1:N+1] = dI[:,:,0].cumsum(axis=1)
    for n in range(N):
        IV[:,n+1] = IV[:,n]+v[:,n]*1/(N+1)
    RTRegression = RomanoTouziRegression(X, phi, rho, I, IV, T, strike,K, r,F)"""
    
    return regr

def resimluationPrimalDual(regr,T,phi,rho,v,X,dW,dB,dBB,dI,N1,K,KK,X0,M2,r,F,modeprimal,modedual):
    """For given regression coefficients in Longstaff-Schwarz algorithm with signatures, we now compute lower-biased prices using LS, and with the 
    resulting value-function, we construct approximations for the Doob-MG, and hence upper-biased prices..
    M2 = number of resimulations
    regr = coefficients from regression """
    
    M,N = dW.shape
    subindex = [int((j+1)*N/N1) for j in range(N1)]
    subindex2 = [int((j)*N/N1) for j in range(N1+1)]
    subindex3 = [int((j+1)*N/N1-1) for j in range(N1)]
    D = int((1-(2)**(K+1))/(-1) -1)
    tt = np.linspace(0,T,N+1)
    #discounted payoff on whole grid
    Z = np.exp(-r*tt)*phi(X)
    #compute signature depending on lifts we choose in mode in primal mode
    if modeprimal == "(QV,I,X)-Lift":
        D = int((1-(2+1)**(K+1))/(-2) -1)
        QV = np.zeros(shape = (M,N+1))
        dX = X[:,1:N+1]-X[:,0:N]
        for n in range(N):
            QV[:,n+1] = QV[:,n]+v[:,n]*1/(N+1)
        dZ = np.zeros((M,N,2))
        dZ[:,:,0] = dI[:,:,0]
        dZ[:,:,1] = dX
        S = signatureQV(tt,dZ,QV,K)
    if modeprimal == "([I],I)-Lift":
        D = int((1-(1+1)**(K+1))/(-1) -1)
        QV = np.zeros(shape = (M,N+1))
        for n in range(N):
            QV[:,n+1] = QV[:,n]+v[:,n]*1/(N+1)
        S = signatureQV(tt,dI,QV,K)
    if modeprimal == "(t,I)-Lift":
        D = int((1-(1+1)**(K+1))/(-1) -1)
        S = signature(tt,dI,K)
        SD = np.zeros((M,N+1,D))
        SD[:,1:N+1] = S
    if modeprimal == '(QV,X)-Lift':
        D = int((1-(1+1)**(K+1))/(-1) -1)
        dX = X[:,1:N+1]-X[:,0:N]
        dX = dX.reshape(M,N,1)
        QV = np.zeros(shape = (M,N+1))
        for n in range(N):
            QV[:,n+1] = QV[:,n]+v[:,n]*1/(N+1)
        S = signatureQV(tt,dX,QV,K)
        SD = np.zeros((M,N+1,D))
        SD[:,1:N+1] = S
    if modeprimal == '(QV,X,Y)-Lift':
        D = int((1-(2+1)**(K+1))/(-2) -1)
        dX = X[:,1:N+1]-X[:,0:N]
        dZ = Z[:,1:N+1]-Z[:,0:N]
        dZZ = np.zeros((M,N,2))
        dZZ[:,:,0] = dX
        dZZ[:,:,1] = dZ
        QV = np.zeros(shape = (M,N+1))
        for n in range(N):
            QV[:,n+1] = QV[:,n]+v[:,n]*1/(N+1)
        S = signatureQV(tt,dZZ,QV,K)
        SD = np.zeros((M,N+1,D))
        SD[:,1:N+1] = S
    if modeprimal == '(QV,W,B)-Lift':
        D = int((1-(1+2)**(K+1))/(-1-1) -1)
        dZZ = np.zeros((M,N,2))
        dZZ[:,:,0] = dW
        dZZ[:,:,1] = dB
        QV = np.zeros(shape = (M,N+1))
        for n in range(N):
            QV[:,n+1] = QV[:,n]+v[:,n]*1/(N+1)
        S = signatureQV(tt,dZZ,QV,K)
        SD = np.zeros((M,N+1,D))
        SD[:,1:N+1] = S
    if modeprimal == '(t,I,X)-Lift':
        D = int((1-(2+1)**(K+1))/(-2) -1)
        if K == 0:
            rrrr = 1
        
        else:
    
            dX = X[:,1:N+1]-X[:,0:N]
            dZZ = np.zeros((M,N,2))
            dZZ[:,:,0] = dI[:,:,0]
            dZZ[:,:,1] = dX
            S = signature(tt,dZZ,K)
            SD = np.zeros((M,N+1,D))
            SD[:,1:N+1] = S
    if modeprimal == '(QV,I,Y)-Lift':
        D = int((1-(2+1)**(K+1))/(-2) -1)
        dZ = Z[:,1:N+1]-Z[:,0:N]
        dZZ = np.zeros((M,N,2))
        dZZ[:,:,0] = dI[:,:,0]
        dZZ[:,:,1] = dZ
        QV = np.zeros(shape = (M,N+1))
        for n in range(N):
            QV[:,n+1] = QV[:,n]+v[:,n]*1/(N+1)
        S = signatureQV(tt,dZZ,QV,K)
    if modeprimal == '(t,V)-Lift':
         D = int((1-(1+1)**(K+1))/(-1) -1)
         dX = np.sqrt(v[:,1:N+1])-np.sqrt(v[:,0:N])
         dX = dX.reshape(M,N,1)
         S = signature(tt,dX,K)
         SD = np.zeros((M,N+1,D))
         SD[:,1:N+1] = S
    if modeprimal == '(QV,V)-Lift':
         D = int((1-(1+1)**(K+1))/(-1) -1)
         dX = np.sqrt(v[:,1:N+1])-np.sqrt(v[:,0:N])
         dX = dX.reshape(M,N,1)
         QV = np.zeros(shape = (M,N+1))
         for n in range(N):
             QV[:,n+1] = QV[:,n]+v[:,n]*1/(N+1)
             
         S = signatureQV(tt,dX,QV,K)
         SD = np.zeros((M,N+1,D))
         SD[:,1:N+1] = S
    if modeprimal == '(t,V)-Lift':
         D = int((1-(1+1)**(K+1))/(-1) -1)
         dX = np.sqrt(v[:,1:N+1])-np.sqrt(v[:,0:N])
         dX = dX.reshape(M,N,1)
             
         S = signature(tt,dX,K)
         SD = np.zeros((M,N+1,D))
         SD[:,1:N+1] = S
    if modeprimal == '(t,W,X)-Lift':
        D = int((1-(2+1)**(K+1))/(-2) -1)
        if D == 0:
             ooo=1
        else:
            dX = X[:,1:N+1]-X[:,0:N]
            dZZ = np.zeros((M,N,2))
            dZZ[:,:,0] = dW
            dZZ[:,:,1] = dX
            S = signature(tt,dZZ,K)
            SD = np.zeros((M,N+1,D))
            SD[:,1:N+1] = S
         
    if modeprimal == '(t,B,X)-Lift':
         D = int((1-(2+1)**(K+1))/(-2) -1)
         dX = X[:,1:N+1]-X[:,0:N]
         dZZ = np.zeros((M,N,2))
         dZZ[:,:,0] = dB
         dZZ[:,:,1] = dX
         S = signature(tt,dZZ,K)
    if modeprimal == '(t,V,X)-Lift':
        D = int((1-(2+1)**(K+1))/(-2) -1)
        dZ = np.sqrt(v[:,1:N+1])-np.sqrt(v[:,0:N])
        dX = X[:,1:N+1]-X[:,0:N]
        dZZ = np.zeros((M,N,2))
        dZZ[:,:,0] = dZ
        dZZ[:,:,1] = dX
        S = signature(tt,dZZ,K)
        
    
    
    YY = phi(X[:,subindex])
    DD = int((KK+1)*(KK+2)/2)
    P = np.zeros((M,N+1,DD))
    for k in range(KK+1):
        for j in range(0,k+1):
            C = np.zeros((KK+1,KK+1))
            C[k,j] = 1
            P[:,:,int(k*(k+1)/2+j)] = np.polynomial.laguerre.lagval2d(X, np.sqrt(v), C)
            #S[:,:,int(k*(k+1)/2+j)] = np.sqrt(V)**j*X**(k-j)
    if KK == 0:
        S1 = np.ones(shape=(M,N1,D+1))
        S1[:,:,1:D+1] = SD[:,subindex,:] #adding Signature value on first level (=1)
        Sig = S1
    if K == 0:
        Sig = P[:,subindex,:]
    if K >0 and KK > 0:
        S1 = np.ones(shape=(M,N1,D+1))
        S1[:,:,1:D+1] = SD[:,subindex,:] #adding Signature value on first level (=1)
        Sig = np.zeros((M,N1,DD+D))
        Sig[:,:,0:DD] = P[:,subindex,:]
        Sig[:,:,DD:DD+D+1] = SD[:,subindex,:]
    
    
    value = YY[:,-1]
    #integrand for Doob-Martingale
    #interest
    dtt = np.exp(-r*(T/(N1+1)))
    
    
    
    
    
    
    #gibt besseren Wert
    optimalvalues = valuesatoptimal(0,Sig,YY,regr,T,r)
    #print(optimalstopping(0,S1,YY,regr,T))
    y0_lb = np.mean(optimalvalues)
    #Snell-envelope mit Tisikilis van roy:
    Snell = np.zeros((M,N1+1))
    Snell[:,0] = y0_lb
    #y0_lb = np.mean(np.maximum(Z[:,0],optimalvalues))
    #print('Primal lower biased',y0_lb,'MC:', np.std(stopp(0,S3,YY,regr,T)/np.sqrt(M2)))
    #valuefct = valuefunction(S3,YY,regr,T,r)
    #print('Primal',np.mean(valuefct[:,0]))
    print('Primal lower biased ',y0_lb,'MC:', np.std(optimalvalues)/np.sqrt(M2))
    #Signature for Dual-Procedure
    if modedual == "(I,QV,X)-Lift":
        D = int((1-(2+1)**(K+1))/(-2) -1)
        QV = np.zeros(shape = (M,N+1))
        dX = X[:,1:N+1]-X[:,0:N]
        for n in range(N):
            QV[:,n+1] = QV[:,n]+v[:,n]*1/(N+1)
        dZ = np.zeros((M,N,2))
        dZ[:,:,0] = dI[:,:,0]
        dZ[:,:,1] = dX
        SD = signatureQV(tt,dZ,QV,K)
    if modedual == '(t,B,X)-Lift':
         D = int((1-(2+1)**(K+1))/(-2) -1)
         dX = X[:,1:N+1]-X[:,0:N]
         dZZ = np.zeros((M,N,2))
         dZZ[:,:,0] = dB
         dZZ[:,:,1] = dX
         S = signature(tt,dZZ,K)
    if modedual == "([I],I)-Lift":
        D = int((1-(1+1)**(K+1))/(-1) -1)
        #Add Snell envelope to Signature?
        #Snell = valuefunction(S2,phi(X),regr,T,r)
        #plt.plot(Snell[0,:])
        QV = np.zeros(shape = (M,N+1))
        for n in range(N):
            QV[:,n+1] = QV[:,n]+v[:,n]*1/(N+1)
        SD = signatureQV(tt,dI,QV,K)
        
        """SSS1 = np.ones(shape=(M,N,D+1+K))
        SSS1[:,:,K+1:D+1+K] = S
        for k in range(K):
            #using laguerre polynomials for states
            SSS1[:,:,k+1] = F[k](X[:,1:N+1])
        Europ = np.zeros((M,N+1))
        for j in range(N):
            if j == 0:
                Europ[:,j] = RTregr[0]
            else:
                Europ[:,j] = RTregr[j].predict(SSS1[:,j-1,:])
        Europ[:,-1] = Z[:,-1]"""
        #dE = Europ[:,1:N+1]-Europ[:,0:N]
        #dE = dE.reshape(M,N,1)
        #SD = signatureQV(tt,dE,QV,K)
    if modedual == "(t,I)-Lift":
        K = K+1
        D = int((1-(1+1)**(K+1))/(-1) -1)
        SD = signature(tt,dI,K)
    if modedual == '(QV,X)-Lift':
        K = K+1
        I =dI[:,:,0].cumsum(axis=1)
        II = np.zeros((M,N+1))
        II[:,1:N+1] = I
        D = int((1-(1+1)**(K+1))/(-1) -1)
        dX = X[:,1:N+1]-X[:,0:N]
        dX = dX.reshape(M,N,1)
        QV = np.zeros(shape = (M,N+1))
        for n in range(N):
            QV[:,n+1] = QV[:,n]+np.sqrt(v[:,n])*1/(N+1)
        
        SD = signatureQV(tt,dX,QV,K)
    if modedual == '(t,W,X)-Lift':
         K = K+1
         D = int((1-(2+1)**(K+1))/(-2) -1)
         dX = X[:,1:N+1]-X[:,0:N]
         dZZ = np.zeros((M,N,2))
         dZZ[:,:,0] = dW
         dZZ[:,:,1] = dX
         SD = signature(tt,dZZ,K)
    if modeprimal == '(t,V)-Lift':
         D = int((1-(1+1)**(K+1))/(-1) -1)
         dX = np.sqrt(v[:,1:N+1])-np.sqrt(v[:,0:N])
         dX = dX.reshape(M,N,1)
             
         S = signature(tt,dX,K)
         SD = np.zeros((M,N+1,D))
         SD[:,1:N+1] = S
    if modedual == '(t,X)-Lift':
        K = K+1
        I =dI[:,:,0].cumsum(axis=1)
        II = np.zeros((M,N+1))
        II[:,1:N+1] = I
        D = int((1-(1+1)**(K+1))/(-1) -1)
        dX = X[:,1:N+1]-X[:,0:N]
        dX = dX.reshape(M,N,1)
        SD = signature(tt,dX,K)
    if modedual == '(t,I,X)-Lift':
        D = int((1-(2+1)**(K+1))/(-2) -1)
        dX = X[:,1:N+1]-X[:,0:N]
        dZZ = np.zeros((M,N,2))
        dZZ[:,:,0] = dI[:,:,0]
        dZZ[:,:,1] = dX
        S = signature(tt,dZZ,K)
    if modedual == '(QV,X,Y)-Lift':
        D = int((1-(2+1)**(K+1))/(-2) -1)
        dX = X[:,1:N+1]-X[:,0:N]
        dZ = Z[:,1:N+1]-Z[:,0:N]
        dZZ = np.zeros((M,N,2))
        dZZ[:,:,0] = dX
        dZZ[:,:,1] = dZ
        QV = np.zeros(shape = (M,N+1))
        for n in range(N):
            QV[:,n+1] = QV[:,n]+v[:,n]*1/(N+1)
        SD = signatureQV(tt,dZZ,QV,K)
    if modedual == '(QV,I,Y)-Lift':
        D = int((1-(2+1)**(K+1))/(-2) -1)
        dZ = Z[:,1:N+1]-Z[:,0:N]
        dZZ = np.zeros((M,N,2))
        dZZ[:,:,0] = dI[:,:,0]
        dZZ[:,:,1] = dZ
        QV = np.zeros(shape = (M,N+1))
        for n in range(N):
            QV[:,n+1] = QV[:,n]+v[:,n]*1/(N+1)
        SD = signatureQV(tt,dZZ,QV,K)
    if modedual == '(QV,W,B)-Lift':
        D = int((1-(2+1)**(K+1))/(-2) -1)
        dZ = Z[:,1:N+1]-Z[:,0:N]
        dZZ = np.zeros((M,N,2))
        dZZ[:,:,0] = dW
        dZZ[:,:,1] = dB
        QV = np.zeros(shape = (M,N+1))
        for n in range(N):
            QV[:,n+1] = QV[:,n]+v[:,n]*1/(N+1)
        SD = signatureQV(tt,dZZ,QV,K)
    if modedual == '(t,W,B)-Lift':
        D = int((1-(2+1)**(K+1))/(-2) -1)
        dZ = Z[:,1:N+1]-Z[:,0:N]
        dZZ = np.zeros((M,N,2))
        dZZ[:,:,0] = dW
        dZZ[:,:,1] = dB
       
        SD = signature(tt,dZZ,K)
    if modedual == '(QV,B,Y)-Lift':
        D = int((1-(2+1)**(K+1))/(-2) -1)
        dZ = Z[:,1:N+1]-Z[:,0:N]
        dZZ = np.zeros((M,N,2))
        dZZ[:,:,0] = dZ
        dZZ[:,:,1] = dB
        QV = np.zeros(shape = (M,N+1))
        for n in range(N):
            QV[:,n+1] = QV[:,n]+v[:,n]*1/(N+1)
        SD = signatureQV(tt,dZZ,QV,K)
    if modedual == '(t,V)-Lift':
         D = int((1-(1+1)**(K+1))/(-1) -1)
         dX = np.sqrt(v[:,1:N+1])-np.sqrt(v[:,0:N])
         dX = dX.reshape(M,N,1)
         SD = signature(tt,dX,K)
    if modedual == '(QV,V)-Lift':
         D = int((1-(1+1)**(K+1))/(-1) -1)
         dX = np.sqrt(v[:,1:N+1])-np.sqrt(v[:,0:N])
         dX = dX.reshape(M,N,1)
         QV = np.zeros(shape = (M,N+1))
         for n in range(N):
             QV[:,n+1] = QV[:,n]+v[:,n]*1/(N+1)
         SD = signatureQV(tt,dX,QV,K)
    if modedual == '(QV,B)-Lift':
         D = int((1-(1+1)**(K+1))/(-1) -1)
         dX = dB.reshape(M,N,1)
         QV = np.zeros(shape = (M,N+1))
         for n in range(N):
             QV[:,n+1] = QV[:,n]+v[:,n]*1/(N+1)
         SD = signatureQV(tt,dX,QV,K)
    if modedual == '(t,W)-Lift':
         D = int((1-(1+1)**(K+1))/(-1) -1)
         dX = dW.reshape(M,N,1)
         SD = signature(tt,dX,K)
    
    if modedual == '(t,V,X)-Lift':
        D = int((1-(2+1)**(K+1))/(-2) -1)
        dZ = np.sqrt(v[:,1:N+1])-np.sqrt(v[:,0:N])
        dX = X[:,1:N+1]-X[:,0:N]
        dZZ = np.zeros((M,N,2))
        dZZ[:,:,0] = dZ
        dZZ[:,:,1] = dX
        SD = signature(tt,dZZ,K)
    if modedual == '(t,I,X)-Lift':
        D = int((1-(2+1)**(K+1))/(-2) -1)
        dX = X[:,1:N+1]-X[:,0:N]
        dZZ = np.zeros((M,N,2))
        dZZ[:,:,0] = dI[:,:,0]
        dZZ[:,:,1] = dX
        S = signature(tt,dZZ,K)
    if modedual == '(t,V,X)-Lift':
        D = int((1-(2+1)**(K+1))/(-2) -1)
        dZ = np.sqrt(v[:,1:N+1])-np.sqrt(v[:,0:N])
        dX = X[:,1:N+1]-X[:,0:N]
        dZZ = np.zeros((M,N,2))
        dZZ[:,:,0] = dZ
        dZZ[:,:,1] = dX
        SD = signature(tt,dZZ,K)
    if modedual == '(QV,B,Y)-Lift':
        D = int((1-(2+1)**(K+1))/(-2) -1)
        dZ = np.sqrt(v[:,1:N+1])-np.sqrt(v[:,0:N])
        QV = np.zeros(shape = (M,N+1))
        for n in range(N):
            QV[:,n+1] = QV[:,n]+v[:,n]*1/(N+1)
        dX = X[:,1:N+1]-X[:,0:N]
        dZZ = np.zeros((M,N,2))
        dZZ[:,:,0] = dZ
        dZZ[:,:,1] = dB
        SD = signatureQV(tt,dZZ,QV,K)
    if modedual == '(QV,X,BB)-Lift':
         D = int((1-(2+1)**(K+1))/(-2) -1)
         dZZ = np.zeros((M,N,2))
         QV = np.zeros(shape = (M,N+1))
         for n in range(N):
             QV[:,n+1] = QV[:,n]+v[:,n]*1/(N+1)
         dZZ[:,:,0] = dBB
         dZZ[:,:,1] = X[:,1:N+1]-X[:,0:N]
         SD = signature(tt,dZZ,K)
    #Start Dual-procedure
    #first compute reshape dual-signature
    S4 = np.ones(shape=(M,N,D+1))
    S4[:,:,1:D+1] = SD
    S5 = np.zeros(shape = (M,N+1,D+1))
    S5[:,1:N+1,:] = S4
    S5[:,0,0] = 1
    S6 = S5[:,subindex,:]
    #Dual Regression:
    regrDoob1 = [0]*(N1)
    
    for j in range(N1):
        regrDoob1[j] = Ridge(alpha = 1000,fit_intercept=False).fit(S6[:,j,:],valuesatoptimal(j,Sig,YY,regr,T,r))
        print('Score:',regrDoob1[j].score(S6[:,j,:],(valuesatoptimal(j,Sig,YY,regr,T,r))))
        print('RMSE',np.mean((regrDoob1[j].predict(S6[:,j,:]) - valuesatoptimal(j,Sig,YY,regr,T,r))**2))
        if j == 10:
            print(regrDoob1[j].coef_)
    beta = np.zeros((N1,int((1-(2)**(K))/(-1) -1)+1))
    
    """Snell = np.zeros((M,N1))
    Snell[:,0] = y0_lb
    for j in range(N1-1):
        Snell[:,j+1] = regrDoob1[j+1].predict(S6[:,j+1,:])
    plt.plot(Snell[0,:],'r')
    plt.plot(X[0,subindex2])"""
    for k in range(N1):
        for l in range(1,int((1-(2)**(K))/(-1) -1)+2):
            beta[k,l-1] = regrDoob1[k].coef_[2*l]
    
    print('beta', beta[0,:])
    DoobMG = np.zeros(shape=(M,N+1))
    return y0_lb,beta

#new method
def upperbiasedprice(y0_lo,beta,T,phi,rho,v,X,dW,dB,dBB,dI,N1,K,X0,M2,r,F,modedual):
    M,N = dW.shape
    subindex = [int((j+1)*N/N1) for j in range(N1)]
    subindex2 = [int((j)*N/N1) for j in range(N1+1)]
    D = int((1-(2)**(K+1))/(-1) -1)
    tt = np.linspace(0,T,N+1)
    #discounted payoff on whole grid
    Z = np.exp(-r*tt)*phi(X)
    """Compute Doob-MG and upper biased prices"""
    if modedual == '(t,X)-Lift':
        K = K+1
        I =dI[:,:,0].cumsum(axis=1)
        II = np.zeros((M,N+1))
        II[:,1:N+1] = I
        D = int((1-(1+1)**(K+1))/(-1) -1)
        dX = X[:,1:N+1]-X[:,0:N]
        dX = dX.reshape(M,N,1)
        SD = signature(tt,dX,K-1)
    if modedual == "(I,QV,X)-Lift":
        D = int((1-(2+1)**(K+1))/(-2) -1)
        QV = np.zeros(shape = (M,N+1))
        dX = X[:,1:N+1]-X[:,0:N]
        for n in range(N):
            QV[:,n+1] = QV[:,n]+v[:,n]*1/(N+1)
        dZ = np.zeros((M,N,2))
        dZ[:,:,0] = dI[:,:,0]
        dZ[:,:,1] = dX
        SD = signatureQV(tt,dZ,QV,K)
    if modedual == '(t,I,X)-Lift':
        D = int((1-(2+1)**(K+1))/(-2) -1)
        dX = X[:,1:N+1]-X[:,0:N]
        dZZ = np.zeros((M,N,2))
        dZZ[:,:,0] = dI[:,:,0]
        dZZ[:,:,1] = dX
        S = signature(tt,dZZ,K)
    if modedual == '(t,B,X)-Lift':
         D = int((1-(2+1)**(K+1))/(-2) -1)
         dX = X[:,1:N+1]-X[:,0:N]
         dZZ = np.zeros((M,N,2))
         dZZ[:,:,0] = dB
         dZZ[:,:,1] = dX
         S = signature(tt,dZZ,K)
    if modedual == "([I],I)-Lift":
        D = int((1-(1+1)**(K+1))/(-1) -1)
        QV = np.zeros(shape = (M,N+1))
        for n in range(N):
            QV[:,n+1] = QV[:,n]+v[:,n]*1/(N+1)
        SD = signatureQV(tt,dI,QV,K)
    if modedual == "(t,I)-Lift":
        K = K+1
        D = int((1-(1+1)**(K+1))/(-1) -1)
        SD = signature(tt,dI,K-1)
    if modedual == '(QV,X)-Lift':
        K = K+1
        D = int((1-(1+1)**(K+1))/(-1) -1)
        dX = X[:,1:N+1]-X[:,0:N]
        dX = dX.reshape(M,N,1)
        QV = np.zeros(shape = (M,N+1))
        for n in range(N):
            QV[:,n+1] = QV[:,n]+v[:,n]*1/(N+1)
        SD = signatureQV(tt,dX,QV,K-1)
    if modedual == '(QV,X,Y)-Lift':
        D = int((1-(2+1)**(K+1))/(-2) -1)
        dX = X[:,1:N+1]-X[:,0:N]
        dZ = Z[:,1:N+1]-Z[:,0:N]
        dZZ = np.zeros((M,N,2))
        dZZ[:,:,0] = dX
        dZZ[:,:,1] = dZ
        QV = np.zeros(shape = (M,N+1))
        for n in range(N):
            QV[:,n+1] = QV[:,n]+v[:,n]*1/(N+1)
        SD = signatureQV(tt,dZZ,QV,K)
    if modedual == '(QV,I,Y)-Lift':
        D = int((1-(2+1)**(K+1))/(-2) -1)
        dZ = Z[:,1:N+1]-Z[:,0:N]
        dZZ = np.zeros((M,N,2))
        dZZ[:,:,0] = dI[:,:,0]
        dZZ[:,:,1] = dZ
        QV = np.zeros(shape = (M,N+1))
        for n in range(N):
            QV[:,n+1] = QV[:,n]+v[:,n]*1/(N+1)
        SD = signatureQV(tt,dZZ,QV,K)
    if modedual == '(QV,W,B)-Lift':
        D = int((1-(2+1)**(K+1))/(-2) -1)
        dZ = Z[:,1:N+1]-Z[:,0:N]
        dZZ = np.zeros((M,N,2))
        dZZ[:,:,0] = dW
        dZZ[:,:,1] = dB
        QV = np.zeros(shape = (M,N+1))
        for n in range(N):
            QV[:,n+1] = QV[:,n]+v[:,n]*1/(N+1)
        SD = signatureQV(tt,dZZ,QV,K)
    if modedual == '(t,V)-Lift':
         D = int((1-(1+1)**(K+1))/(-1) -1)
         dX = np.sqrt(v[:,1:N+1])-np.sqrt(v[:,0:N])
         dX = dX.reshape(M,N,1)
         SD = signature(tt,dX,K)
    if modedual == '(QV,V)-Lift':
         D = int((1-(1+1)**(K+1))/(-1) -1)
         dX = np.sqrt(v[:,1:N+1])-np.sqrt(v[:,0:N])
         dX = dX.reshape(M,N,1)
         QV = np.zeros(shape = (M,N+1))
         for n in range(N):
             QV[:,n+1] = QV[:,n]+v[:,n]*1/(N+1)
         SD = signatureQV(tt,dX,QV,K)
    if modedual == '(t,W)-Lift':
         D = int((1-(1+1)**(K+1))/(-1) -1)
         dX = dW.reshape(M,N,1)
         SD = signature(tt,dX,K)
    if modedual == '(t,W,X)-Lift':
         K = K+1
         D = int((1-(2+1)**(K+1))/(-2) -1)
         dX = X[:,1:N+1]-X[:,0:N]
         dZZ = np.zeros((M,N,2))
         dZZ[:,:,0] = dW
         dZZ[:,:,1] = dX
         SD = signature(tt,dZZ,K-1)
    if modedual == '(t,W,B)-Lift':
         D = int((1-(2+1)**(K+1))/(-2) -1)
         dZZ = np.zeros((M,N,2))
         dZZ[:,:,0] = dB
         dZZ[:,:,1] = dW
         SD = signature(tt,dZZ,K)
    
    DD = int((1-(1+1)**(K))/(-1) -1)
    S4 = np.ones(shape=(M,N,DD+1))
    S4[:,:,1:DD+1] = SD
    S5 = np.zeros(shape = (M,N+1,DD+1))
    S5[:,1:N+1,:] = S4
    S5[:,0,0] = 1
    DoobMG = np.zeros(shape=(M,N+1))
    #DoobMG[:,0] = Z[:,0]
    for n in range(N):
        #DoobMG[:,n+1] = DoobMG[:,n]+np.dot(S5[:,n,:],beta[n,:])*X[:,n]*np.sqrt(v[:,n])*rho*dW[:,n]+np.dot(S5[:,n,:],beta[n,:])*X[:,n]*np.sqrt(v[:,n])*(1-rho**2)*dB[:,n]
        #DoobMG[:,n+1] = DoobMG[:,n]+alphanew(n,S5[:,n,:],N1,beta,T,tt,y0_lo)*X[:,n]*np.sqrt(v[:,n])*dBB[:,n]
        DoobMG[:,n+1] = DoobMG[:,n]+alphainterpol(n,S5[:,n,:],N1,beta,T,tt)*X[:,n]*np.sqrt(v[:,n])*dBB[:,n]
    #DoobMG = X   
    #print(alphainterpol(2,S5[:,2,:],N1,beta,T,tt))
    YYY = Z[:,subindex2]
    #print(np.mean(DoobMG,axis=0))
    y0_up = np.mean(np.max(YYY-DoobMG[:,subindex2],axis=1))
    plt.plot(DoobMG[3,:],'r')
    #print(subindex2)
    plt.plot(Z[3,:])
    print('Doob = 0 Case',np.mean(np.max(YYY,axis=1)))
    print('Dual upper-biased', np.mean(np.max(YYY-DoobMG[:,subindex2],axis=1)))
    print('STD DOOB_MG',np.std(np.max(YYY-DoobMG[:,subindex2],axis=1)))
    print(100*(y0_up-y0_lo)/y0_up,'%')
    return y0_up,DoobMG

        
    
def resimulatedDual(regrDoob1,regrDoob2,T,phi,rho,V,X,dW,dB,dI,N1,K,X0,M2,r):
    M,N = dW.shape
    subindex = [int((j+1)*N/N1) for j in range(N1)]
    subindex2 = [int((j)*N/N1) for j in range(N1+1)]
    D = int((1-(2)**(K+1))/(-1) -1)
    tt = np.linspace(0,T,N+1)
    #Signature of the augmented path (t,I_t) on full-discretized grid
    S = signature(tt, dI, K)
    S = S[:,int(N/N1)-1:len(tt):int(N/N1),:] #Signature only at exercise dates
    S1 = np.ones(shape=(M,N1,D+1))
    S1[:,:,1:D+1] = S #adding Signature value on first level (=1)
    #computing price-process
    YY = np.zeros(shape = (M,N1))
    for k in range(N1):
        YY[:,k] = np.exp(-r*tt[subindex[k]])*phi(X[:,subindex[k]]) 
    value = np.zeros(shape = (M,N1))
    value[:,N1-1] = YY[:,N1-1]
    #integrand for Doob-Martingale
    S2 = np.zeros(shape = (M,N1,D+1+K))
    S2[:,:,K:D+1+K] = S1
    WW = dW.cumsum(axis=1)
    BB = dB.cumsum(axis=1)
    W= np.zeros(shape = (M,N1+1))
    B = np.zeros(shape = (M,N1+1))
    W[:,1:N1+1] = WW[:,int(N/N1)-1:N+1:int(N/N1)]
    W[:,1:N1+1] = BB[:,int(N/N1)-1:N+1:int(N/N1)]
    #Compute Doob-Martingale
    DoobMG = np.zeros(shape=(M,N1+1))
    DoobMG[:,1] = regrDoob1[N1-1]*W[:,1]+regrDoob2[N1-1]*B[:,1]
    for n in range(1,N1):
        DoobMG[:,n+1] = DoobMG[:,n]+regrDoob1[n-1].predict(S2[:,n-1,:])*(W[:,n+1]-W[:,n])+regrDoob2[n-1].predict(S2[:,n-1,:])*(B[:,n+1]-B[:,n])
    YYY = np.zeros(shape = (M,N1+1))
    YYY[:,1:N1+1] = YY
    y0_up = np.mean(np.max(YYY-DoobMG,axis=1))
    print(y0_up)
   
    return y0_up, DoobMG
def RegressionCoefficients(T,phi,rho,v,X,dW,dB,dI,N1,K,X0,r):
    """ Computing Regression coefficient for Longstaff-Schwarz with signatures.
    T = maturity
    rho = correlation factor in [-1,1]
    v = Samples of volatility process (in rBergomi), Mx(N+1) (M=simulations, N = discretization of [0,T])
    dW,dB = increments of two independent Brownian motions, arrays of the form MxN
    dI = increments of the martingale \intv_sdW_s, array of the form MxN
    
    K = depth of Signature
    X0 = starting value for log-price
    """
    
    M,N = dW.shape
    subindex = [int((j+1)*N/N1) for j in range(N1)]
    D = int((1-(2)**(K+1))/(-1) -1)
    tt = np.linspace(0,T,N+1)
    #Signature of the augmented path (t,I_t) on full-discretized grid
    S = signature(tt, dI, K)
    S = S[:,int(N/N1)-1:len(tt):int(N/N1),:] #Signature only at exercise dates
    S1 = np.ones(shape=(M,N1,D+1))
    S1[:,:,1:D+1] = S #adding Signature value on first level (=1)
    #computing price-process
    YY = np.zeros(shape = (M,N1))
    for k in range(N1):
        YY[:,k] = np.exp(-r*tt[subindex[k]])*phi(X[:,subindex[k]]) 
    print(np.mean(YY[:,-1],axis=0))
    value = np.zeros(shape = (M,N1))
    value[:,N1-1] = YY[:,N1-1]
    regr = [0]*(N1-1)
    S2 = np.zeros(shape = (M,N1,D+1+K))
    S2[:,:,K:D+1+K] = S1
    for k in range(K):
        S2[:,:,k] = X[:,subindex]**(k+1)
    for j in reversed(range(1,N1)):
        #only in the money paths
        ITM = []
        for k in range(M):
            if YY[k,j-1]>0:
                ITM.append(k)
        if len(ITM)==0:
            value[:,j-1] = value[:,j]
        else:
            regr[j-1] = Regression(S2[ITM,j-1,:],value[ITM,j],mode = 'linear')
            reg = regr[j-1].predict(S2[ITM,j-1,:])
            #Not actually needed if we resimulate
            value[:,j-1] = value[:,j]
            for m in range(len(ITM)):
                if reg[m] > YY[ITM[m],j-1]:
                    value[ITM[m],j-1] = value[ITM[m],j]
                else:
                    value[ITM[m],j-1] = YY[ITM[m],j-1]
        
    print(np.mean(value[:,0]))
    return regr
def resimluation(regr,T,phi,rho,v,X,dW,dB,dI,N1,K,X0,M2,r):
    """For given regression coefficients in Longstaff-Schwarz algorithm with signatures, resimulate everything as done in RegressionCoefficients,
    and similarly compute the value function with new samples (leads to lower biased values)
    M2 = number of resimulations
    regr = coefficients from regression"""
    
    M,N = dW.shape
    subindex = [int((j+1)*N/N1) for j in range(N1)]
    D = int((1-(2)**(K+1))/(-1) -1)
    tt = np.linspace(0,T,N+1)
    #Signature of the augmented path (t,I_t) on full-discretized grid
    S = signature(tt, dI, K)
    S = S[:,int(N/N1)-1:len(tt):int(N/N1),:] #Signature only at exercise dates
    S1 = np.ones(shape=(M2,N1,D+1))
    S1[:,:,1:D+1] = S #adding Signature value on first level (=1)
    #computing price-process
    YY = np.zeros(shape = (M2,N1))
    for k in range(N1):
        YY[:,k] = np.exp(-r*tt[subindex[k]])*phi(X[:,subindex[k]])
    value = np.zeros(shape = (M2,N1))
    value[:,N1-1] = YY[:,N1-1]
    S2 = np.zeros(shape = (M2,N1,D+1+K))
    S2[:,:,K:D+1+K] = S1
    #additionally define optimal stopping times
    tau = np.zeros(shape = (M2,N1))
    tau[:,N1-1] = N1
    for k in range(K):
        S2[:,:,k] = X[:,subindex]**(k+1)
    
    for j in reversed(range(1,N1)):
        ITM = []
        for k in range(M2):
            if YY[k,j-1]>0:
                ITM.append(k)
        if len(ITM)==0:
            value[:,j-1] = value[:,j]
        else:
            reg = regr[j-1].predict(S2[ITM,j-1,:])
            value[:,j-1] = value[:,j]
            for m in range(len(ITM)):
                if reg[m] > YY[ITM[m],j-1]:
                    value[ITM[m],j-1] = value[ITM[m],j]
                    tau[ITM[m],j-1] = tau[ITM[m],j] #stopping time stays the same (continuation)
                else:
                    value[ITM[m],j-1] = YY[ITM[m],j-1]
                    tau[ITM[m],j-1] = j-1 #new optimal stopping time as payoff dominated regression
            
        
    y0 = np.mean(value[:,0])
    print(y0)
    return y0,value,tau




def DualwithLP(T,phi,rho,v,X,dW,dB,dI,N1,K,strike,X0,r,F,mode):
    """Compute value of optimal stopping for fbm with Hurst h at time T=1, 
    M = simulations
    N = discretization for fbm
    N1 = discretization for max in dual problem
    K = depth of signature"""
    
    M,N = dW.shape
    subindex = [int((j+1)*N/N1) for j in range(N1)]
    subindex2 = [int((j)*N/N1) for j in range(N1+1)]
    
    tt = np.linspace(0,1,N+1)
    Z = np.exp(-r*tt)*phi(X)
    if mode == "(I,QV,X)-Lift":
        D = int((1-(2+1)**(K+1))/(-2) -1)
        QV = np.zeros(shape = (M,N+1))
        dX = X[:,1:N+1]-X[:,0:N]
        for n in range(N):
            QV[:,n+1] = QV[:,n]+v[:,n]*1/(N+1)
        #deltaa = conditionaldeltas(N, X[:,0:N], dI[:,:,0], QV, rho, 80)
        dZ = np.zeros((M,N,2))
        dZ[:,:,0] = dI[:,:,0]
        dZ[:,:,1] = dX
        S = signatureQV(tt,dZ,QV,K)
    if mode == "([I],I)-Lift":
        D = int((1-(1+1)**(K+1))/(-1) -1)
        QV = np.zeros(shape = (M,N+1))
        for n in range(N):
            QV[:,n+1] = QV[:,n]+v[:,n]*1/(N+1)
        S = signatureQV(tt,dI,QV,K)
    if mode == "(t,I)-Lift":
        D = int((1-(1+1)**(K+1))/(-1) -1)
        S = signature(tt,dI,K)
    if mode == "(t,X)-Lift":
        D = int((1-(1+1)**(K+1))/(-1) -1)
        dX = X[:,1:N+1]-X[:,0:N]
        dX = dX.reshape(M,N,1)
        S = signature(tt,dX,K)
    if mode == '(QV,X)-Lift':
        D = int((1-(1+1)**(K+1))/(-1) -1)
        dX = X[:,1:N+1]-X[:,0:N]
        dX = dX.reshape(M,N,1)
        QV = np.zeros(shape = (M,N+1))
        for n in range(N):
            QV[:,n+1] = QV[:,n]+v[:,n]*1/(N+1)
        S = signatureQV(tt,dX,QV,K)
    if mode == '(t,V)-Lift':
        D = int((1-(1+1)**(K+1))/(-1) -1)
        dX = np.sqrt(v[:,1:N+1])-np.sqrt(v[:,0:N])
        dX = dX.reshape(M,N,1)
        S = signature(tt,dX,K)
    if mode == '(QV,V)-Lift':
        D = int((1-(1+1)**(K+1))/(-1) -1)
        dX = np.sqrt(v[:,1:N+1])-np.sqrt(v[:,0:N])
        dX = dX.reshape(M,N,1)
        QV = np.zeros(shape = (M,N+1))
        for n in range(N):
            QV[:,n+1] = QV[:,n]+v[:,n]*1/(N+1)
        S = signatureQV(tt,dX,QV,K)
        SD = np.zeros((M,N+1,D))
        SD[:,1:N+1] = S
    if mode == '(QV,Y)-Lift':
            D = int((1-(1+1)**(K+1))/(-1) -1)
            dX = Z[:,1:N+1]-Z[:,0:N]
            dX = dX.reshape(M,N,1)
            QV = np.zeros(shape = (M,N+1))
            for n in range(N):
                QV[:,n+1] = QV[:,n]+v[:,n]*1/(N+1)
            S = signatureQV(tt,dX,QV,K)
    if mode == '(QV,W,B)-Lift':
        D = int((1-(2+1)**(K+1))/(-2) -1)
        dZ = Z[:,1:N+1]-Z[:,0:N]
        dZZ = np.zeros((M,N,2))
        dZZ[:,:,0] = dW
        dZZ[:,:,1] = dB
        QV = np.zeros(shape = (M,N+1))
        for n in range(N):
            QV[:,n+1] = QV[:,n]+v[:,n]*1/(N+1)
        S = signatureQV(tt,dZZ,QV,K)
    if mode == '(QV,X,Y)-Lift':
        D = int((1-(2+1)**(K+1))/(-2) -1)
        dX = X[:,1:N+1]-X[:,0:N]
        dZ = Z[:,1:N+1]-Z[:,0:N]
        dZZ = np.zeros((M,N,2))
        dZZ[:,:,0] = dX
        dZZ[:,:,1] = dZ
        QV = np.zeros(shape = (M,N+1))
        for n in range(N):
            QV[:,n+1] = QV[:,n]+v[:,n]*1/(N+1)
        S = signatureQV(tt,dZZ,QV,K)
    if mode == '(QV,I,Y)-Lift':
        D = int((1-(2+1)**(K+1))/(-2) -1)
        dZ = Z[:,1:N+1]-Z[:,0:N]
        dZZ = np.zeros((M,N,2))
        dZZ[:,:,0] = dI[:,:,0]
        dZZ[:,:,1] = dZ
        QV = np.zeros(shape = (M,N+1))
        for n in range(N):
            QV[:,n+1] = QV[:,n]+v[:,n]*1/(N+1)
        S = signatureQV(tt,dZZ,QV,K)
    if mode == '(t,V,X)-Lift':
        D = int((1-(2+1)**(K+1))/(-2) -1)
        dZ = np.sqrt(v[:,1:N+1])-np.sqrt(v[:,0:N])
        dX = X[:,1:N+1]-X[:,0:N]
        dZZ = np.zeros((M,N,2))
        dZZ[:,:,0] = dZ
        dZZ[:,:,1] = dX
        S = signature(tt,dZZ,K)
    
        
    if mode == '(QV,V,X)-Lift':
        D = int((1-(2+1)**(K+1))/(-2) -1)
        dZ = np.sqrt(v[:,1:N+1])-np.sqrt(v[:,0:N])
        QV = np.zeros(shape = (M,N+1))
        for n in range(N):
            QV[:,n+1] = QV[:,n]+v[:,n]*1/(N+1)
        dX = X[:,1:N+1]-X[:,0:N]
        dZZ = np.zeros((M,N,2))
        dZZ[:,:,0] = dZ
        dZZ[:,:,1] = dX
        S = signatureQV(tt,dZZ,QV,K)
    if mode == '(t,V,Y)-Lift':
        D = int((1-(2+1)**(K+1))/(-2) -1)
        dZ = np.sqrt(v[:,1:N+1])-np.sqrt(v[:,0:N])
        dX = Z[:,1:N+1]-Z[:,0:N]
        dZZ = np.zeros((M,N,2))
        dZZ[:,:,0] = dZ
        dZZ[:,:,1] = dX
        S = signature(tt,dZZ,K)
    if mode == '(QV,VX)-Lift':
        D = int((1-(1+1)**(K+1))/(-1) -1)
        XV = X*np.sqrt(v)
        dXV = XV[:,1:N+1]-XV[:,0:N]
        dXV = dXV.reshape(M,N,1)
        QV = np.zeros(shape = (M,N+1))
        for n in range(N):
            QV[:,n+1] = QV[:,n]+v[:,n]*1/(N+1)
        S = signatureQV(tt,dXV,QV,K)
    if mode == '(t,I,RT)-Lift':
        D = int((1-(2+1)**(K+1))/(-2) -1)
        RT = np.zeros((M,N+1)) 
        QV = np.zeros(shape = (M,N+1))
        for n in range(N):
            QV[:,n+1] = QV[:,n]+v[:,n]*1/(N+1)
        II = np.zeros((M,N+1))
        II[:,1:N+1] = dI[:,:,0].cumsum(axis=1)
        for k in range(N1+1):
            RT[:,k] = RomanoTouzi(k, X[:,k], phi, np.min(X), np.max(X), rho, II, QV, N1, T, 80, tt)
        dZZ = np.zeros((M,N,2))
        dZZ[:,:,0] = dI[:,:,0]
        dZZ[:,:,1] = RT[:,1:N+1]-RT[:,0:N]
        S = signature(tt,dZZ,K)
            
    #D = int((1-(2)**(K+1))/(-1) -1)
    QV = np.zeros(shape = (M,N+1))
    for n in range(N):
        QV[:,n+1] = QV[:,n]+v[:,n]*1/(N+1)
    
    
    YY = np.zeros(shape = (M,N1))
    for k in range(N1):
        YY[:,k] = np.exp(-r*tt[subindex[k]])*phi(X[:,subindex[k]])
    YYY = np.zeros(shape = (M,N1+1))
    YYY[:,1:N1+1] = YY
    S2 = np.ones(shape = (M,N,D+1+K))
    S2[:,:,K+1:D+1+K] = S
    #add conditional deltas/prices instead of price process (this is contained in the signature anyways)
    #delta1,delta2 = conditionaldeltas(N,phi,X,dI[:,:,0], QV, rho, 80)
    #euro1,euro2 = conditionalprice(N,phi,X,dI[:,:,0], QV, rho, 80)
    #funcc = [RT1,RT2,F[1](X),F[2](X)]
    #for k in range(K):
        #S2[:,:,k+1] = F[k](X[:,1:N+1])
        #S2[:,:,k+1] = F[k](Z[:,1:N+1])
        #S2[:,:,1] = delta2[:,1:N+1]
        #S2[:,:,k+1] = funcc[k][:,0:N]
    RT = np.zeros((M,N+1)) 
    QV = np.zeros(shape = (M,N+1))
    for n in range(N):
        QV[:,n+1] = QV[:,n]+v[:,n]*1/(N+1)
    II = np.zeros((M,N+1))
    II[:,1:N+1] = dI[:,:,0].cumsum(axis=1)
    for k in range(N1+1):
        RT[:,k] = RomanoTouzi(k, X[:,k], phi, rho, II, QV, T, strike, tt,r)
    S3 = np.zeros(shape = (M,N+1,D+1+K))
    S3[:,1:N+1,:] = S2
    S3[:,0,0] = 1
    for k in range(K):
        #S3[:,:,k+1] = funcc[k]
        S3[:,:,k+1] = F[k](X) 
        #S3[:,:,1] = RT
        

    SI = np.zeros(shape = (M,N+1,2*(D+1+K)))
    for k in range(D+1+K):
        for j in range(N):
            SI[:,j+1,k] = SI[:,j,k]+S3[:,j,k]*dW[:,j]
            SI[:,j+1,D+1+K+k] = SI[:,j,D+1+K+k]+S3[:,j,k]*dB[:,j]
    #try only in the money paths
    
    ITM = []
    for m in range(M):
        if np.max(Z[m,:])>0:
            ITM.append(m)
    #xx = dualLPsparse(YYY[ITM,:],tt,N1,N,D,len(ITM),SI[ITM,:,:],K,subindex2)
    xx = dualLPsparse(YYY,tt,N1,N,D,M,SI,K,subindex2)
    DoobMg = np.dot(SI,xx)  
    #var = np.mean(np.max(YYY-np.dot(SI,xx),axis=1)**2)-np.mean(np.max(F[:,0:len(tt):int(N/N1)]-np.dot(Y,xx),axis=1))**2
    print('Dual unbiased Bermuda:',np.mean(np.max(YYY-DoobMg[:,subindex2],axis=1)), 'MC:',np.std(np.max(YYY-DoobMg[:,subindex2],axis=1)))
    #print('Dual unbiased American:',np.mean(np.max(Z-DoobMg,axis=1)), 'MC:', np.std(np.max(Z-DoobMg,axis=1)/np.sqrt(M)))
    counter = 0
    for j in range(M):
        if np.max(YYY[j,:])>0:
            counter = counter +1
    print(counter)
    #plt.plot(DoobMg[ITM[0],:])
    #plt.plot(phi(X)[ITM[0],:])
    return xx

def dualLPsparse(F,tt,N1,N,D,M,Y,K,subindex):
    L = len(Y[0,0,:])
    s = time.time()
    m = gu.Model()
    c = np.zeros(M+L)
    c[0:M] = 1/M
    B = np.zeros(M*(N1+1))
    A = np.zeros(shape = (M*(N1+1),int(L+M)))
    for l in range(M):
        B[l*(N1+1):(l+1)*(N1+1)] = F[l,:]
        A[l*(N1+1):(l+1)*(N1+1),l] = 1
        A[l*(N1+1):(l+1)*(N1+1),M:M+int(L)] = Y[l,subindex,:]
    x = m.addMVar(shape = M+L,name = "x")
    m.setObjective(c @ x, GRB.MINIMIZE)
    m.addConstr(A @ x >= B, name = "c")
    m.Params.LogToConsole = 0
    m.optimize()
    xx = np.zeros(len(m.getVars()))
    o = m.getVars()
    for k in range(len(m.getVars())):
        xx[k] = o[k].x
    ss = time.time()
    #print(ss-s)
    return xx[M:M+int(L)]

def resimDUALMG(xx,T,phi,rho,v,X,dW,dB,dI,N1,K,strike,X0,M2,r,F,mode):
    """Having coefficients in hand, we compute the optimal martingale for resimulated paths. Here M can be very large"""
    
    M,N = dW.shape
    subindex = [int((j+1)*N/N1) for j in range(N1)]
    subindex2 = [int((j)*N/N1) for j in range(N1+1)]
    tt = np.linspace(0,1,N+1)
    Z = np.exp(-r*tt)*phi(X)
    if mode == '(QV,VX)-Lift':
        D = int((1-(1+1)**(K+1))/(-1) -1)
        XV = X*np.sqrt(v)
        dXV = XV[:,1:N+1]-XV[:,0:N]
        dXV = dXV.reshape(M,N,1)
        QV = np.zeros(shape = (M,N+1))
        for n in range(N):
            QV[:,n+1] = QV[:,n]+v[:,n]*1/(N+1)
        S = signatureQV(tt,dXV,QV,K)
    if mode == "(I,QV,X)-Lift":
        D = int((1-(2+1)**(K+1))/(-2) -1)
        QV = np.zeros(shape = (M,N+1))
        dX = X[:,1:N+1]-X[:,0:N]
        for n in range(N):
            QV[:,n+1] = QV[:,n]+v[:,n]*1/(N+1)
        #deltaa = conditionaldeltas(N, X[:,0:N], dI[:,:,0], QV, rho, 80)
        dZ = np.zeros((M,N,2))
        dZ[:,:,0] = dI[:,:,0]
        dZ[:,:,1] = dX
        S = signatureQV(tt,dZ,QV,K)
    if mode == "([I],I)-Lift":
        D = int((1-(1+1)**(K+1))/(-1) -1)
        QV = np.zeros(shape = (M,N+1))
        for n in range(N):
            QV[:,n+1] = QV[:,n]+v[:,n]*1/(N+1)
        S = signatureQV(tt,dI,QV,K)
    if mode == "(t,I)-Lift":
        D = int((1-(1+1)**(K+1))/(-1) -1)
        S = signature(tt,dI,K)
    if mode == '(QV,X)-Lift':
        D = int((1-(1+1)**(K+1))/(-1) -1)
        dX = X[:,1:N+1]-X[:,0:N]
        dX = dX.reshape(M,N,1)
        QV = np.zeros(shape = (M,N+1))
        for n in range(N):
            QV[:,n+1] = QV[:,n]+v[:,n]*1/(N+1)
        S = signatureQV(tt,dX,QV,K)
    if mode == '(QV,Y)-Lift':
            D = int((1-(1+1)**(K+1))/(-1) -1)
            dX = Z[:,1:N+1]-Z[:,0:N]
            dX = dX.reshape(M,N,1)
            QV = np.zeros(shape = (M,N+1))
            for n in range(N):
                QV[:,n+1] = QV[:,n]+v[:,n]*1/(N+1)
            S = signatureQV(tt,dX,QV,K)
    if mode == '(QV,X,Y)-Lift':
        D = int((1-(2+1)**(K+1))/(-2) -1)
        dX = X[:,1:N+1]-X[:,0:N]
        dZ = Z[:,1:N+1]-Z[:,0:N]
        dZZ = np.zeros((M,N,2))
        dZZ[:,:,0] = dX
        dZZ[:,:,1] = dZ
        QV = np.zeros(shape = (M,N+1))
        for n in range(N):
            QV[:,n+1] = QV[:,n]+v[:,n]*1/(N+1)
        S = signatureQV(tt,dZZ,QV,K)
        
    if mode == '(QV,W,B)-Lift':
        D = int((1-(2+1)**(K+1))/(-2) -1)
        dZ = Z[:,1:N+1]-Z[:,0:N]
        dZZ = np.zeros((M,N,2))
        dZZ[:,:,0] = dW
        dZZ[:,:,1] = dB
        QV = np.zeros(shape = (M,N+1))
        for n in range(N):
            QV[:,n+1] = QV[:,n]+v[:,n]*1/(N+1)
        S = signatureQV(tt,dZZ,QV,K)
    if mode == '(QV,I,Y)-Lift':
        D = int((1-(2+1)**(K+1))/(-2) -1)
        dZ = Z[:,1:N+1]-Z[:,0:N]
        dZZ = np.zeros((M,N,2))
        dZZ[:,:,0] = dI[:,:,0]
        dZZ[:,:,1] = dZ
        QV = np.zeros(shape = (M,N+1))
        for n in range(N):
            QV[:,n+1] = QV[:,n]+v[:,n]*1/(N+1)
        S = signatureQV(tt,dZZ,QV,K)
    if mode == '(t,V,X)-Lift':
        D = int((1-(2+1)**(K+1))/(-2) -1)
        dZ = np.sqrt(v[:,1:N+1])-np.sqrt(v[:,0:N])
        dX = X[:,1:N+1]-X[:,0:N]
        dZZ = np.zeros((M,N,2))
        dZZ[:,:,0] = dZ
        dZZ[:,:,1] = dX
        S = signature(tt,dZZ,K)
    if mode == '(t,V*X)-Lift':
        D = int((1-(1+1)**(K+1))/(-1) -1)
        dX = X[:,1:N+1]*np.sqrt(v[:,1:N+1])-X[:,0:N]*np.sqrt(v[:,0:N])
        dX = dX.reshape(M,N,1)
        S = signature(tt,dX,K)
    if mode == '(t,V,Y)-Lift':
        D = int((1-(2+1)**(K+1))/(-2) -1)
        dZ = np.sqrt(v[:,1:N+1])-np.sqrt(v[:,0:N])
        dX = Z[:,1:N+1]-Z[:,0:N]
        dZZ = np.zeros((M,N,2))
        dZZ[:,:,0] = dZ
        dZZ[:,:,1] = dX
        S = signature(tt,dZZ,K)
    if mode == "(t,X)-Lift":
        D = int((1-(1+1)**(K+1))/(-1) -1)
        dX = X[:,1:N+1]-X[:,0:N]
        dX = dX.reshape(M,N,1)
        S = signature(tt,dX,K)
    if mode == '(t,V)-Lift':
        D = int((1-(1+1)**(K+1))/(-1) -1)
        dX = np.sqrt(v[:,1:N+1])-np.sqrt(v[:,0:N])
        dX = dX.reshape(M,N,1)
        S = signature(tt,dX,K)
    if mode == '(QV,V)-Lift':
        D = int((1-(1+1)**(K+1))/(-1) -1)
        dX = np.sqrt(v[:,1:N+1])-np.sqrt(v[:,0:N])
        dX = dX.reshape(M,N,1)
        QV = np.zeros(shape = (M,N+1))
        for n in range(N):
            QV[:,n+1] = QV[:,n]+v[:,n]*1/(N+1)
        S = signatureQV(tt,dX,QV,K)
        
    if mode == '(t,I,RT)-Lift':
        D = int((1-(2+1)**(K+1))/(-2) -1)
        RT = np.zeros((M,N+1)) 
        QV = np.zeros(shape = (M,N+1))
        for n in range(N):
            QV[:,n+1] = QV[:,n]+v[:,n]*1/(N+1)
        II = np.zeros((M,N+1))
        II[:,1:N+1] = dI[:,:,0].cumsum(axis=1)
        for k in range(N1+1):
            RT[:,k] = RomanoTouzi(k, X[:,k], phi, np.min(X), np.max(X), rho, II, QV, N1, T, 80, tt)
        dZZ = np.zeros((M,N,2))
        dZZ[:,:,0] = dI[:,:,0]
        dZZ[:,:,1] = RT[:,1:N+1]-RT[:,0:N]
        plt.plot(RT[0,:])
        S = signature(tt,dZZ,K)

    
    YY = np.zeros(shape = (M,N1))
    for k in range(N1):
        YY[:,k] = np.exp(-r*tt[subindex[k]])*phi(X[:,subindex[k]])
    
    YYY = np.zeros(shape = (M,N1+1))
    YYY[:,1:N1+1] = YY
   
    RT = np.zeros((M,N+1)) 
    QV = np.zeros(shape = (M,N+1))
    for n in range(N):
        QV[:,n+1] = QV[:,n]+v[:,n]*1/(N+1)
    II = np.zeros((M,N+1))
    II[:,1:N+1] = dI[:,:,0].cumsum(axis=1)
    for k in range(N1+1):
        RT[:,k] = RomanoTouzi(k, X[:,k], phi, rho, II, QV, T, strike, tt,r)
    S2 = np.ones(shape = (M,N,D+1+K))
    S2[:,:,K+1:D+1+K] = S
    S3 = np.zeros(shape = (M,N+1,D+1+K))
    S3[:,1:N+1,:] = S2
    S3[:,0,0] = 1
    for k in range(K):
        S3[:,:,k+1] = F[k](X) 
        #S3[:,:,1] = RT
    DoobMG = np.zeros(shape = (M,N+1))
    alpha1 = np.dot(S3,xx[0:(D+1+K)])
    alpha2 = np.dot(S3,xx[D+1+K:2*(D+1+K)])
    for k in range(N):
        DoobMG[:,k+1] = DoobMG[:,k]+alpha1[:,k]*dW[:,k] + alpha2[:,k]*dB[:,k]
        
    #print('Dual American upper bias:',np.mean(np.max(Z-DoobMG,axis=1)),np.std(np.max(Z-DoobMG,axis=1))/np.sqrt(M))
    ZZZ  = DoobMG[:,subindex2]
    print('Dual upper-biased Bermuda:',np.mean(np.max(Z[:,subindex2]-ZZZ,axis=1)), 'MC:',np.std(np.max(Z[:,subindex2]-ZZZ,axis=1)))
    y0up = np.mean(np.max(YYY-ZZZ,axis=1))
    plt.plot(DoobMG[0,:])
    plt.plot(Z[0,:])
    return y0up, DoobMG
    
def DualPricingOnlyND(M,M2,N,N1,T,phi,rho,K,strike,X0,H,xi,eta,r,mode,d):
    """Pricing American options (Bermuddan options) in rBergomi using only dual-method.
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
    r = interest-rate (add interest rates in prices.... (TBD))
    """
    D = int((1-(d+1)**(K+1))/(-d) -1)
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
    dB = rB.dB(dW1, dW2, rho = -0.9)
    X = rB.S(V, dB) #array of Mx(N+1)
    X = X0*X
    del dB
    #compute discounted process
    for k in range(N+1):
        X[:,k] = np.exp(r*tt[k])*X[:,k] #discounted price-process
    print('European:',np.mean(np.exp(-r)*phi(X[:,-1])), 'MC-Error:', np.std(np.exp(-r)*phi(X[:,-1]))/np.sqrt(M))
    I = np.zeros(shape = (M,int(T*N)+1))
    for n in range(int(T*N)):
        I[:,n+1] = I[:,n] + np.sqrt(V[:,n])*dW1[:,n,0]
    dI = I[:,1:int(T*N)+1]-I[:,0:int(T*N)]
    dI = dI.reshape(M,int(T*N),1)
    #compute LS-regression, to get optimal coefficients
    #regr = RegressionCoefficients(T,phi,rho,V,X,dW1[:,:,0],dW2,dI,N1,K,X0,r)
    F = [0]*K #laguerre polynomial
    for k in range(K):
        F[k] = sc.special.genlaguerre(k+1, 0)
    xx = DualwithLP(T,phi,rho,V,X,dW1[:,:,0],dW2,dI,N1,K,strike,X0,r,F,mode)
    print(xx)
    del rB,dW1,dW2,Y,V,X,I,dI 
    
    #Resimulation
    rB = rBergomi(N, M2, T, -0.5+H)
    #two independent Brownian motion increments
    dW1 = rB.dW1()
    dW2 = rB.dW2()
    #volatility process V,array of Mx(N+1)
    Y = rB.Y(dW1)
    V = rB.V(Y, xi, eta)
    #price-process in rBergomi
    dB = rB.dB(dW1, dW2, rho = -0.9)
    X = rB.S(V, dB) #array of Mx(N+1)
    X = X0*X
    del dB
    #compute discounted process
    for k in range(N+1):
        X[:,k] = np.exp(r*tt[k])*X[:,k]
    #discounted price-process
    print('European 2:',np.mean(np.exp(-r)*phi(X[:,-1])), 'MC-Error:', np.std(np.exp(-r)*phi(X[:,-1]))/np.sqrt(M2))
    
    I = np.zeros(shape = (M2,int(T*N)+1))
    for n in range(int(T*N)):
        I[:,n+1] = I[:,n] + np.sqrt(V[:,n])*dW1[:,n,0]
    dI = I[:,1:int(T*N)+1]-I[:,0:int(T*N)]
    dI = dI.reshape(M2,int(T*N),1)
    #compute lower biased and upper biased prices, by computing Doob-Martingales
    y0up, DoobMG = resimDUALMG(xx, T, phi, rho, V, X, dW1[:,:,0], dW2, dI, N1, K,strike, X0, M2, r, F,mode)
    
    
    return y0up, DoobMG
    

#Dual method without Signature, assume Markovian

def DualPricingOnlyNDOS(M,M2,N,N1,T,KK,phi,rho,K,strike,X0,H,xi,eta,r,mode,d):
    """Pricing American options (Bermuddan options) in rBergomi using only dual-method.
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
    r = interest-rate (add interest rates in prices.... (TBD))
    """
    D = int((1-(d+1)**(KK+1))/(-d) -1)
    tt = np.linspace(0,T,N+1)
    
    #Using rBergomi-Package for volatility and Brownian motions
    X,V,I,dI,dW1,dW2,dB = SimulationofrBergomi(M,N,T,phi,rho,K,X0,H,xi,eta,r)
    Z = np.exp(-r*tt)*phi(X)
    #compute discounted process
    print('European:',np.mean(np.exp(-r)*phi(X[:,-1])), 'MC-Error:', np.std(np.exp(-r)*phi(X[:,-1]))/np.sqrt(M))
    #compute LS-regression, to get optimal coefficients
    #regr = RegressionCoefficients(T,phi,rho,V,X,dW1[:,:,0],dW2,dI,N1,K,X0,r)
    #polynomials u
    
    
    DD = int((K+1)*(K+2)/2)+1
    S = np.zeros((M,N+1,DD))
    for k in range(K+1):
        for j in range(0,k+1):
            C = np.zeros((K+1,K+1))
            C[k,j] = 1
            S[:,:,int(k*(k+1)/2+j)] = np.polynomial.laguerre.lagval2d(X, np.sqrt(V), C)
            #S[:,:,int(k*(k+1)/2+j)] = np.sqrt(V)**j*X**(k-j)"""
    S[:,:,-1] = Z
    
    if D == 0:
        xx,SI,A,B = DualwithLPOS(T,phi,rho,V,X,dW1[:,:,0],dW2,N1,K,strike,X0,r,S)
    else:
        QV = np.zeros(shape = (M,N+1))
        for n in range(N):
            QV[:,n+1] = QV[:,n]+V[:,n]*1/(N+1)
        dX = X[:,1:N+1]-X[:,0:N]
        dX = dX.reshape(M,N,1)
        dI = dI.reshape(M,N,1)
        dWW = dW1[:,:,0].reshape(M,N,1)
        dY = Z[:,1:N+1]-Z[:,0:N]
        dV = np.sqrt(V[:,1:N+1])-np.sqrt(V[:,0:N])
        dV = dV.reshape(M,N,1)
        dZ = np.zeros((M,N,2))
        dZ[:,:,0] = dWW[:,:,0]
        dZ[:,:,1] = dW2
        #dZ[:,:,1] = dX[:,:,0]
        #Sig = signature(tt,dZ,KK)
        #Sig = signatureQV(tt,dV,QV,KK)
        Sig = signature(tt,dX,KK)
        
        """
        XX = np.zeros((M,N+1,3))
        W = np.zeros((M,N+1))
        W[:,1:N+1] = np.cumsum(dW1[:,:,0],axis=1)
        XX[:,:,0] = tt
        XX[:,:,1] = X
        XX[:,:,2] = W
        Sigg = full_log_signature(XX,KK )
        D = len(Sigg[0,0,:])"""
        Sigg = np.zeros((M,N+1,D))
        Sigg[:,1:N+1,:] = Sig
        SS = np.zeros((M,N+1,DD+D))
        SS[:,:,0:DD]= S
        SS[:,:,DD:DD+D+1] = Sigg
        for j in range(len(Sigg[0,0,:])):
            Sigg[:,:,j] = Sigg[:,:,j]*np.sqrt(V)*X
        xx,SI= DualwithLPOS(T,phi,rho,V,X,dW1[:,:,0],dW2,N1,K,strike,X0,r,SS)
    
    print(xx)
    del dI,dW1,dW2,V,X,I,dB
    M = M2
    #Resimulation
    X,V,I,dI,dW1,dW2,dB = SimulationofrBergomi(M,N,T,phi,rho,K,X0,H,xi,eta,r)
    Z = np.exp(-r*tt)*phi(X)
    #discounted price-process
    print('European 2:',np.mean(np.exp(-r)*phi(X[:,-1])), 'MC-Error:', np.std(np.exp(-r)*phi(X[:,-1]))/np.sqrt(M2))
    DD = int((K+1)*(K+2)/2)+1
    S = np.zeros((M,N+1,DD))
    for k in range(K+1):
        for j in range(0,k+1):
            C = np.zeros((K+1,K+1))
            C[k,j] = 1
            S[:,:,int(k*(k+1)/2+j)] = np.polynomial.laguerre.lagval2d(X, np.sqrt(V), C)
            #S[:,:,int(k*(k+1)/2+j)] = np.sqrt(V)**j*X**(k-j)
    S[:,:,-1] = Z
    if D == 0:
        y0up, DoobMG = resimDUALMGOS(xx, T, phi, rho, V, X, dW1[:,:,0], dW2, N1, K,strike, X0, M2, r,S)
    else:
        QV = np.zeros(shape = (M,N+1))
        for n in range(N):
            QV[:,n+1] = QV[:,n]+V[:,n]*1/(N+1)
        dX = X[:,1:N+1]-X[:,0:N]
        dX = dX.reshape(M,N,1)
        dI = dI.reshape(M,N,1)
        dWW = dW1[:,:,0].reshape(M,N,1)
        dY = Z[:,1:N+1]-Z[:,0:N]
        dV = np.sqrt(V[:,1:N+1])-np.sqrt(V[:,0:N])
        dV = dV.reshape(M,N,1)
        dZ = np.zeros((M,N,2))
        dZ[:,:,0] = dWW[:,:,0]
        dZ[:,:,1] = dW2
        #dZ[:,:,1] = dX[:,:,0]
        #Sig = signature(tt,dZ,KK)
        #Sig = signatureQV(tt,dV,QV,KK)
        Sig = signature(tt,dX,KK)
        """
        XX = np.zeros((M,N+1,3))
        W = np.zeros((M,N+1))
        W[:,1:N+1] = np.cumsum(dW1[:,:,0],axis=1)
        XX[:,:,0] = tt
        XX[:,:,1] = X
        XX[:,:,2] = W
        Sigg = full_log_signature(XX,KK )
        D = len(Sigg[0,0,:])"""
        
        
        Sigg = np.zeros((M,N+1,D))
        Sigg[:,1:N+1,:] = Sig
        SS = np.zeros((M,N+1,DD+D))
        SS[:,:,0:DD]= S
        SS[:,:,DD:DD+D+1] = Sigg
        y0up, DoobMG = resimDUALMGOS(xx, T, phi, rho, V, X, dW1[:,:,0], dW2, N1, K,strike, X0, M2, r,SS)
    

    #compute lower biased and upper biased prices, by computing Doob-Martingales
    
    return y0up, DoobMG


#only with polynomial integrands
def DualwithLPOS(T,phi,rho,v,X,dW1,dW2,N1,K,strike,X0,r,S):
    """Compute value of optimal stopping for fbm with Hurst h at time T=1, 
    M = simulations
    N = discretization for fbm
    N1 = discretization for max in dual problem
    K = depth of signature"""
    
    M,N,DDD = S.shape 
    N = N-1
    subindex = [int((j+1)*N/N1) for j in range(N1)]
    subindex2 = [int((j)*N/N1) for j in range(N1+1)]
    tt = np.linspace(0,1,N+1)
    Z = np.exp(-r*tt)*phi(X)
    #plt.plot(Z[0,:])
    YY =Z[:,subindex2]
    SI = np.zeros(shape = (M,N+1,2*DDD))
    for k in range(DDD):
        for n in range(N):
            SI[:,n+1,k] = SI[:,n,k]+S[:,n,k]*dW1[:,n]
            SI[:,n+1,DDD+k] = SI[:,n,DDD+k]+S[:,n,k]*dW2[:,n]
    xx= dualLPsparseOS(YY,tt,N1,N,M,SI,K,subindex2)
    DoobMg = np.dot(SI,xx)  
    #var = np.mean(np.max(YYY-np.dot(SI,xx),axis=1)**2)-np.mean(np.max(F[:,0:len(tt):int(N/N1)]-np.dot(Y,xx),axis=1))**2
    print('Dual unbiased Bermuda:',np.mean(np.max(YY-DoobMg[:,subindex2],axis=1)), 'MC:',np.std(np.max(YY-DoobMg[:,subindex2],axis=1))/np.sqrt(M))
    plt.plot(DoobMg[0,:])
    plt.plot(phi(X)[0,:])
    return xx,SI

def dualLPsparseOS(F,tt,N1,N,M,Y,K,subindex):
    L = len(Y[0,0,:])
    s = time.time()
    m = Model()
    c = np.zeros(M+L)
    c[0:M] = 1/M
    #print(c)
    #plt.plot(Y[0,:,:])
    B = np.zeros(M*(N1+1))
    A = np.zeros(shape = (M*(N1+1),int(L+M)))
    for l in range(M):
        B[l*(N1+1):(l+1)*(N1+1)] = F[l,:]
        A[l*(N1+1):(l+1)*(N1+1),l] = 1
        A[l*(N1+1):(l+1)*(N1+1),M:M+int(L)] = Y[l,subindex,:]
    x = m.addMVar(shape = M+L,name = "x")
    m.setObjective(c @ x, GRB.MINIMIZE)
    m.addConstr(A @ x >= B, name = "c")
    m.Params.LogToConsole = 0
    m.optimize()
    xx = np.zeros(len(m.getVars()))
    o = m.getVars()
    for k in range(len(m.getVars())):
        xx[k] = o[k].x
    ss = time.time()
    print(np.dot(c,xx))
    #print(ss-s)
    return xx[M:M+int(L)]

def resimDUALMGOS(xx,T,phi,rho,v,X,dW1,dW2,N1,K,strike,X0,M2,r,S):
    """Having coefficients in hand, we compute the optimal martingale for resimulated paths. Here M can be very large"""
    
    M,N,DDD = S.shape
    N = N-1
    subindex = [int((j+1)*N/N1) for j in range(N1)]
    subindex2 = [int((j)*N/N1) for j in range(N1+1)]
    tt = np.linspace(0,1,N+1)
    Z = np.exp(-r*tt)*phi(X)

    YYY = Z[:,subindex2]
    #DoobMG = np.zeros(shape = (M,N+1))
    SI = np.zeros(shape = (M,N+1,2*DDD))
    for k in range(DDD):
        for n in range(N):
            SI[:,n+1,k] = SI[:,n,k]+S[:,n,k]*dW1[:,n]
            SI[:,n+1,DDD+k] = SI[:,n,DDD+k]+S[:,n,k]*dW2[:,n]
    DoobMG = np.dot(SI,xx)  
    """
    alpha1 = np.dot(S,xx[0:DDD])
    alpha2 = np.dot(S,xx[DDD:2*DDD+1])
    for k in range(N):
        DoobMG[:,k+1] = DoobMG[:,k]+alpha1[:,k]*dW[:,k] + alpha2[:,k]*dB[:,k]
    #print('Dual American upper bias:',np.mean(np.max(Z-DoobMG,axis=1)),np.std(np.max(Z-DoobMG,axis=1))/np.sqrt(M))
    """
    ZZZ  = DoobMG[:,subindex2]
    print('Dual upper-biased Bermuda:',np.mean(np.max(Z[:,subindex2]-ZZZ,axis=1)), 'MC:',np.std(np.max(Z[:,subindex2]-ZZZ,axis=1))/np.sqrt(M))
    y0up = np.mean(np.max(YYY-ZZZ,axis=1))
    #plt.plot(DoobMG[0,:])
    #plt.plot(Z[0,:])
    return y0up, DoobMG


def FinalPrimalDual(M1,M2,M3,N,N1,T,KK_primal,KK_dual,K,phi,rho,strike,X0,H,xi,eta,r,d):
    """Computes lower and upper bounds for american options in rBergomi, using LS and SAA with signatures.
    M1 = Number of Samples for Linear Regression
    M2 = NUmber of Samples for LP Dual (M2<<M1)
    M3 = Number of Samples for resimulation in both
    N,N = discretization and exercise dates
    T = maturity
    KK = number of Legendre-Polynomials in Basis
    K = Depth of Signature of (t,W,B)
    phi = payoff
    rho = correlation
    strike = strike of Put option
    X0 = initial value of X
    H,xi,eta = paramter for rBergomi volatility
    r = interest rate
    Output: Interval for American option price"""
    D = int((1-(d+1)**(K+1))/(-d) -1)
    tt = np.linspace(0,T,N+1)
    subindex = [int((j+1)*N/N1) for j in range(N1)]
    subindex2 = [int((j)*N/N1) for j in range(N1+1)]
    X,V,I,dI,dW1,dW2,dB = SimulationofrBergomi(M1,N,T,phi,rho,strike,X0,H,xi,eta,r)
    #discounted Payoff
    Z = np.exp(-r*tt)*phi(X)
    if KK_primal == 0:
        #Compute full signature of (t,W,B)
        dBM = np.zeros((M1,N,2))
        dBM[:,:,0]=dW1[:,:,0]
        dBM[:,:,1]=dW2
        QV = np.zeros(shape = (M1,N+1))
        for n in range(N):
            QV[:,n+1] = QV[:,n]+V[:,n]*1/(N+1)
        dX = X[:,1:N+1]-X[:,0:N]
        SS = signatureQV(tt,dX.reshape(M,N,1),QV,K)
        S = np.zeros((M1,N+1,D))
        S[:,1:N+1,:] = SS
        Sig = np.ones((M1,N+1,D+1))
        Sig[:,:,1:D+1] = S
        Basis_primal = Sig
    if KK_dual == 0:
        #Compute full signature of (t,W,B)
        dBM = np.zeros((M1,N,2))
        dBM[:,:,0]=dW1[:,:,0]
        dBM[:,:,1]=dW2
        QV = np.zeros(shape = (M1,N+1))
        for n in range(N):
            QV[:,n+1] = QV[:,n]+V[:,n]*1/(N+1)
        SS = signatureQV(tt,dBM,QV,K)
        S = np.zeros((M1,N+1,D))
        S[:,1:N+1,:] = SS
        Sig = np.ones((M1,N+1,D+1))
        Sig[:,:,1:D+1] = S
        Basis_dual = Sig
    if K == 0:
        DD_primal = int((KK_primal+1)*(KK_primal+2)/2) #Number of polynomials 2 dim
        P_primal = np.zeros((M1,N+1,DD_primal))
        for k in range(KK_primal+1):
            for j in range(0,k+1):
                C = np.zeros((KK_primal+1,KK_primal+1))
                C[k,j] = 1
                P_primal[:,:,int(k*(k+1)/2+j)] = np.polynomial.laguerre.lagval2d(X, np.sqrt(V), C)
        Basis_primal = P_primal
        
        DD_dual = int((KK_dual+1)*(KK_dual+2)/2) #Number of polynomials 2 dim
        P_dual= np.zeros((M1,N+1,DD_dual))
        for k in range(KK_dual+1):
            for j in range(0,k+1):
                C = np.zeros((KK_dual+1,KK_dual+1))
                C[k,j] = 1
                P_dual[:,:,int(k*(k+1)/2+j)] = np.polynomial.laguerre.lagval2d(X, np.sqrt(V), C)
        Basis_dual = P_dual
    if K>0 and KK_primal > 0 and KK_dual:
        #Compute full signature of (t,W,B)
        dBM = np.zeros((M1,N,2))
        dBM[:,:,0]=dW1[:,:,0]
        dBM[:,:,1]=dW2
        QV = np.zeros(shape = (M1,N+1))
        for n in range(N):
            QV[:,n+1] = QV[:,n]+V[:,n]*1/(N+1)
        dV = np.sqrt(V[:,1:N+1])-np.sqrt(V[:,0:N])
        dX = X[:,1:N+1]-X[:,0:N]
        SS = signature(tt,dX.reshape(M1,N,1),K)
        #SS = signatureQV(tt,dBM,QV,K)
        #SS = signatureQV(tt,dV.reshape(M1,N,1),QV,K)
        S = np.zeros((M1,N+1,D))
        S[:,1:N+1,:] = SS
        #Sig = np.ones((M1,N+1,D+1))
        #Sig[:,:,1:D+1] = S
        #Primal Basis functions
        
        DD_primal = int((KK_primal+1)*(KK_primal+2)/2)+1 #Number of polynomials 2 dim
        P_primal = np.zeros((M1,N+1,DD_primal))
        for k in range(KK_primal+1):
            for j in range(0,k+1):
                C = np.zeros((KK_primal+1,KK_primal+1))
                C[k,j] = 1
                P_primal[:,:,int(k*(k+1)/2+j)] = np.polynomial.laguerre.lagval2d(X, np.sqrt(V), C)
        P_primal[:,:,-1] = phi(X)
        Basis_primal = np.zeros((M1,N+1,DD_primal+D))
        Basis_primal[:,:,0:D]=S
        Basis_primal[:,:,D:DD_primal+D+1] = P_primal
        
        
    
        DD_dual = int((KK_dual+1)*(KK_dual+2)/2)+1 #Number of polynomials 2 dim
        P_dual= np.zeros((M1,N+1,DD_dual))
        for k in range(KK_dual+1):
            for j in range(0,k+1):
                C = np.zeros((KK_dual+1,KK_dual+1))
                C[k,j] = 1
                P_dual[:,:,int(k*(k+1)/2+j)] = np.polynomial.laguerre.lagval2d(X, np.sqrt(V), C)
        P_dual[:,:,-1] = phi(X)
        Basis_dual = np.zeros((M1,N+1,DD_dual+D))
        Basis_dual[:,:,0:D]=S
        Basis_dual[:,:,D:DD_dual+D+1] = P_dual
    
   
    print('number of basis-functions primal:',D+DD_primal)
    #Linear Regression for Primal
    Basis_Reg = Basis_primal[:,subindex,:]
    YY = phi(X[:,subindex])
    value = YY[:,-1]
    regr = [0]*(N1-1)
    dtt = np.exp(-r*(T/(N1+1)))
    for j in reversed(range(1,N1)):
        #only in the money paths
        ITM = []
        value = value*dtt
        for k in range(M1):
            if YY[k,j-1]>0:
                ITM.append(k)
        
        if len(ITM)==0:
            continue
        else:
            regr[j-1] = LinearRegression().fit(Basis_Reg[ITM,j-1,:],value[ITM])
            print(regr[j-1].score(Basis_Reg[ITM,j-1,:],value[ITM]))
            reg = regr[j-1].predict(Basis_Reg[ITM,j-1,:])
            print('RMSE',np.mean((reg - value[ITM])**2))
            if j == 7:
                print(regr[j-1].coef_)
            for m in range(len(ITM)):
                if reg[m] > YY[ITM[m],j-1]:
                    continue
                else:
                    value[ITM[m]] = YY[ITM[m],j-1]
    print('biased LS value',np.mean(value))
    #Martingale-Construction for Dual
    
    SI = np.zeros((M2,N+1,2*(D+DD_dual)))

    for k in range(DD_dual+D):
        for n in range(N):
            SI[:,n+1,k] = SI[:,n,k]+Basis_dual[0:M2,n,k]*dW1[0:M2,n,0]
            SI[:,n+1,DD_dual+D+k] = SI[:,n,DD_dual+D+k]+Basis_dual[0:M2,n,k]*dW2[0:M2,n]
    U = np.random.uniform(0,5,M1)
    L = np.zeros((M1,N+1))
    L[:,1:N+1]= Z[:,1:N+1]
    L[:,0] = U

    xx= dualLPsparseOS(L[0:M2,subindex2],tt,N1,N,M2,SI,K,subindex2)
    print('biased SAA value',np.mean(np.max(Z[0:M2,subindex2]-np.dot(SI,xx)[:,subindex2],axis=1)))
    print('standard-deviation',np.std(np.max(Z[0:M2,subindex2]-np.dot(SI,xx)[:,subindex2],axis=1)))
    print(xx)
    del X,V,I,dI,dW1,dW2,dB,Basis_Reg,Basis_primal
    X,V,I,dI,dW1,dW2,dB = SimulationofrBergomi(M3,N,T,phi,rho,strike,X0,H,xi,eta,r)
    #discounted Payoff
    Z = np.exp(-r*tt)*phi(X)
    if KK_primal == 0:
        #Compute full signature of (t,W,B)
        dBM = np.zeros((M3,N,2))
        dBM[:,:,0]=dW1[:,:,0]
        dBM[:,:,1]=dW2
        QV = np.zeros(shape = (M3,N+1))
        for n in range(N):
            QV[:,n+1] = QV[:,n]+V[:,n]*1/(N+1)
        SS = signatureQV(tt,dBM,QV,K)
        S = np.zeros((M3,N+1,D))
        S[:,1:N+1,:] = SS
        Sig = np.ones((M3,N+1,D+1))
        Sig[:,:,1:D+1] = S
        Basis_primal = Sig
    if KK_dual == 0:
        #Compute full signature of (t,W,B)
        dBM = np.zeros((M3,N,2))
        dBM[:,:,0]=dW1[:,:,0]
        dBM[:,:,1]=dW2
        QV = np.zeros(shape = (M3,N+1))
        for n in range(N):
            QV[:,n+1] = QV[:,n]+V[:,n]*1/(N+1)
        SS = signatureQV(tt,dBM,QV,K)
        S = np.zeros((M3,N+1,D))
        S[:,1:N+1,:] = SS
        Sig = np.ones((M3,N+1,D+1))
        Sig[:,:,1:D+1] = S
        Basis_dual = Sig
    if K == 0:
        DD_primal = int((KK_primal+1)*(KK_primal+2)/2) #Number of polynomials 2 dim
        P_primal = np.zeros((M3,N+1,DD_primal))
        for k in range(KK_primal+1):
            for j in range(0,k+1):
                C = np.zeros((KK_primal+1,KK_primal+1))
                C[k,j] = 1
                P_primal[:,:,int(k*(k+1)/2+j)] = np.polynomial.laguerre.lagval2d(X, np.sqrt(V), C)
        Basis_primal = P_primal
        
        DD_dual = int((KK_dual+1)*(KK_dual+2)/2) #Number of polynomials 2 dim
        P_dual= np.zeros((M3,N+1,DD_dual))
        for k in range(KK_dual+1):
            for j in range(0,k+1):
                C = np.zeros((KK_dual+1,KK_dual+1))
                C[k,j] = 1
                P_dual[:,:,int(k*(k+1)/2+j)] = np.polynomial.laguerre.lagval2d(X, np.sqrt(V), C)
        Basis_dual = P_dual
    if K>0 and KK_primal > 0 and KK_dual:
        #Compute full signature of (t,W,B)
        dBM = np.zeros((M3,N,2))
        dBM[:,:,0]=dW1[:,:,0]
        dBM[:,:,1]=dW2
        QV = np.zeros(shape = (M3,N+1))
        for n in range(N):
            QV[:,n+1] = QV[:,n]+V[:,n]*1/(N+1)
        dV = np.sqrt(V[:,1:N+1])-np.sqrt(V[:,0:N])
        dX = X[:,1:N+1]-X[:,0:N]
        SS = signature(tt,dX.reshape(M3,N,1),K)
        #SS = signatureQV(tt,dV.reshape(M3,N,1),QV,K)
        #SS = signatureQV(tt,dBM,QV,K)
        S = np.zeros((M3,N+1,D))
        S[:,1:N+1,:] = SS
        #Sig = np.ones((M1,N+1,D+1))
        #Sig[:,:,1:D+1] = S
        #Primal Basis functions
        

        
        DD_primal = int((KK_primal+1)*(KK_primal+2)/2)+1 #Number of polynomials 2 dim
        P_primal = np.zeros((M3,N+1,DD_primal))
        for k in range(KK_primal+1):
            for j in range(0,k+1):
                C = np.zeros((KK_primal+1,KK_primal+1))
                C[k,j] = 1
                P_primal[:,:,int(k*(k+1)/2+j)] = np.polynomial.laguerre.lagval2d(X, np.sqrt(V), C)
        P_primal[:,:,-1] = phi(X)
        Basis_primal = np.zeros((M3,N+1,DD_primal+D))
        Basis_primal[:,:,0:D]=S
        Basis_primal[:,:,D:DD_primal+D+1] = P_primal

        DD_dual = int((KK_dual+1)*(KK_dual+2)/2)+1 #Number of polynomials 2 dim
        P_dual= np.zeros((M3,N+1,DD_dual))
        for k in range(KK_dual+1):
            for j in range(0,k+1):
                C = np.zeros((KK_dual+1,KK_dual+1))
                C[k,j] = 1
                P_dual[:,:,int(k*(k+1)/2+j)] = np.polynomial.laguerre.lagval2d(X, np.sqrt(V), C)
        P_dual[:,:,-1] = phi(X)
        Basis_dual = np.zeros((M3,N+1,DD_dual+D))
        Basis_dual[:,:,0:D]=S
        Basis_dual[:,:,D:DD_dual+D+1] = P_dual
        
        
        

    
    #lower bounded primal:
    Basis_Reg = Basis_primal[:,subindex,:]
    YY = phi(X[:,subindex])
    ttt = np.linspace(0,1,N1+1)
    value = np.zeros(M3)
    value1 = YY[:,-1]
    Regression = np.zeros((M3,N1))
    for n in range(N1-1):
        if regr[n] == 0:
            Regression[:,n] = 10**10
        else:
            Regression[:,n] = regr[n].predict(Basis_Reg[:,n,:])
        
    for m in range(M3):
        j = 0
        while YY[m,j] == 0 or YY[m,j]<Regression[m,j]:
            j = j+1
            if j == N1-1:
                break
        value[m] =np.exp(-r*(ttt[j+1]-ttt[1]))*YY[m,j]
    y0_lb = np.mean(value)
    print('lower-biasd LS',y0_lb)    
    
    #upper bounded dual:
    
    SI = np.zeros((M3,N+1,2*(D+DD_dual)))

    for k in range(DD_dual+D):
        for n in range(N):
            SI[:,n+1,k] = SI[:,n,k]+Basis_dual[:,n,k]*dW1[:,n,0]
            SI[:,n+1,DD_dual+D+k] = SI[:,n,DD_dual+D+k]+Basis_dual[:,n,k]*dW2[:,n]
    Doob = np.dot(SI,xx)
    y0_ub = np.mean(np.max(Z[:,subindex2]-Doob[:,subindex2],axis=1))
    print('upper-biased SAA',y0_ub)
    return y0_lb
        
            

    
    
    
    
    
    
    
    
    
    
    