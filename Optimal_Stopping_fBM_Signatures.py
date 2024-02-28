#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 24 10:19:10 2023

Main functions computing lower and upper bounds with signatures as described in "Primal and dual optimal stopping with signatures",
The details for the primal problem can be found in Section 3.1
The details for the dual problem can be found in Section 3.2

@author: pelizzari

"""
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
import scipy as sc
from helpfunctions import SignatureFull, LP_solver
import time
from FBM_package import FBM
import matplotlib.pyplot as plt


def UpperBoundfBm(M,M2,N,N1,T,K,h):
    """Derive upper-bounds to optimal stopping problem with N1 exercise dates on [0,T]
    Input:

    M = Number of samples used in first simulation, along which we solve the LP
    M2 = Number of samples used for resimulation (typically M2 >> M) 
    N1 = Number of exercise points in [0,T] in the sense of Bermudda-options
    N = Multiple of N1, number of discretization points for the grid [0,T] (typically N>>N1)
    T = Final time for the stopping problem
    K = Depth of linear (time-extended) signature
    h = Hurst parameter
    
    Output: 
    y0 = true-upper bound for the optimal stopping problem
    STD = sample standard deviation of estimator
    run_time= time (in seconds) needed for computation
    
    """
    s = time.time()
    #D = Number of entries of signature of 2-dim path (t,X_t)
    D = int((1-(1+1)**(K+1))/(-1) -1)
    
    #exercise dates indices in discretization dates
    subindex2 = [int((j)*N/N1) for j in range(N1+1)]
    
    
    #generate samples of FBM paths and increments, and corresponding Brownian incremements
    
    tt = np.linspace(0,T,N+1)
    F,dfBm,dW = FBM(N,M,h,method='cholesky').fbm()
    F = F.transpose()
    dW = dW.transpose()
    dfBm = dfBm.transpose()
    #compute linear signature using ii-package
    
    S = SignatureFull(tt, dfBm.reshape(M,N,1), K)

    #Construct familiy of basis-martingales \int <S,l>dW used for the linear programm, using Euler-approximation
    MG = np.zeros((M,N+1,1+D))
    for dd in range(D):
        for n in range(N):
            MG[:,n+1,dd] = MG[:,n,dd]+S[:,n,dd]*dW[:,n]
    
    #Solve linear programm described in Remark 3.9, using Gurobi optimization https://www.gurobi.com
    
    xx = LP_solver(F,tt,N1,N,D,M,MG[:,subindex2,:],subindex2)
    
    #To get true-upper bounds, we resimulate paths and use the computed coefficients xx to compute upper-bound 
    del F,dW,dfBm,MG
   
    #Resimulate all the paths similar as before, with new sample size M2
    F,dfBm,dW = FBM(N,M2,h,method='cholesky').fbm()
    F = F.transpose()
    dW = dW.transpose()
    dfBm = dfBm.transpose()
    
    #Similar we compute signature and martingales for the new samples
    S = SignatureFull(tt, dfBm.reshape(M2,N,1), K)
    MG = np.zeros((M2,N+1,D+1))
    for dd in range(D):
        for n in range(N):
            MG[:,n+1,dd] = MG[:,n,dd]+S[:,n,dd]*dW[:,n]
    #Now we can compute the true upper-bounds and standard deviation
    y0 = np.mean(np.max(F[:,subindex2]-np.dot(MG,xx)[:,subindex2],axis=1))
    STD = np.std(np.max(F[:,subindex2]-np.dot(MG,xx)[:,subindex2],axis=1))
    print('true upper bound',np.mean(np.max(F[:,subindex2]-np.dot(MG,xx)[:,subindex2],axis=1)),'with STD', np.std(np.max(F[:,subindex2]-np.dot(MG,xx)[:,subindex2],axis=1)))
    
    ss = time.time()
    run_time = ss-s
    return y0,STD,run_time



def LongstaffSchwartzfBm(M,M2,N,N1,T,K,h):
    """Derive lower-bounds to optimal stopping problem with N1 exercise dates on [0,T],
    applying Longstaff-Schwartz with signatures described in Section 3.2
    Input:

    M = Number of samples used in first simulation, along which we solve the LP
    M2 = Number of samples used for resimulation (typically M2 >> M) 
    N1 = Number of exercise points in [0,T] in the sense of Bermudda-options
    N = Number of discretization points for the grid [0,T] (typically N>>N1)
    T = Final time for the stopping problem
    K = Depth of linear (time-extended) signature
    h = Hurst parameter
    
    Output: 
    y0 = true lower bound for the optimal stopping problem
    regr = coefficients from regression with signatures
    run_time= time (in seconds) needed for computation
    
    """
    s = time.time()
    #D = Number of entries of signature of 2-dim path (t,X_t)
    D = int((1-(2)**(K+1))/(-1) -1)
    
    #define index set of exercise dates
    
    subindex = [int((j+1)*N/N1) for j in range(N1)]
    subindex2 = [int((j)*N/N1) for j in range(N1+1)]
    
    
    #generate samples of FBM paths and increments, and corresponding Brownian incremements
    tt = np.linspace(0,T,N+1)
    F,dfBm,dW = FBM(N,M,h,method='cholesky').fbm()
    F = F.transpose()
    dW = dW.transpose()
    dfBm = dfBm.transpose()
    dfBm = dfBm.reshape(M,N,1)

    #compute linear signature using ii-package
    S= SignatureFull(tt, dfBm.reshape(M,N,1), K)
    
    #define fBm and signature at exercise dates, without time 0
    F_exercise = F[:,subindex]
    S_exercise= S[:,subindex,:]
    
    #regr will contain the regression coefficients of Longstaff-Schwarz at all exercise dates, see Section 3.2 for details
    regr = [0]*(N1-1)
    
    #Final value of optimal stopping problem is value of fBm
    value = F_exercise[:,-1]
    
    #Perform regression at each exercise date
    for k in reversed(range(1,N1)):
        #Linear regression with signatures, approximating continuation value
        regr[k-1] = LinearRegression().fit(S_exercise[:,k-1,:],value)
        reg = regr[k-1].predict(S_exercise[:,k-1,:])
        print(regr[k-1].score(S_exercise[:,k-1,:],value))
        #update value at for each sample, using Longstaff-Schwarz recursion
        for m in range(M):
            if reg[m]> F_exercise[m,k-1]:
                
                continue
            else:
                value[m] = F_exercise[m,k-1]

    
    del F_exercise,S_exercise,dW,dfBm,F
    #Resimulation with fresh samples, using computed regression coefficients to get value at 0
    
    
    F,dfBm,dW = FBM(N,M2,h,method='cholesky').fbm()
    F = F.transpose()
    dW = dW.transpose()
    dfBm = dfBm.transpose()
    
    #Compute signature again, and values of fBm and signature at exercise dates
    S= SignatureFull(tt, dfBm.reshape(M2,N,1), K)
    F_exercise= F[:,subindex]
    S_exercise= S[:,subindex,:]
  
    value = F_exercise[:,-1]
    
    #compute continuation with the regression coefficients for all exercise dates
    reg = np.zeros((M2,N1))
    for j in range(N1-1):
        reg[:,j] = regr[j].predict(S_exercise[:,j,:])
    
    #for each sample, we use the stopping rule to store the value at the optimal time
    for m in range(M2):
        i = 0
        while reg[m,i]> F_exercise[m,i]:
            i = i+1
            if i == N1-1:
                break
        value[m] = F_exercise[m,i]

    #value corresponds to the value at the first exercise date, taking the mean yields the price
    y0 = np.mean(value)
    STD = np.std(value)/np.sqrt(M2)
    ss = time.time()
    run_time = ss-s
    return y0,regr,run_time