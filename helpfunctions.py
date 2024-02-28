#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 27 12:01:17 2024


Collection of helpfunctions needed for optimal stopping code for fractional Brownian motion
@author: lucapelizzari
"""

import numpy as np
import iisignature as ii
import time as time
import gurobipy as gu
from gurobipy import GRB
#functions compupting the full linear signature, based on ii-signature package https://pypi.org/project/iisignature/


def _prepare_sig(tGrid, dx, deg):
    """Auxiliary function for computing signatures. See help for signature."""
    N = len(tGrid) - 1
    if len(dx.shape) == 1:
        # assume that M = d = 1
        dx.reshape((1, dx.shape[0],1))
    if len(dx.shape) == 2:
        # assume that either d = 1 or M = 1
        if dx.shape[0] == N:
            dx.reshape((1, dx.shape[0], dx.shape[1]))
        elif dx.shape[1] == N:
            dx.reshape((dx.shape[0], dx.shape[1], 1))
    assert len(dx.shape) == 3 and dx.shape[1] == N, \
        f"dx is misshaped as {dx.shape}"
    M,_,d = dx.shape
    
    x = np.zeros((M,N+1,d))
    x[:,1:(N+1),:] = np.cumsum(dx, axis=1)
    tArr = np.repeat(tGrid,M).reshape((M,N+1,1), order='F')
    z = np.concatenate((tArr, x), axis=2)
    
    return M, d, z

def SignatureFull(tGrid, dx, deg):
    """
    Compute the signature of a path (t,x) up to degree deg.

    Parameters
    ----------
    tGrid : numpy array
        Time grid of t, size N+1.
    dx : numpy array
        Increments of the path x, an array of dimension MxNxd.
    deg : integer
        Degree of the signature to compute.

    Returns
    -------
    sig : numpy array
        The signature of (t,x) at all the times, an array of size (M,k).

    """
    M, d, z = _prepare_sig(tGrid, dx, deg)
    
    # We need to compute the signature of z
    k = ii.siglength(d+1, deg)
    N = len(tGrid)
    sig = np.zeros((M,N,k+1))
    for m in range(M):
        sig[m,1:N,1:k+1] = ii.sig(z[m,:,:], deg,2)
    sig[:,:,0] = 1

    return sig

#Similarly we consider the same function for signature lifts of (t,X,[X])

def _prepare_sigQV(tGrid, dx,QV, deg):
    """Auxiliary function for computing signatures. See help for signature."""
    N = len(tGrid) - 1
    if len(dx.shape) == 1:
        # assume that M = d = 1
        dx.reshape((1, dx.shape[0],1))
    if len(dx.shape) == 2:
        # assume that either d = 1 or M = 1
        if dx.shape[0] == N:
            dx.reshape((1, dx.shape[0], dx.shape[1]))
        elif dx.shape[1] == N:
            dx.reshape((dx.shape[0], dx.shape[1], 1))
    assert len(dx.shape) == 3 and dx.shape[1] == N, \
        f"dx is misshaped as {dx.shape}"
    M,_,d = dx.shape
    QV = QV.reshape(M,N+1,1)
    x = np.zeros((M,N+1,d))
    x[:,1:(N+1),:] = np.cumsum(dx, axis=1)
    z = np.concatenate((QV, x), axis=2)
    return M, d, z

def signatureQV(tGrid, dx,QV, deg):
    """
    Compute the signature of a path (t,x,[x]) up to degree deg.

    Parameters
    ----------
    tGrid : numpy array
        Time grid of t, size N+1.
    dx : numpy array
        Increments of the path x, an array of dimension MxNxd.
    deg : integer
        Degree of the signature to compute.

    Returns
    -------
    sig : numpy array
        The signature of (t,x) at all the times, an array of size (M,k).

    """
    M, d, z = _prepare_sigQV(tGrid, dx,QV, deg)
    
    # We need to compute the signature of z
    k = ii.siglength(d+1, deg)
    N = len(tGrid)
    sig = np.zeros((M,N-1,k))
    for m in range(M):
        sig[m,:] = ii.sig(z[m,:,:], deg,2)

    return sig

#Next function solves the linear programm described in Remark 3.9, using Gurobi optimization https://www.gurobi.com


def LP_solver(F,tt,N1,N,D,M,Y,subindex):
    """Solve the LP described in Remark 3.9
    F = Samples of fBm, array of (M,N+1)
    tt = discretized interval [0,T]
    N1 = exercise dates for optimal stopping
    N = discretization times (typically N>>N1)
    D = size of signature
    M = sample size
    Y = samples of martingale bases on exercise dates, array of (M,N1+1,D)
    subindex = indices of exercise dates nested in discretization
    
    Output: Solution to LP
    """
    #L denotes the number of basis functions martingales we have
    L = int(len(Y[0,0,:]))
    s = time.time()
    #start minimization procedure with Gurobi, see https://www.gurobi.com for details
    m = gu.Model()
    
    #define objectiv c^Tx for LP
    c = np.zeros(M+L)
    c[0:M] = 1/M
    #Define matrices and vectors for the LP, see Remark 3.9
    B = np.zeros(M*(N1+1))
    A = np.zeros(shape = (M*(N1+1),L+M))
    for l in range(M):
        B[l*(N1+1):(l+1)*(N1+1)] =F[l,subindex]
        A[l*(N1+1):(l+1)*(N1+1),l] = 1
        A[l*(N1+1):(l+1)*(N1+1),M:M+L] = Y[l,:,:]
    x = m.addMVar(shape = M+L,name = "x")
    m.setObjective(c @ x, GRB.MINIMIZE)
    m.addConstr(A @ x >= B, name = "c")
    m.Params.LogToConsole = 0
    m.optimize()
    xx = x.X
    ss = time.time()
    print(ss-s)
    return xx[M:M+L]