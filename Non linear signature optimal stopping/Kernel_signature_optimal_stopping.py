#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 30 16:05:21 2024

@author: lucapelizzari
"""

import numpy as np
import scipy.special as sc
from sklearn.kernel_ridge import KernelRidge
import torch
from sklearn.linear_model import Ridge
import time as time
import gurobipy as gu
from gurobipy import GRB
class KernelLongstaffSchwartzPricer:
    """
    Computes the lower bound of optimal stopping problem using kernel ridge regression with signature kernels.
    """
    def __init__(self, N1, T, r, L, mode):
        self.N1 = N1
        self.T = T
        self.r = r
        self.L = L
        self.mode = mode
    def price(self,Kernel_training,Payoff_training,A_training,Kernel_testing,Payoff_testing,A_testing,ridge):
        M, N = A_training.shape
        N = N - 1  # N includes the time-point 0
        M2 = A_testing.shape[0]
        subindex = [int((j+1)*N/self.N1) for j in range(self.N1)]
        subindex2 = [int((j)*N/self.N1) for j in range(self.N1+1)]
        tt = np.linspace(0, self.T, N+1)
        ttt = np.linspace(0, self.T, self.N1+1)
        #Get signature kernel matrices at exercise date
        dtt = np.exp(-self.r*self.T/(self.N1+1))
        value_training = Payoff_training[:,-1]
        Payoff_exercise_training = Payoff_training[:,subindex]
        Payoff_exercise_testing = Payoff_testing[:,subindex]
        

        #Learning the continuation functions using kernel ridge regression
        regr = [None]*self.N1
        if self.mode == "Standard":
            """
            Standard mode: use all paths for regression
            """
            for k in reversed(range(1, self.N1)):
                value_training = value_training * dtt
                regr[k-1] = Ridge(alpha=ridge).fit(Kernel_training[k-1][:,:,-1], value_training)
                continuatuion_value_estimate = regr[k-1].predict(Kernel_training[k-1][:,:,-1])
                print(f"Regression score at exercise date {k}: {regr[k-1].score(Kernel_training[k-1][:,:,-1], value_training)}")
                for m in range(M):
                    if continuatuion_value_estimate[m] <= Payoff_exercise_training[m,k-1]:
                        value_training[m] = Payoff_exercise_training[m,k-1]
                
        elif self.mode == "American Option":
            """
            American Option mode: use only in the money paths for regression
            """
            for j in reversed(range(1, self.N1)):
                value_training = value_training * dtt
                ITM = [m for m in range(M) if Payoff_exercise_training[m,j-1] > 0]
                
                if len(ITM) == 0:
                    continue
                
                regr[j-1] = Ridge(alpha=ridge).fit(Kernel_training[j-1][ITM,:,-1], value_training[ITM])

                
                continuatuion_value_estimate = regr[j-1].predict(Kernel_training[j-1][ITM,:,-1])
                print(f"Regression score at exercise date {j}: {regr[j-1].score(Kernel_training[j-1][ITM,:,-1], value_training[ITM])}")
                
                for m in range(len(ITM)):
                    if continuatuion_value_estimate[m] <= Payoff_exercise_training[ITM[m],j-1]:
                        value_training[ITM[m]] = Payoff_exercise_training[ITM[m],j-1]

        print(f'Biased estimator: {np.mean(value_training)}')

        # Testing phase
        value_testing = Payoff_exercise_testing[:,-1]
        reg = np.zeros((M2, self.N1))
        for j in range(self.N1-1):
            if regr[j] is not None:
                reg[:,j] = regr[j].predict(Kernel_testing[j][:,:,-1])

        if self.mode == "Standard":
            for m in range(M2):
                i = 0
                while i < self.N1-1 and reg[m,i] > Payoff_exercise_testing[m,i]:
                    i += 1
                value_testing[m] = Payoff_exercise_testing[m,i] * np.exp(-self.r*self.T*(ttt[i+1]-ttt[0]))
        elif self.mode == "American Option":
            
            for m in range(M2):
                i = 0
                while i < self.N1-1 and (Payoff_exercise_testing[m,i] == 0 or reg[m,i] > Payoff_exercise_testing[m,i]):
                    i += 1
                value_testing[m] = Payoff_exercise_testing[m,i] * np.exp(-self.r*self.T*(ttt[i+1]-ttt[1]))

        y0 = np.mean(value_testing)
        point_estimate = np.mean(value_training)
        STD = np.std(value_testing) / np.sqrt(M2)
        return y0, STD, regr, point_estimate
    
class KernelDualPricer:
    """
    Computes upper bounds of optimal stopping problem using signature kernels.
    """
    def __init__(self, N1, T, r, L,LP_solver):
        self.N1 = N1
        self.T = T
        self.r = r
        self.L = L
        self.LP_solver = LP_solver
    def price(self,Kernel_training,dW_training,Payoff_training,A_training,Kernel_testing,dW_testing,Payoff_testing,A_testing):
        M, N = A_training.shape
        N = N - 1  # N includes the time-point 0
        M2 = A_testing.shape[0]
        subindex = [int((j+1)*N/self.N1) for j in range(self.N1)]
        subindex2 = [int((j)*N/self.N1) for j in range(self.N1+1)]
        tt = np.linspace(0, self.T, N+1)
        ttt = np.linspace(0, self.T, self.N1+1)
        #Get signature kernel matrices at exercise date
        dtt = np.exp(-self.r*self.T/(self.N1+1))
        value_training = Payoff_training[:,-1]
        Payoff_exercise_training = Payoff_training[:,subindex]
        Payoff_exercise_testing = Payoff_testing[:,subindex]
        MG_training = np.zeros((M,N+1,self.L))
        MG_testing = np.zeros((M,N+1,self.L))
        for dd in range(self.L):
            MG_training[:,1:N+1,dd] = np.cumsum(Kernel_training[:,dd,0:N]*dW_training,axis=1)
            MG_testing[:,1:N+1,dd] = np.cumsum(Kernel_testing[:,dd,0:N]*dW_testing,axis=1)
        if self.LP_solver == "Gurobi":
            solver = LPSolver(F=Payoff_training, tt=np.linspace(0, self.T, self.N1 + 1), N1=self.N1, N=N, D=self.L, M=M, Y=MG_training[:,subindex2,:], subindex=subindex2)
            res = solver.solve_gurobi()
        elif self.LP_solver == "CVXPY":
            solver = LPSolver(F=Payoff_training, tt=np.linspace(0, self.T, self.N1 + 1), N1=self.N1, N=N, D=self.L, M=M, Y=MG_training[:,subindex2,:], subindex=subindex2)
            res = solver.solve_cvxpy()
        elif self.LP_solver == "Adam":
            solver = AdamSolver(F=Payoff_training, tt=np.linspace(0, self.T, self.N1 + 1), N1=self.N1, N=N, D=self.L, M=M, Y=MG_training[:,subindex2,:], subindex=subindex2)
        else:
            raise ValueError(f"Invalid LP solver: {self.LP_solver}")
        
        upper_bound = np.mean(np.max(Payoff_testing[:,subindex]-np.dot(MG_testing,res)[:,subindex],axis=1))
        upper_bound_std = np.std(np.max(Payoff_testing[:, subindex] - np.dot(MG_testing,res)[:,subindex], axis=1))
        
        return upper_bound, upper_bound_std, MG_testing

class LPSolver:
    def __init__(self, F, tt, N1, N, D, M, Y, subindex):
        """
        Initialize the LPSolver with common parameters.
        
        Parameters:
        F : array, shape (M, N+1)
            Samples of Payoff
        tt : array
            Discretized interval [0,T]
        N1 : int
            Number of exercise dates for optimal stopping
        N : int
            Number of discretization times (typically N>>N1)
        D : int
            Size of signature
        M : int
            Sample size
        Y : array, shape (M, N1+1, D)
            Samples of martingale bases at exercise dates
        subindex : array
            Indices of exercise dates nested in discretization
        """
        self.F = F
        self.tt = tt
        self.N1 = N1
        self.N = N
        self.D = D
        self.M = M
        self.Y = Y
        self.subindex = subindex
        self.L = int(len(Y[0, 0, :]))  # Number of basis functions martingales
        
        # Precompute common matrices
        self.c = np.zeros(M + self.L)
        self.c[0:M] = 1/M
        self.B = np.zeros(M * (N1 + 1))
        self.A = np.zeros(shape=(M * (N1 + 1), self.L + M))
        for l in range(M):
            self.B[l*(N1+1):(l+1)*(N1+1)] = F[l, subindex]
            self.A[l*(N1+1):(l+1)*(N1+1), l] = 1
            self.A[l*(N1+1):(l+1)*(N1+1), M:M+self.L] = Y[l, :, :]

    def solve_cvxpy(self):

        """Solve the LP using CVXPY"""
        start_time = time.time()
        
        x = cp.Variable(self.M + self.L)
        objective = cp.Minimize(self.c @ x)
        constraints = [self.B <= self.A @ x]
        prob = cp.Problem(objective, constraints)
        result = prob.solve()
        xx = x.value
        
        end_time = time.time()
        print(f'{end_time - start_time:.2f} seconds needed to solve the linear program using CVXPY')
        return xx[self.M:self.M + self.L]

    def solve_gurobi(self):
        """Solve the LP using Gurobi"""
        start_time = time.time()
        
        m = gu.Model()
        x = m.addMVar(shape=self.M + self.L, lb=-10**3, name="x")
        m.setObjective(self.c @ x, GRB.MINIMIZE)
        m.addConstr(self.A @ x >= self.B, 'g')
        
        m.Params.LogToConsole = 1
        m.optimize()
        xx = x.X
        
        end_time = time.time()
        print(f'{end_time - start_time:.2f} seconds needed to solve the linear program using Gurobi')
        return xx[self.M:self.M + self.L]


