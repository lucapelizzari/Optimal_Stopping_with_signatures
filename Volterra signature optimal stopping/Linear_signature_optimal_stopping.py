#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 29 14:54:16 2024

@author: lucapelizzari
"""
import numpy as np
import scipy as sc
from sklearn.linear_model import Ridge
import time
import cvxpy as cp
import gurobipy as gu
from gurobipy import GRB

class LinearLongstaffSchwartzPricer:
    """
    Computes the lower bound of optimal stopping problem using linear regression based on the signature of the augmented path.
    """
    def __init__(self, N1, T, r, mode="Standard", ridge = 10**(-9)):
        """
        Parameters
        ----------
        N1 : int
            Number of exercise dates for optimal stopping
        T : float
            Time horizon for the option
        r : float
            Risk-free interest rate
        mode : str
            "Standard" or "American Option", where "American Option" only uses in the money paths for regression
        ridge : float
            Ridge parameter for regularization
        """
        self.N1 = N1
        self.T = T
        self.r = r
        self.mode = mode
        self.ridge = ridge

    def price(self, S_training_sig, Payoff_training, S_testing_sig, Payoff_testing):
        """
        Computes the lower bound for the price of a path-dependent option using linear regression based on the signature of the augmented path.

        Parameters
        ----------
        S_training_sig : numpy array
            Signature (+ possibly polynomial features) of the augmented path for the training set
        Payoff_training : numpy array
            Payoff for training paths
        S_testing_sig : numpy array
            Signature (+ possibly polynomial features) of the augmented path for the testing set
        Payoff_testing : numpy array
            Payoff for testing paths
        
        Returns
        -------
        lower_bound : float
            Lower bound for the price of the option
        lower_bound_std : float
            Standard deviation of the lower bound
        regr : list
            List of Ridge regressors for each exercise date
        """
        M, N, feature_dim = S_training_sig.shape
        N= N-1
        M2, _, _ = S_testing_sig.shape
        subindex = [int((j+1)*N/self.N1) for j in range(self.N1)]
        subindex2 = [int((j)*N/self.N1) for j in range(self.N1+1)]
        ttt = np.linspace(0, self.T, self.N1 + 1)
        
        Payoff_exercise_training = Payoff_training[:, subindex]
        S_exercise_training_sig = S_training_sig[:, subindex, :]  # Adjust index for signatures
        
        Payoff_exercise_testing = Payoff_testing[:, subindex]
        S_exercise_testing_sig = S_testing_sig[:, subindex, :]  # Adjust index for signatures
        
        regr = [None] * (self.N1 - 1)
        value_training = Payoff_exercise_training[:, -1]
        
        dtt = np.exp(-self.r * self.T / (self.N1 + 1))
        if self.mode == "Standard":
            """
            Standard mode: use all paths for regression
            """
            for k in reversed(range(1,self.N1)):
                value_training = value_training*dtt
                
                regr[k-1] = Ridge(alpha=self.ridge).fit(S_exercise_training_sig[:,k-1,:],value_training)
                continuatuion_value_estimate = regr[k-1].predict(S_exercise_training_sig[:,k-1,:])
                print("Regression score at exercise date",k,regr[k-1].score(S_exercise_training_sig[:,k-1,:],value_training))
                
                #update value at for each sample, using Longstaff-Schwarz recursion
                for m in range(M):
                    if continuatuion_value_estimate[m] <= Payoff_exercise_training[m,k-1]:
                        value_training[m] = Payoff_exercise_training[m,k-1]
     
        
        elif self.mode == "American Option":
            """
            American Option mode: use only in the money paths for regression
            """
            for j in reversed(range(1,self.N1)):
                value_training = value_training * dtt
                ITM = [m for m in range(M) if Payoff_exercise_training[m, j-1] > 0]
                
                if len(ITM) <= 1:
                    continue
                else:
                    regr[j-1] = Ridge(alpha=self.ridge).fit(S_exercise_training_sig[ITM,j-1,:],value_training[ITM])
                    continuatuion_value_estimate = regr[j-1].predict(S_exercise_training_sig[ITM,j-1,:])
                    print("Regression score at exercise date",j,regr[j-1].score(S_exercise_training_sig[ITM,j-1,:],value_training[ITM]))
                    for m in range(len(ITM)):
                        """
                        Update the value at for each path, using Longstaff-Schwarz recursion
                        """
                        if continuatuion_value_estimate[m] <= Payoff_exercise_training[ITM[m],j-1]:
                            value_training[ITM[m]] = Payoff_exercise_training[ITM[m],j-1]
        
        else:
            raise ValueError(f"Invalid mode: {self.mode}")
        
        # Compute true lower bound for testing data
        value_testing = Payoff_exercise_testing[:, -1]
        reg = np.zeros((M2, self.N1))
        
        for j in range(self.N1 - 1):
            """
            Compute the continuation value for each path at each exercise date
            """
            if regr[j] is None:
                reg[:, j] = 10**8
            else:
                reg[:,j] = regr[j].predict(S_exercise_testing_sig[:,j,:])
        
        if self.mode == "Standard":
            """
            Standard mode: use all paths for regression
            """
            for m in range(M2):
                i = 0
                while reg[m,i]> Payoff_exercise_testing[m,i] and i < self.N1-1:
                    i = i+1
                value_testing[m] = Payoff_exercise_testing[m,i]*np.exp(-self.r*self.T*(ttt[i+1]-ttt[1]))

        
        elif self.mode == "American Option":
            """
            American Option mode: use only in the money paths for regression
            """
            for m in range(M2):
                i = 0
                while i < self.N1 - 1 and (Payoff_exercise_testing[m, i] == 0 or reg[m, i] > Payoff_exercise_testing[m, i]):
                    i += 1
                value_testing[m] = Payoff_exercise_testing[m, i] * np.exp(-self.r * self.T * (ttt[i+1] - ttt[1]))
        
        lower_bound = np.mean(value_testing)
        lower_bound_std = np.std(value_testing)
        
        return lower_bound, lower_bound_std, regr


class LinearDualPricer:
    """
    Computes the upper bound of optimal stopping problem using linear programming based on the martingale representation of the option.

    

    """
    def __init__(self, N1, N, T, r, LP_solver="Gurobi"):
        """
        Parameters
        ----------
        N1 : int
            Number of exercise dates for optimal stopping
        N : int
            Number of discretization times (typically N>>N1)
        T : float
            Time horizon for the option
        r : float
            Risk-free interest rate
        LP_solver : str
            "Gurobi" or "CVXPY", solver for the linear program
        """
        self.N1 = N1
        self.N = N
        self.T = T
        self.r = r
        self.LP_solver = LP_solver

    def price(self, S_training_sig, Payoff_training, dW_training, S_testing_sig, Payoff_testing, dW_testing):
        """
        Computes the upper bound for the price of a path-dependent option using signature martingale representation.

        Parameters
        ----------
        S_training_sig : numpy array
            Signature (+ possibly polynomial features) of the augmented path for the training set
        Payoff_training : numpy array
            Payoff for training paths
        dW_training : numpy array
            Increments of the Brownian motion for the training paths
        S_testing_sig : numpy array
            Signature (+ possibly polynomial features) of the augmented path for the testing set
        Payoff_testing : numpy array
            Payoff for testing paths
        dW_testing : numpy array
            Increments of the Brownian motion for the testing paths
        
        Returns
        -------
        upper_bound : float
            Upper bound for the price of the option
        upper_bound_std : float
            Standard deviation of the upper bound
        MG_testing : numpy array
            Doob martingale approximation
        """
        M, _, D = S_training_sig.shape
        M2, _, _ = S_testing_sig.shape
        subindex = [int((j+1)*self.N/self.N1) for j in range(self.N1)]
        subindex2 = [int((j)*self.N/self.N1) for j in range(self.N1+1)]
        MG_training = np.zeros((M,self.N+1,D))
        for dd in range(D):
            MG_training[:,1:self.N+1,dd] = np.cumsum(S_training_sig[:,0:self.N,dd]*dW_training,axis=1)
        if self.LP_solver == "Gurobi":
            solver = LPSolver(F=Payoff_training, tt=np.linspace(0, self.T, self.N1 + 1), N1=self.N1, N=self.N, D=D, M=M, Y=MG_training[:,subindex2,:], subindex=subindex2)
            res = solver.solve_gurobi()
        elif self.LP_solver == "CVXPY":
            solver = LPSolver(F=Payoff_training, tt=np.linspace(0, self.T, self.N1 + 1), N1=self.N1, N=self.N, D=D, M=M, Y=MG_training[:,subindex2,:], subindex=subindex2)
            res = solver.solve_cvxpy()
        else:
            raise ValueError(f"Invalid LP solver: {self.LP_solver}")
        # Testing for fresh samples
        MG_testing = np.zeros((M2,self.N+1,D))
        for dd in range(D):
            MG_testing[:,1:self.N+1,dd] = np.cumsum(S_testing_sig[:,0:self.N,dd]*dW_testing,axis=1)

        # Compute the upper bound and standard deviation
        
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
