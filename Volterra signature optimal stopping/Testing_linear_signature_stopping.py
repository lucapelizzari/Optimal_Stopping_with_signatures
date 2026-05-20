#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 29 15:28:18 2024

@author: lucapelizzari
"""

import numpy as np
from Signature_computer import SignatureComputer
from Linear_signature_optimal_stopping import LinearLongstaffSchwartzPricer, LinearDualPricer
from rBergomi_simulation import SimulationofrBergomi
import time as time


def main():
    # Set up parameters
    M = 2**18  # number of samples
    M2 = 2**18
    M_dual = 10**3
    M2_dual = 10**3
    M_val_dual = int(M * 0.85)
    T = 1  # final time
    N = 600  # number of time-steps
    N1 = 12
    h = 0.07  # Hurst parameter
    K = 4
    ridge = 10**(-5)
    sigma = 0.5
    penalty = 1
    static_kernel_spec = 'Linear'
    phi = lambda x: np.maximum(1 - np.exp(x), 0) 
    xi = 0.09
    eta = 1.9
    r = 0.05
    rho = -0.9
    X0 = 1
    poly_degree = 1
    poly_degree_dual = 4
    K_dual = 4
    tt = np.linspace(0, T, N+1)
    signature_spec = "linear"
    signature_spec_dual = "linear"


    # Generate training and testing data
    S_training, V_training, Payoff_training, dW_training = generate_data(M, N, T, phi, rho, K, X0, h, xi, eta, r)
    S_testing, V_testing, Payoff_testing, dW_testing = generate_data(M2, N, T, phi, rho, K, X0, h, xi, eta, r)
    vol_testing = np.sqrt(V_testing)
    vol_training = np.sqrt(V_training)
    A_training = np.zeros((M, N+1))
    #A_training[:, 1:N+1] = tt[1:N+1]
    A_training[:, 1:N+1] = np.cumsum(V_training[:,0:N]/(N+1),axis=1)
    A_testing = np.zeros((M, N+1))
    A_testing[:, 1:N+1] = np.cumsum(V_testing[:,0:N]/(N+1),axis=1)
    #A_testing[:, 1:N+1] = tt[1:N+1]

    # Initialize SignatureComputer
    sub_N = N
    subindex_sub = [int(j*N/sub_N) for j in range(0,sub_N+1)]
    sig_computer = SignatureComputer(T, sub_N, K, signature_spec, signature_lift="normal", poly_degree=4)

    # Compute signatures
    s = time.time()
    signatures_training = sig_computer.compute_signature(S_training[:,subindex_sub], vol_training[:,subindex_sub], A_training[:,subindex_sub], Payoff_training[:,subindex_sub],Payoff_training[:,subindex_sub],Payoff_training[:,subindex_sub],Payoff_training[:,subindex_sub])
    print('time for sig linear',time.time()-s)
    signatures_testing = sig_computer.compute_signature(S_testing[:,subindex_sub], vol_testing[:,subindex_sub], A_testing[:,subindex_sub], Payoff_testing[:,subindex_sub],Payoff_training[:,subindex_sub],Payoff_training[:,subindex_sub],Payoff_training[:,subindex_sub])

    # Linear Longstaff-Schwartz Pricing


    ls_pricer = LinearLongstaffSchwartzPricer(
        N1=N1,
        T=T,
        r=r,
        mode = "American Option",
        ridge = 10**(-5)
    )

    lower_bound, lower_bound_std, regr, point_estimate = ls_pricer.price(
        signatures_training,
        Payoff_training,
        signatures_testing,
        Payoff_testing
    )
    print(f"Linear Longstaff-Schwartz lower bound: {lower_bound} ± {lower_bound_std/np.sqrt(M2)}, point-estimate {point_estimate}")
    
    dual_pricer = LinearDualPricer(
        N1=N1,
        N=N,
        T=T,
        r = r,
        LP_solver = "CVXPY"
    )

  
    ss = time.time()
    upper_bound, upper_bound_std, MG_testing = dual_pricer.price(
        signatures_training[0:M_dual,:,:],
        Payoff_training[0:M_dual,subindex_sub]*np.exp(-r*tt),
        dW_training[0:M_dual,:],
        signatures_testing[0:M_dual,:,:],
        Payoff_testing[0:M_dual,subindex_sub]*np.exp(-r*tt),
        dW_testing[0:M_dual,:]
    )
    print('training linear',time.time()-ss)

    

    # Linear Dual Pricing

    

    

def generate_data(M, N, T, phi, rho, K, X0, H, xi, eta, r):
    X, V, I, dI, dW1, dW2, dB, Y = SimulationofrBergomi(M, N, T, phi, rho, K, X0, H, xi, eta, r)
    X = np.log(X)
    # Calculate Payoff
    Payoff = phi(X)
    
    return X, V, Payoff, dB


if __name__ == "__main__":
    main()
