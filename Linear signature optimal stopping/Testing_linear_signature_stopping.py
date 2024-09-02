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

def main():
    # Set up parameters
    M = 2**17  # number of samples
    M2 = 2**17
    M_dual = 10**4
    M2_dual = 10**5
    M_val_primal = 0
    M_val_dual = int(M * 0.85)
    T = 1  # final time
    N = 60  # number of time-steps
    N1 = 12
    h = 0.1  # Hurst parameter
    K = 4
    ridge = 10**(-5)
    sigma = 0.5
    penalty = 1
    static_kernel_spec = 'Linear'
    phi = lambda x: np.maximum(1 - x, 0) 
    xi = 0.09
    eta = 1.9
    r = 0.05
    rho = -0.9
    X0 = 1
    poly_degree = 5
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
    A_training[:, 1:N+1] = tt[1:N+1]
    A_testing = np.zeros((M2, N+1))
    A_testing[:, 1:N+1] = tt[1:N+1]

    # Initialize SignatureComputer
    sig_computer = SignatureComputer(T, N, K, signature_spec, signature_lift="payoff-and-polynomial-extended", poly_degree=poly_degree)

    # Compute signatures
    signatures_training = sig_computer.compute_signature(S_training, vol_training, A_training, Payoff_training)
    signatures_testing = sig_computer.compute_signature(S_testing, vol_testing, A_testing, Payoff_testing)

    # Linear Longstaff-Schwartz Pricing
    ls_pricer = LinearLongstaffSchwartzPricer(
        N1=N1,
        T=T,
        r=r,
        mode="American Option",
        ridge=ridge
    )

    lower_bound, lower_bound_std, ls_regression_models = ls_pricer.price(
        signatures_training,
        Payoff_training,
        signatures_testing,
        Payoff_testing
    )

    print(f"Linear Longstaff-Schwartz lower bound: {lower_bound} ± {lower_bound_std/np.sqrt(M2)}")

    # Linear Dual Pricing
    dual_pricer = LinearDualPricer(
        N1=N1,
        N=N,
        T=T,
        r=r,
        LP_solver="Gurobi"
    )

    upper_bound, upper_bound_std, MG = dual_pricer.price(
        signatures_training[:M_dual,:,:],
        Payoff_training[:M_dual,:],
        dW_training[:M_dual,:],
        signatures_testing[:M2_dual,:,:],
        Payoff_testing[:M2_dual,:],
        dW_testing[:M2_dual,:]
    )

    print(f"Linear Dual upper bound: {upper_bound} ± {upper_bound_std/np.sqrt(M2)}")
    gap = upper_bound - lower_bound
    gap_std = np.sqrt(upper_bound_std**2 + lower_bound_std**2)
    print(f"Gap between upper and lower bounds: {gap} ± {gap_std}")

def generate_data(M, N, T, phi, rho, K, X0, H, xi, eta, r):
    X, V, I, dI, dW1, dW2, dB, Y = SimulationofrBergomi(M, N, T, phi, rho, K, X0, H, xi, eta, r)
    
    # Calculate Payoff
    Payoff = phi(X)
    
    return X, V, Payoff, dB


if __name__ == "__main__":
    main()
