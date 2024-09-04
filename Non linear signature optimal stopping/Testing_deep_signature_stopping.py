#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 29 11:16:32 2024

@author: lucapelizzari
"""
import numpy as np
from Signature_computer import SignatureComputer
from Deep_signatures_optimal_stopping import DeepLongstaffSchwartzPricer, DeepDualPricer
from rBergomi_simulation import SimulationofrBergomi

def main():
    # Set up parameters
    M = 2**17  # number of samples
    M2 = 2**17
    M_val_dual = int(M * 0.85)
    T = 1  # final time
    N = 240 # number of time-steps
    N1 = 12
    h = 0.5  # Hurst parameter
    L = 2**5
    K = 1
    ridge = 10**(-5)
    phi = lambda x: np.maximum(1 - x, 0) 
    xi = 0.09
    eta = 1.9
    r = 0.05
    rho = -0.9
    X0 = 1
    Batch_Normalization = False
    Dropout = False
    attention_layer = False
    layer_normalization = False
    regularizer = 0
    layers_primal = 5
    layers_dual = 5
    nodes = 10
    batch = 2**8
    poly_degree = 1
    poly_degree_dual = 4
    K_dual = 4
    epo = 15
    rate_primal = 0.0001
    rate_dual = 0.001
    tt = np.linspace(0,T,N+1)
    activation_function_primal = "tanh"
    activation_function_dual = "relu"
    signature_spec = "linear"
    signature_spec_dual = "log"

    # Generate training and testing data
    S_training, V_training, Payoff_training, dW_training = generate_training_data(M, N, T, phi, rho, K, X0, h, xi, eta, r)
    S_testing, V_testing, Payoff_testing, dW_testing = generate_testing_data(M2, N, T, phi, rho, K, X0, h, xi, eta, r)
    vol_testing = np.sqrt(V_testing)
    vol_training = np.sqrt(V_training)
    A_training = np.zeros((M,N+1))
    A_training[:,1:N+1] = tt[1:N+1]
    A_testing = np.zeros((M,N+1))
    A_testing[:,1:N+1] = tt[1:N+1]
    # Initialize SignatureComputer
    sig_computer = SignatureComputer(T, N, K, signature_spec, signature_lift="polynomial-vol", poly_degree=poly_degree)

    # Compute signatures
    signatures_training = sig_computer.compute_signature(S_training, vol_training, A_training, Payoff_training)
    signatures_testing = sig_computer.compute_signature(S_testing, vol_testing, A_testing, Payoff_testing)
    # Longstaff-Schwartz Pricing
    ls_pricer = DeepLongstaffSchwartzPricer(
        N1=N1,
        T=T,
        r=r,
        mode="American Option",
        layers=layers_primal,
        nodes=nodes,
        activation_function=activation_function_primal,
        batch_normalization=Batch_Normalization,
        regularizer=regularizer,
        dropout=Dropout,
        layer_normalization=layer_normalization
    )

    lower_bound, lower_bound_std, ls_regression_models = ls_pricer.price(
        signatures_training,
        Payoff_training,
        signatures_testing,
        Payoff_testing,
        M_val=0,
        batch=batch,
        epochs=epo,
        learning_rate=rate_primal
    )

    
    print(f"Longstaff-Schwartz lower bound: {lower_bound} ± {lower_bound_std/np.sqrt(M2)}")

    # Dual Pricing
    dual_pricer = DeepDualPricer(
        N1=N1,
        N=N,
        T=T,
        r=r,
        layers=layers_dual,
        nodes=nodes,
        activation_function=activation_function_dual,
        batch_normalization=Batch_Normalization,
        regularizer=regularizer,
        dropout=Dropout,
        attention_layer=attention_layer,
        layer_normalization=layer_normalization
    )

    y0, upper_bound, upper_bound_std, dual_model, dual_rule_model = dual_pricer.price(
        signatures_training,
        np.exp(-r*tt)*Payoff_training,
        dW_training,
        signatures_testing,
        np.exp(-r*tt)*Payoff_testing,
        dW_testing,
        M_val=M_val_dual,
        batch=batch,
        epochs=epo,
        learning_rate=rate_dual
    )

    print(f"Dual estimated option value (y0): {y0}")
    print(f"Dual upper bound: {upper_bound} ± {upper_bound_std/np.sqrt(M2)}")
    gap = upper_bound - lower_bound
    gap_std = np.sqrt(upper_bound_std**2 + lower_bound_std**2)
    print(f"Relative Gap between upper and lower bounds: {100*gap/upper_bound}% ± {gap_std/np.sqrt(M2)}")

def generate_training_data(M, N, T, phi, rho, K, X0, H, xi, eta, r):
    X, V, I, dI, dW1, dW2, dB, Y = SimulationofrBergomi(M, N, T, phi, rho, K, X0, H, xi, eta, r)
    
    # Calculate Payoff
    Payoff = phi(X)
    
    return X, V, Payoff, dB

def generate_testing_data(M, N, T, phi, rho, K, X0, H, xi, eta, r):
    X, V, I, dI, dW1, dW2, dB, Y = SimulationofrBergomi(M, N, T, phi, rho, K, X0, H, xi, eta, r)
    
    # Calculate Payoff
    Payoff = phi(X)
    
    return X,V,Payoff,dB

if __name__ == "__main__":
    main()