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
    M = 2**17 # number of samples
    M2 = 2**17
    M_val_dual = int(M * 0.80)
    T = 1  # final time
    N = 600 # number of time-steps
    N1 = 12
    h = 0.07 # Hurst parameter
    K = 4
    strike = 1
    phi = lambda x: np.maximum(strike - np.exp(x), 0) 

    
    xi = 0.09
    eta = 1.9
    r = 0.05
    rho = -0.9
    #rho = 0
    #rho = -1
    X0 = 1
    Batch_Normalization = False
    Dropout = False
    attention_layer = False
    layer_normalization = False
    regularizer_primal = 0
    regularizer_dual = 10**(-8)
    layers_primal = 2
    layers_dual = 3
    nodes_primal = 32
    nodes_dual = 32
    batch_primal = 2**7
    batch_dual = 2**7
    poly_degree = 1
    epo = 20
    rate_primal = 10**(-4)
    rate_dual = 10**(-4)
    tt = np.linspace(0,T,N+1)
    activation_function_primal = "LeakyRelu"
    activation_function_dual = "relu"
    signature_spec = "linear"

    # Generate training and testing data
    S_training, V_training, Payoff_training, dW_training,I_training, M_training = generate_training_data(M, N, T, phi, rho, K, X0, h, xi, eta, r)
    S_testing, V_testing, Payoff_testing, dW_testing, I_testing, M_testing = generate_testing_data(M2, N, T, phi, rho, K, X0, h, xi, eta, r)

    vol_testing = np.sqrt(V_testing)
    vol_training = np.sqrt(V_training)
    #alternative: A_t = <X>_t for primal favorable..
    A_training = np.zeros((M,N+1))
    A_training[:,1:N+1] = tt[1:N+1]
    A_testing = np.zeros((M,N+1))
    A_testing[:,1:N+1] = tt[1:N+1]
    sub_N = N
    subindex_sub = [int(j*N/sub_N) for j in range(0,sub_N+1)]
    sig_computer = SignatureComputer(T, sub_N, K, signature_spec, signature_lift="logprice-payoff-vol-sig", poly_degree=poly_degree)

    signatures_training = sig_computer.compute_signature(S_training[:,subindex_sub], vol_training[:,subindex_sub], A_training[:,subindex_sub], Payoff_training[:,subindex_sub],dW_training[:,:,0],I_training[:,subindex_sub],M_training[:,subindex_sub])
    signatures_testing = sig_computer.compute_signature(S_testing[:,subindex_sub], vol_testing[:,subindex_sub], A_testing[:,subindex_sub], Payoff_testing[:,subindex_sub],dW_testing[:,:,0],I_testing[:,subindex_sub],M_testing[:,subindex_sub])
    

    print('European price',np.mean(Payoff_testing[:,-1]*np.exp(-r)))
    
    #print(f"Deep lower and bound bounds with Hurst: {h}, truncation {K}, rho {rho}, N {N}")
    ls_pricer = DeepLongstaffSchwartzPricer(
        N1=N1,
        T=T,
        r=r,
        mode="American Option",
        layers=layers_primal,
        nodes=nodes_primal,
        activation_function=activation_function_primal,
        batch_normalization=Batch_Normalization,
        regularizer=regularizer_primal,
        dropout=Dropout,
        layer_normalization=layer_normalization
    )

    lower_bound, lower_bound_std, ls_regression_models,point_estimate = ls_pricer.price(
        signatures_training,
        Payoff_training[:,subindex_sub],
        signatures_testing,
        Payoff_testing[:,subindex_sub],
        M_val=0,
        batch=batch_primal,
        epochs=epo,
        learning_rate=rate_primal
    )
    print(f"Longstaff-Schwartz lower bound: {lower_bound} ± {lower_bound_std/np.sqrt(M2)}")
    print(sub_N,'discretization yields estimate:',point_estimate,'and lowerbound',lower_bound)
    dual_pricer = DeepDualPricer(
        N1=N1,
        N=N,
        T=T,
        r=r,
        layers=layers_dual,
        nodes=nodes_dual,
        activation_function=activation_function_dual,
        batch_normalization=Batch_Normalization,
        regularizer=regularizer_dual,
        dropout=Dropout,
        attention_layer=attention_layer,
        layer_normalization=layer_normalization,
        mode_dim = "2-dim"
    )

    y0, upper_bound, upper_bound_std, dual_model, dual_rule_model = dual_pricer.price(
        signatures_training,
        np.exp(-r*tt)*Payoff_training,
        dW_training[:,:,:],
        signatures_testing,
        np.exp(-r*tt)*Payoff_testing,
        dW_testing[:,:,:],
        M_val=M_val_dual,
        batch=batch_dual,
        epochs=epo,
        learning_rate=rate_dual
    )
    print(f"Dual estimated option value (y0): {y0}")
    print(f"Dual upper bound: {upper_bound} ± {upper_bound_std/np.sqrt(M2)}")
    #gap = upper_bound - lower_bound
    #gap_std = np.sqrt(upper_bound_std**2 + lower_bound_std**2)
    #print(f"Relative Gap between upper and lower bounds: {100*gap/upper_bound}% ± {gap_std/np.sqrt(M2)}")

def generate_training_data(M, N, T, phi, rho, K, X0, H, xi, eta, r):
    X, V, I, dI, dW1, dW2, dB, Y = SimulationofrBergomi(M, N, T, phi, rho, K, X0, H, xi, eta, r)
    
    # Calculate Payoff
    X = np.log(X)
    #YYY = np.zeros((X.shape[0],X.shape[1]))
    #YYY[:,1:N+1] = np.cumsum(Y[:,0:N]*(1/(N+1)),axis=1)
    #X = YYY
    #Payoff = YYY
    Payoff = phi(X)
    I = np.zeros((X.shape[0],X.shape[1]))
    MM = np.zeros((X.shape[0],X.shape[1]))
    MM[:,1:] = np.cumsum(dW2[:,:],axis=1)
    #I = np.zeros((X.shape[0],X.shape[1]))
    #I[:,1:] = np.cumsum(dW1[:,:,0],axis=1)
    #I[:,1:] = np.cumsum(np.sqrt(V[:,0:N])*dW2[:,:],axis=1)
    dZ = np.zeros((dW1.shape))
    dZ[:,:,0] = dW1[:,:,0]
    dZ[:,:,1] = dW2[:,:]
    dB = dZ

    #MM= dW2
    return X, V, Payoff, dB,I,MM

def generate_testing_data(M, N, T, phi, rho, K, X0, H, xi, eta, r):
    X, V, I, dI, dW1, dW2, dB, Y = SimulationofrBergomi(M, N, T, phi, rho, K, X0, H, xi, eta, r)
    
    #I = np.zeros((X.shape[0],X.shape[1]))
    #I[:,1:] = np.cumsum(dW1[:,:,0],axis=1)
    #I = np.zeros((X.shape[0],X.shape[1]))
    #YYY = np.zeros((X.shape[0],X.shape[1]))
    #YYY[:,1:N+1] = np.cumsum(Y[:,0:N]*(1/(N+1)),axis=1)
    #X = YYY
    #Payoff = YYY
    I[:,1:] = np.cumsum(dW1[:,:,0],axis=1)
    MM = np.zeros((X.shape[0],X.shape[1]))
    MM[:,1:] = np.cumsum(dW2[:,:],axis=1)
    #I[:,1:] = np.cumsum(np.sqrt(V[:,0:N])*dW2[:,:],axis=1)
    # Calculate Payoff
    X = np.log(X)

    Payoff = phi(X)
    dZ = np.zeros((dW1.shape))
    dZ[:,:,0] = dW1[:,:,0]
    dZ[:,:,1] = dW2[:,:]
    #MM = dW2
    dB = dZ
    return X,V,Payoff,dB,I,MM

if __name__ == "__main__":
    main()