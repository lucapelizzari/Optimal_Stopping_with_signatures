#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 30 15:04:05 2024

@author: lucapelizzari
"""

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer
import tensorflow as tf
from Deep_signatures_optimal_stopping import DeepLongstaffSchwartzPricer
from Signature_computer import SignatureComputer
from rBergomi_simulation import SimulationofrBergomi

class LongstaffSchwartzWrapper(BaseEstimator, RegressorMixin):
    """
    Wrapper class for Longstaff-Schwartz pricer with hyperparameter optimization.
    """
    def __init__(self, N1, T, r, mode="Standard", layers=3, nodes=64, activation_function='relu',
                 batch_normalization=False, regularizer=0, dropout=False, layer_normalization=False,
                 batch_size=256, epochs=10, learning_rate=0.001):
        
        self.N1 = N1
        self.T = T
        self.r = r
        self.mode = mode
        self.layers = layers
        self.nodes = nodes
        self.activation_function = activation_function
        self.batch_normalization = batch_normalization
        self.regularizer = regularizer
        self.dropout = dropout
        self.layer_normalization = layer_normalization
        self.batch_size = batch_size
        self.epochs = epochs
        self.learning_rate = learning_rate

    def fit(self, X, y):
        tf.keras.backend.clear_session()
        # Global variables (you might want to pass these as parameters instead)
        global S_training_sig, Payoff_training, S_testing_sig, Payoff_testing, M_val

        pricer = DeepLongstaffSchwartzPricer(
            N1=self.N1,
            T=self.T,
            r=self.r,
            mode=self.mode,
            layers=self.layers,
            nodes=self.nodes,
            activation_function=self.activation_function,
            batch_normalization=self.batch_normalization,
            regularizer=self.regularizer,
            dropout=self.dropout,
            layer_normalization=self.layer_normalization
        )

        self.option_value, _, _ = pricer.price(
            S_training_sig, Payoff_training, S_testing_sig, Payoff_testing,
            batch=self.batch_size, epochs=self.epochs, learning_rate=self.learning_rate, M_val=M_val
        )
        return self

    def predict(self, X):
        return np.full(X.shape[0], self.option_value)

    def score(self, X, y=None):
        return self.option_value if self.option_value is not None else -np.inf

def custom_scorer(estimator, X, y=None):
    estimator.fit(X, y)
    return estimator.score(X)

# Main script
if __name__ == "__main__":
    # Set up parameters
    M = 2**17  # number of samples
    M2 = 2**17
    T = 1  # final time
    N = 120  # number of time-steps
    M_val = 0
    N1 = 12
    h = 0.15  # Hurst parameter
    K = 2
    sigma = 0.5
    xi = 0.09
    eta = 1.9
    r = 0.05
    rho = -0.9
    X0 = 1

    # Generate data
    tt=np.linspace(0,T,N+1)
    phi = lambda x: np.maximum(1 - x, 0)  # Example payoff function
    F_training, V_training, I_training, dI_training, dW1_training, dW2_training, dB_training, Y_training = SimulationofrBergomi(M, N, T, phi, rho, K, X0, h, xi, eta, r)
    F_testing, V_testing, I_testing, dI_testing, dW1_testing, dW2_testing, dB_testing, Y_testing = SimulationofrBergomi(M2, N, T, phi, rho, K, X0, h, xi, eta, r)
    vol_training = np.sqrt(V_training)
    vol_testing = np.sqrt(V_testing)
    Payoff_training = phi(F_training)
    Payoff_testing = phi(F_testing)
    A_training = np.zeros_like(F_training)
    A_testing = np.zeros_like(F_testing)
    A_training[:,1:] = tt[1:]
    A_testing[:,1:] = tt[1:]
    # Prepare data (you might need to adjust this part based on your specific requirements)
    poly_degree = 1
    sig_computer = SignatureComputer(T, N, K, "log", signature_lift="polynomial-vol", poly_degree=poly_degree)
    S_training_sig =sig_computer.compute_signature(F_training, vol_training, A_training, Payoff_training)
    S_testing_sig = sig_computer.compute_signature(F_testing, vol_testing, A_testing, Payoff_testing)
    

    # Set up Grid Search
    param_grid = {
        'layers': [4,5],
        'nodes': [10, 32],
        'activation_function': ['tanh','relu','LeakyRelu'],
        'batch_normalization': [False],
        'regularizer': [0,10**(-9)],
        'dropout': [False],
        'layer_normalization': [False],
        'mode': ['American Option'],
        'batch_size': [32,256,512],
        'epochs': [10],
        'learning_rate': [0.001,0.0001]
    }

    grid_search = GridSearchCV(
        LongstaffSchwartzWrapper(N1=N1, T=T, r=r),
        param_grid,
        cv=4,
        n_jobs=16,
        scoring=custom_scorer,
        refit=False
    )

    # Dummy data for sklearn compatibility
    X_dummy = np.zeros((100, 1))
    y_dummy = np.zeros(100)


    # Perform Grid Search
    grid_search.fit(X_dummy, y_dummy)

    # Print results
    print("Best parameters:", grid_search.best_params_)
    print("Best option value:", grid_search.best_score_)

    # Get all results
    results = grid_search.cv_results_

    # Create a list of dictionaries containing parameters and scores
    all_results = []
    for mean_score, params in zip(results['mean_test_score'], results['params']):
        result_dict = params.copy()
        result_dict['score'] = mean_score
        all_results.append(result_dict)

    # Sort results by score in descending order
    all_results_sorted = sorted(all_results, key=lambda x: x['score'], reverse=True)

    # Save results
    np.save('all_hyperparameter_results_4.npy', all_results_sorted)

    # Print top 10 results
    print("\nTop 10 Results:")
    for i, result in enumerate(all_results_sorted[:10], 1):
        print(f"{i}. Score: {result['score']:.6f}, Parameters: {result}")

    # Save best parameters and score
    np.save('hyperparameter_optimal_4', grid_search.best_params_)
    np.save('value_optimal_4', grid_search.best_score_)
