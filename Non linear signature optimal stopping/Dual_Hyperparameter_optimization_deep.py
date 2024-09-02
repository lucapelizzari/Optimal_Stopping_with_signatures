#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 30 15:29:41 2024

@author: lucapelizzari
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 29 10:22:26 2024

@author: lucapelizzari
This file performs hyperparameter optimization for the DeepDualPricer.
"""

from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.model_selection import GridSearchCV
import numpy as np
import tensorflow as tf
from Deep_signatures_optimal_stopping import DeepDualPricer
from Signature_computer import SignatureComputer
from rBergomi_simulation import SimulationofrBergomi

class DeepDualWrapper(BaseEstimator, RegressorMixin):
    def __init__(self, N1, N, T, r, layers=3, nodes=64, activation_function='relu',
                 batch_normalization=False, regularizer=0.01, dropout=False,
                 attention_layer=False, layer_normalization=False,
                 batch_size=256, epochs=10, learning_rate=0.001):
        self.N1 = N1
        self.N = N
        self.T = T
        self.r = r
        self.layers = layers
        self.nodes = nodes
        self.activation_function = activation_function
        self.batch_normalization = batch_normalization
        self.regularizer = regularizer
        self.dropout = dropout
        self.attention_layer = attention_layer
        self.layer_normalization = layer_normalization
        self.batch_size = batch_size
        self.epochs = epochs
        self.learning_rate = learning_rate

    def fit(self, X, y):
        tf.keras.backend.clear_session()
        # Global variables (you might want to pass these as parameters instead)
        global S_training_sig, Payoff_training, dW_training, S_testing_sig, Payoff_testing, dW_testing, M_val

        pricer = DeepDualPricer(
            N1=self.N1,
            N=self.N,
            T=self.T,
            r=self.r,
            layers=self.layers,
            nodes=self.nodes,
            activation_function=self.activation_function,
            batch_normalization=self.batch_normalization,
            regularizer=self.regularizer,
            dropout=self.dropout,
            attention_layer=self.attention_layer,
            layer_normalization=self.layer_normalization
        )

        self.option_value, _, _, _, _ = pricer.price(
            S_training_sig, Payoff_training, dW_training,
            S_testing_sig, Payoff_testing, dW_testing,
            M_val, batch=self.batch_size, epochs=self.epochs, learning_rate=self.learning_rate
        )
        return self

    def predict(self, X):
        return np.full(X.shape[0], self.option_value)

    def score(self, X, y=None):
        return -self.option_value if self.option_value is not None else np.inf

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
    M_val = int(M*0.85)
    N1 = 12
    h = 0.2  # Hurst parameter
    K = 4
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
    dW_training = dB_training
    dW_testing = dB_testing
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
        'nodes': [32,64],
        'activation_function': ['relu','LeakyRelu'],
        'batch_normalization': [False],
        'regularizer': [0, 1e-8],
        'dropout': [False],
        'attention_layer': [False],
        'layer_normalization': [False],
        'batch_size': [32,128,258],
        'epochs': [10],
        'learning_rate': [0.001,0.01]
    }

    grid_search = GridSearchCV(
        DeepDualWrapper(N1=N1, N=N, T=T, r=r),
        param_grid,
        cv=4,
        n_jobs=12,
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
    print("Best (lowest) option value:", -grid_search.best_score_)

    # Get all results
    results = grid_search.cv_results_

    # Create a list of dictionaries containing parameters and scores
    all_results = []
    for mean_score, params in zip(results['mean_test_score'], results['params']):
        result_dict = params.copy()
        result_dict['score'] = -mean_score  # Negate the score to get the actual option value
        all_results.append(result_dict)

    # Sort results by score in ascending order (lowest upper bound first)
    all_results_sorted = sorted(all_results, key=lambda x: x['score'])

    # Save results
    np.save('all_hyperparameter_results_dual.npy', all_results_sorted)

    # Print top 10 results
    print("\nTop 10 Results (Lowest Upper Bounds):")
    for i, result in enumerate(all_results_sorted[:10], 1):
        print(f"{i}. Upper Bound: {result['score']:.6f}, Parameters: {result}")

    # Save best parameters and score
    np.save('hyperparameter_optimal_dual', grid_search.best_params_)
    np.save('value_optimal_dual', -grid_search.best_score_)
