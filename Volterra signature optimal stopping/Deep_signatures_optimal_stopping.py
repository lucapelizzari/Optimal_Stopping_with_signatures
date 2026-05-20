#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 29 10:22:26 2024

@author: lucapelizzari
This file contains three ways of computing lower and upper bounds to the optimal stopping problem: using linear signatures, using 
deep neural networks on signatures, and using the signature kernel.

"""

import shap
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import layers, models, optimizers, regularizers
from functools import reduce
from tensorflow.keras.layers import Input, Layer, Multiply
import matplotlib.pyplot as plt
import timeshap
class DeepLongstaffSchwartzPricer:
    """
    Computes the lower bound of optimal stopping problem using deep neural networks on signatures.
    """
    def __init__(self, N1, T, r, mode="Standard", layers=3, nodes=64, activation_function='relu',
                 batch_normalization=False, regularizer=0, dropout=False, layer_normalization=False):
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
        layers : int
            Number of hidden layers
        nodes : int
            Number of neurons in each hidden layer
        activation_function : str
            Activation function for hidden layers
        batch_normalization : bool
            Whether to use batch normalization at the input
        regularizer : float
            L2 regularization factor
        dropout : bool
            Whether to use dropout
        layer_normalization : bool
            Whether to use layer normalization
        """
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

    def price(self, S_training_sig, Payoff_training, S_testing_sig, Payoff_testing, 
              batch=32, epochs=100, learning_rate=0.001, M_val=0):
        """
        Parameters
        ----------
        S_training_sig : numpy array
            (Log) Signature (+ possibly polynomial features) of the augmented path for the training set
        Payoff_training : numpy array
            Payoff for training paths
        S_testing_sig : numpy array
            (Log) Signature (+ possibly polynomial features) of the augmented path for the testing set
        Payoff_testing : numpy array
            Payoff for testing paths
        batch : int
            Batch size for training
        epochs : int
            Number of epochs for training
        learning_rate : float
            Learning rate for training
        M_val : int
            Number of validation paths
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
        value = Payoff_exercise_training[:, -1]
        
        dtt = np.exp(-self.r * self.T / (self.N1 + 1))
        
        if self.mode == "Standard":
            """
            Standard mode: Use all paths for regression at each exercise date
            """
            M_old = M
            for k in reversed(range(1, self.N1)):
                M_new = M_old - M_val
                S_exercise_training_new = S_exercise_training_sig[:M_new, k-1, :]
                value_new = value[:M_new] * dtt
                S_exercise_training_validation = S_exercise_training_sig[M_new:M_old, k-1, :]
                value_validation = value[M_new:M_old]
                
                regr[k-1] = LongstaffSchwartzModel(
                    feature_dim=feature_dim,
                    layers_number=self.layers,
                    nodes=self.nodes+feature_dim,
                    activation_function=self.activation_function,
                    batch_normalization=self.batch_normalization,
                    regularizer=self.regularizer,
                    dropout=self.dropout,
                    layer_normalization=self.layer_normalization
                )
                
                regr[k-1].compile(learning_rate=learning_rate, loss='mse', metrics=[tf.keras.metrics.R2Score()])
                
                if k < self.N1 - 1:
                    """
                    Transfer weights from the next exercise date to the current one, and reduce to one epoch.
                    """
                    try:
                        regr[k-1].set_weights(regr[k].get_weights())
                        epochs = 1

                    except ValueError as e:
                        print(f"Unable to transfer weights from step {k} to {k-1}: {e}")
                        print("Continuing with randomly initialized weights")
                
                print(f"Regression at exercise date {k}")
                early_stopping = EarlyStopping(monitor='loss', patience=5,restore_best_weights=True)
                if M_val == 0:
                    regr[k-1].fit(
                    S_exercise_training_new,
                    value_new,
                    batch_size=batch,
                    epochs=epochs,
                    verbose=1,
                    callbacks=[early_stopping]
                )
                else:
                    regr[k-1].fit(
                    S_exercise_training_new,
                    value_new,
                    batch_size=batch,
                    epochs=epochs,
                    verbose=1,
                    validation_data=(S_exercise_training_validation, value_validation),
                    callbacks=[early_stopping]
                    )


                
                
                reg = regr[k-1].predict(S_exercise_training_new)
                
                for m in range(M_new):
                    if reg[m] <= Payoff_exercise_training[m, k-1]:
                        value_new[m] = Payoff_exercise_training[m, k-1]
                
                value = value_new
                M_old = M_new
     
        
        elif self.mode == "American Option":
            """
            American Option mode: Only use in the money paths for regression at each exercise date
            """
            for j in reversed(range(1, self.N1)):
                value = value * dtt
                ITM = [m for m in range(M) if Payoff_exercise_training[m, j-1] > 0]
                
                if len(ITM) <= 1:
                    continue
                
                regr[j-1] = LongstaffSchwartzModel(
                    feature_dim=feature_dim,
                    layers_number=self.layers,
                    nodes=self.nodes+feature_dim,
                    activation_function=self.activation_function,
                    batch_normalization=self.batch_normalization,
                    regularizer=self.regularizer,
                    dropout=self.dropout,
                    layer_normalization=self.layer_normalization
                )
                
                regr[j-1].compile(learning_rate=learning_rate, loss='mse', metrics=[tf.keras.metrics.R2Score()])
                
                if j < self.N1 - 1:
                    """
                    Transfer weights from the next exercise date to the current one, and reduce to one epoch.
                    """
                    try:
                        regr[j-1].set_weights(regr[j].get_weights())
                        epochs = 1

                    except ValueError as e:
                        print(f"Unable to transfer weights from step {j} to {j-1}: {e}")
                        print("Continuing with randomly initialized weights")
                
                print(f"Regression at exercise date {j}")
                early_stopping = EarlyStopping(monitor='loss', patience=5)
                # Fit the model
                regr[j-1].fit(
                    S_exercise_training_sig[ITM, j-1, :],
                    value[ITM],
                    batch_size=batch,
                    epochs=epochs,
                    verbose=1,
                    callbacks=[early_stopping]
                )
                
                value_estimate = regr[j-1].predict(S_exercise_training_sig[ITM, j-1, :])
                
                for m, itm in enumerate(ITM):
                    # Update the value of the option if the estimated value is less than the payoff
                    if value_estimate[m] <= Payoff_exercise_training[itm, j-1]:
                        value[itm] = Payoff_exercise_training[itm, j-1]
        
        else:
            raise ValueError(f"Invalid mode: {self.mode}")
        
        # Compute true lower bound for testing data
        value_testing = Payoff_exercise_testing[:, -1]
        reg = np.zeros((M2, self.N1))
        
        for j in range(self.N1 - 1):
            # If the model is not trained, set the value to a large number
            if regr[j] is None:
                reg[:, j] = 10**8
            else:
                # Predict the value of the option at the current exercise date
                reg[:, j] = regr[j].predict(S_exercise_testing_sig[:, j, :])[:, 0]
        
        if self.mode == "Standard":
            """
            Standard mode: Use all paths for regression at each exercise date
            """
            for m in range(M2):
                i = 0
                while i < self.N1 - 1 and reg[m, i] > Payoff_exercise_testing[m, i]:
                    i += 1
                value_testing[m] = Payoff_exercise_testing[m, i] * np.exp(-self.r * self.T * (ttt[i+1] - ttt[0]))
        
        elif self.mode == "American Option":
            """
            American Option mode: Only use in the money paths for regression at each exercise date
            """
            for m in range(M2):
                i = 0
                while i < self.N1 - 1 and (Payoff_exercise_testing[m, i] == 0 or reg[m, i] > Payoff_exercise_testing[m, i]):
                    i += 1
                value_testing[m] = Payoff_exercise_testing[m, i] * np.exp(-self.r * self.T * (ttt[i+1] - ttt[1]))
        
        lower_bound = np.mean(value_testing)
        point_estimate = np.mean(value)
        lower_bound_std = np.std(value_testing)
        
        return lower_bound, lower_bound_std, regr, point_estimate

class DeepDualPricer:
    """
    Computes upper bounds of optimal stopping problem using deep neural networks on signatures.
    """
    def __init__(self, N1, N, T, r, layers=3, nodes=64, activation_function='relu',
                 batch_normalization=False, regularizer=0.01, dropout=False,
                 attention_layer=False, layer_normalization=False,mode_dim = "1-dim"):
        """
        Parameters
        ----------
        N1 : int
            Number of exercise dates for optimal stopping
        N : int
            Number of time steps
        T : float
            Time horizon for the option
        r : float
            Risk-free interest rate
        layers : int
            Number of hidden layers
        nodes : int
            Number of neurons in each hidden layer
        activation_function : str
            Activation function for hidden layers
        batch_normalization : bool
            Whether to use batch normalization at the input
        regularizer : float
            L2 regularization factor
        dropout : bool
            Whether to use dropout
        attention_layer : bool
            Whether to use attention layer
        layer_normalization : bool
            Whether to use layer normalization
        """
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
        self.mode_dim = mode_dim

    def price(self, S_training_sig, Payoff_training, dW_training, S_testing_sig, Payoff_testing, dW_testing,
              M_val, batch=32, epochs=100, learning_rate=0.001):
        """
        Parameters
        ----------
        S_training_sig : numpy array
            (Log) Signature (+ possibly polynomial features) of the augmented path for the training set
        Payoff_training : numpy array
            Payoff for training paths
        dW_training : numpy array
            Brownian motion increments for training paths
        S_testing_sig : numpy array
            (Log) Signature (+ possibly polynomial features) of the augmented path for the testing set
        Payoff_testing : numpy array
            Payoff for testing paths
        dW_testing : numpy array
            Brownian motion increments for testing paths
        M_val : int
            Number of validation paths
        batch : int
            Batch size for training
        epochs : int
            Number of epochs for training
        learning_rate : float
            Learning rate for training
        """
        M, _, D = S_training_sig.shape
        M2, _, _ = S_testing_sig.shape
        subindex = [int((j+1)*self.N/self.N1) for j in range(self.N1)]
        subindex2 = [int((j)*self.N/self.N1) for j in range(self.N1+1)]
        model, rule_model = DualNetworkModel(
            n=self.N1 + 1,
            n2=self.N + 1,
            I=self.layers,
            q=self.nodes + D,
            d=D,
            activation_function=self.activation_function,
            batch_normalization=self.batch_normalization,
            regularizer=self.regularizer,
            dropout=self.dropout,
            attention_layer=self.attention_layer,
            layer_normalization=self.layer_normalization,
            mode_dim = self.mode_dim
        ).build_network_dual()

        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        model.compile(optimizer=optimizer)

        early_stopping = EarlyStopping(monitor='val_loss', patience=10,restore_best_weights=True)
        if self.mode_dim == "1-dim":
            model.fit(
            [S_training_sig[:M_val, :, :], Payoff_training[:M_val, 1:self.N+1], dW_training[:M_val, :]],
            y=None,
            batch_size=batch,
            epochs=epochs,
            verbose=1,
            validation_data=([S_training_sig[M_val:,:,:], Payoff_training[M_val:, 1:self.N+1], dW_training[M_val:, :]], None),
            callbacks=[early_stopping]
            )
            res = model.predict([S_testing_sig, Payoff_testing[:, 1:self.N+1], dW_testing])
            MG = rule_model.predict([S_testing_sig, dW_testing])
            MG = np.concatenate((np.zeros((M2,1)), MG), axis=-1)
            # Compute the upper bound and standard deviation
            
            upper_bound = np.mean(np.max(Payoff_testing[:, subindex2] - MG[:, subindex2], axis=1))
            upper_bound_std = np.std(np.max(Payoff_testing[:, subindex2] - MG[:, subindex2], axis=1))

            y0 = np.mean(res)

            return y0, upper_bound, upper_bound_std, model, rule_model
        elif self.mode_dim == "2-dim":
            dW1_training = dW_training[:,:,0]
            dW2_training = dW_training[:,:,1]
            dW1_testing = dW_testing[:,:,0]
            dW2_testing = dW_testing[:,:,1]
            model.fit(
            [S_training_sig[:M_val, :, :], Payoff_training[:M_val, 1:self.N+1], dW1_training[:M_val, :],dW2_training[:M_val, :]],
            y=None,
            batch_size=batch,
            epochs=epochs,
            verbose=1,
            validation_data=([S_training_sig[M_val:,:,:], Payoff_training[M_val:, 1:self.N+1], dW1_training[M_val:, :],dW2_training[M_val:, :]], None),
            callbacks=[early_stopping]
            )
            res = model.predict([S_testing_sig, Payoff_testing[:, 1:self.N+1], dW1_testing,dW2_testing])
            MG = rule_model.predict([S_testing_sig, dW1_testing,dW2_testing])
            MG = np.concatenate((np.zeros((M2,1)), MG), axis=-1)
            model.summary()

            # Compute the upper bound and standard deviation
            
            upper_bound = np.mean(np.max(Payoff_testing[:, subindex2] - MG[:, subindex2], axis=1))
            upper_bound_std = np.std(np.max(Payoff_testing[:, subindex2] - MG[:, subindex2], axis=1))

            y0 = np.mean(res)
            return y0, upper_bound, upper_bound_std, model, rule_model

class RReLU(tf.keras.layers.Layer):
    def __init__(self, lower=0.125, upper=0.333, **kwargs):
        super(RReLU, self).__init__(**kwargs)
        self.lower = lower
        self.upper = upper

    def call(self, inputs, training=None):
        if training:
            # Randomly sample the negative slope from a uniform distribution
            alpha = tf.random.uniform(shape=tf.shape(inputs), minval=self.lower, maxval=self.upper)
            return tf.where(inputs >= 0, inputs, alpha * inputs)
        else:
            # Use the average slope during inference
            alpha = (self.lower + self.upper) / 2.0
            return tf.where(inputs >= 0, inputs, alpha * inputs)
        
        

class LongstaffSchwartzModel:
    """
    A neural network model class for the Longstaff-Schwartz approach for optimal stopping.

    Attributes:
        feature_dim (int): The input dimension of the network (dimension of the feature map/signature).
        layers_number (int): Number of hidden layers.
        nodes (int): Number of neurons in each hidden layer.
        activation_function (str): Activation function for hidden layers.
        batch_normalization (bool): Whether to use batch normalization at the input.
        regularizer (float): L2 regularization factor.
        dropout (bool): Whether to use dropout.
        layer_normalization (bool): Whether to use layer normalization.
    """

    def __init__(self, feature_dim, layers_number, nodes, activation_function='relu',
                 batch_normalization=False, regularizer=0.01, dropout=False,
                 layer_normalization=False):
        self.feature_dim = feature_dim
        self.layers_number = layers_number
        self.nodes = nodes
        self.activation_function = activation_function
        self.batch_normalization = batch_normalization
        self.regularizer = regularizer
        self.dropout = dropout
        self.layer_normalization = layer_normalization
        self.model = self.build_model_longstaff_schwartz()

    def build_model_longstaff_schwartz(self):
        model = models.Sequential()

        # Input layer with optional batch normalization
        if self.batch_normalization:
            model.add(layers.BatchNormalization(input_shape=(self.feature_dim,)))
        else:

            model.add(layers.Input(shape=(self.feature_dim,)))

        # Set activation function
        if self.activation_function == "LeakyRelu":
            activation = tf.keras.layers.LeakyReLU(negative_slope=0.3)
        elif self.activation_function == "RRelu":
            activation = RReLU()
            
        else:
            activation = self.activation_function

        # Hidden layers
        for _ in range(self.layers_number):
            if self.layer_normalization:
                model.add(layers.LayerNormalization(epsilon=1e-6))
            model.add(layers.Dense(self.nodes, activation=activation,
                                   kernel_regularizer=regularizers.l2(self.regularizer)))
            if self.layer_normalization:
                model.add(layers.LayerNormalization(epsilon=1e-6))
            if self.dropout:
                model.add(layers.Dropout(0.5))

        # Output layer
        model.add(layers.Dense(1, activation='linear'))

        return model

    def compile(self, learning_rate=0.001, loss='mse', metrics=['mae']):
        optimizer = optimizers.Adam(learning_rate=learning_rate)
        self.model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    def fit(self, X, y, epochs=100, batch_size=32, verbose=1, callbacks=None):
        return self.model.fit(X, y, epochs=epochs, batch_size=batch_size,
                              verbose=verbose, callbacks=callbacks)

    def predict(self, X):
        return self.model.predict(X)

    def summary(self):
        return self.model.summary()

    def save(self, filepath):
        self.model.save(filepath)

    @classmethod
    def load(cls, filepath):
        loaded_model = models.load_model(filepath)
        instance = cls(feature_dim=loaded_model.input_shape[1], layers_number=0, nodes=0)
        instance.model = loaded_model
        return instance

    def get_weights(self):
        return self.model.get_weights()

    def set_weights(self, weights):
        self.model.set_weights(weights)



class DeepMartingales(tf.keras.layers.Layer):
    def __init__(self):
        super(DeepMartingales, self).__init__()
        self.steps = None

    def build(self, input_shape):
        self.steps = input_shape[-1]

    def call(self, inputs, **kwargs):
        rule, dW = inputs
        return tf.cumsum(rule * dW, axis=1)
    
    

class DualStoppingLoss(tf.keras.layers.Layer):
    def call(self, inputs):
        rule, Y = inputs
        
        out = tf.math.reduce_mean(tf.math.reduce_max(Y-rule,axis=1))

        self.add_loss(out)

        return out



class DualNetworkModel:
    def __init__(self, n, n2, I, q, d, activation_function='relu',
                 batch_normalization=False, regularizer=0.01, dropout=False,
                 attention_layer=False, layer_normalization=False,mode_dim = "1-dim"):
        self.n = n  # exercise dates
        self.n2 = n2  # discretization
        self.I = I  # number of layers
        self.q = q  # number of neurons
        self.d = d  # input dimension
        self.mode_dim = mode_dim
        self.activation_function = activation_function
        self.batch_normalization = batch_normalization
        self.regularizer = regularizer
        self.dropout = dropout
        self.attention_layer = attention_layer
        self.layer_normalization = layer_normalization
        self.model, self.rule_model = self.build_network_dual()
    


   

    def rnn_neural_network_dual(self):
        # Choose the activation function
        if self.activation_function == "LeakyRelu":
            activation = tf.keras.layers.LeakyReLU(negative_slope=0.3)
        else:
            activation = self.activation_function

        # Initialize the layers list
        layers_list = []
        
        # Batch Normalization (if applicable)
        if self.batch_normalization:
            layers_list.append(layers.BatchNormalization())
        
        # First RNN layer (Simple RNN)
        layers_list.append(layers.SimpleRNN(self.q, return_sequences=True, activation=activation,
                                            kernel_regularizer=regularizers.l2(self.regularizer)))

        # Additional RNN layers
        for _ in range(self.I - 1):
            if self.layer_normalization:
                layers_list.append(layers.LayerNormalization(epsilon=1e-6))
            if self.attention_layer:
                layers_list.append(layers.MultiHeadAttention(num_heads=4,
                                                            key_dim=max(1, self.q // 4),
                                                            dropout=0.1))
            layers_list.append(layers.SimpleRNN(self.q, return_sequences=True, activation=activation,
                                                kernel_regularizer=regularizers.l2(self.regularizer)))
            if self.layer_normalization:
                layers_list.append(layers.LayerNormalization(epsilon=1e-6))
            if self.dropout:
                layers_list.append(layers.Dropout(0.5))

        # Final output layer
        layers_list.append(layers.Dense(1, activation=None))
        layers_list.append(layers.Flatten())

        return layers_list

    def dense_neural_network_dual(self):
        if self.activation_function == "LeakyRelu":
            activation = tf.keras.layers.LeakyReLU(negative_slope=0.3)
        else:
            activation = self.activation_function

        layers_list = []
        #layers_list.append(FeatureSelectionLayer())
        if self.batch_normalization:
            layers_list.append(layers.BatchNormalization())
        num_attention_heads = 4
        layers_list.append(tf.keras.layers.Dense(self.q, activation=activation,
                                                 kernel_regularizer=regularizers.l2(self.regularizer),kernel_initializer = tf.keras.initializers.GlorotUniform()))
        #layers_list.append(SelfAttentionLayer(num_heads=4, key_dim=max(1, self.q // num_attention_heads), dropout=0.1))
        
        for _ in range(self.I - 1):
            if self.layer_normalization:
                layers_list.append(layers.LayerNormalization(epsilon=1e-6))
            if self.attention_layer:
                layers_list.append(layers.MultiHeadAttention(num_heads=num_attention_heads,
                                                             key_dim=max(1, self.q // num_attention_heads),
                                                             dropout=0.1))
            layers_list.append(tf.keras.layers.Dense(self.q, activation=activation,
                                                     kernel_regularizer=regularizers.l2(self.regularizer),kernel_initializer = tf.keras.initializers.GlorotUniform()))
            if self.layer_normalization:
                layers_list.append(layers.LayerNormalization(epsilon=1e-6))
            if self.dropout:
                layers_list.append(tf.keras.layers.Dropout(0.5))

        layers_list.append(tf.keras.layers.Dense(1, activation=None))
        layers_list.append(tf.keras.layers.Flatten())

        return layers_list

    def build_network_dual(self, initial_model=None):
        print(f"Building network with parameters: I={self.I}, q={self.q}, d={self.d}, activation={self.activation_function}")
        input_logsig = tf.keras.Input(shape=(self.n2, self.d), name='sig')
        input_y = tf.keras.Input(shape=(self.n2-1,), name='Y')
        
        
        if self.mode_dim == "1-dim":
            dnn_layers = self.dense_neural_network_dual()
            print(f"Number of layers created: {len(dnn_layers)}")
            input_BM = tf.keras.Input(shape=(self.n2-1,), name='dW')
            
            try:
                dnn_output = reduce(lambda x, f: f(x), [input_logsig] + dnn_layers)
            except Exception as e:
                print(f"Error in reduce function: {str(e)}")
                print(f"input_logsig shape: {input_logsig.shape}")
                print(f"dnn_layers: {dnn_layers}")
                raise
            rule_layer = DeepMartingales()([dnn_output[:, 0:self.n2-1], input_BM])
            loss_layer = DualStoppingLoss()([
                rule_layer[:, int((self.n2-1)/(self.n-1))-1:self.n2:int((self.n2-1)/(self.n-1))],
                input_y[:, int((self.n2-1)/(self.n-1))-1:self.n2:int((self.n2-1)/(self.n-1))]
            ])

            model = tf.keras.Model([input_logsig, input_y, input_BM], loss_layer)
            rule_model = tf.keras.Model([input_logsig, input_BM], rule_layer)

            if initial_model is not None:
                model.set_weights(initial_model.get_weights())

            return model, rule_model
        elif self.mode_dim == "2-dim":
            dnn_layers_1 = self.dense_neural_network_dual()
            dnn_layers_2 = self.dense_neural_network_dual()
            print(f"Number of layers created: {len(dnn_layers_1)} for each network")
            input_BM_1 = tf.keras.Input(shape=(self.n2-1,), name='dW1')
            input_BM_2 = tf.keras.Input(shape=(self.n2-1,), name='dW2')
            try:
                dnn_output_1 = reduce(lambda x, f: f(x), [input_logsig] + dnn_layers_1)
                dnn_output_2 = reduce(lambda x, f: f(x), [input_logsig] + dnn_layers_2)
            except Exception as e:
                print(f"Error in reduce function: {str(e)}")
                print(f"input_logsig shape: {input_logsig.shape}")
                print(f"dnn_layers: {dnn_layers}")
                raise
            rule_layer_1 = DeepMartingales()([dnn_output_1[:, 0:self.n2-1], input_BM_1])
            rule_layer_2 = DeepMartingales()([dnn_output_2[:, 0:self.n2-1], input_BM_2])
            loss_layer = DualStoppingLoss()([
                rule_layer_1[:, int((self.n2-1)/(self.n-1))-1:self.n2:int((self.n2-1)/(self.n-1))]+rule_layer_2[:, int((self.n2-1)/(self.n-1))-1:self.n2:int((self.n2-1)/(self.n-1))],
                input_y[:, int((self.n2-1)/(self.n-1))-1:self.n2:int((self.n2-1)/(self.n-1))]
            ])

            model = tf.keras.Model([input_logsig, input_y, input_BM_1,input_BM_2], loss_layer)
            rule_model = tf.keras.Model([input_logsig, input_BM_1,input_BM_2], rule_layer_1+rule_layer_2)
            if initial_model is not None:
                model.set_weights(initial_model.get_weights())

            return model, rule_model


        

    def compile(self, optimizer='adam', loss='mse', metrics=['mae']):
        self.model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    def fit(self, X, y, epochs=100, batch_size=32, verbose=1):
        return self.model.fit(X, y, epochs=epochs, batch_size=batch_size, verbose=verbose)

    def predict(self, X):
        return self.rule_model.predict(X)

    def summary(self):
        return self.model.summary()

    def save(self, filepath):
        self.model.save(filepath)

    @classmethod
    def load(cls, filepath):
        loaded_model = models.load_model(filepath)
        # You might need to adjust this part depending on how you want to handle the loaded model
        instance = cls(n=1, n2=1, I=1, q=1, d=loaded_model.input_shape[0][2])
        instance.model = loaded_model
        return instance