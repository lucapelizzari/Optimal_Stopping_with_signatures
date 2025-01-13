#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 30 16:03:23 2024

@author: lucapelizzari
"""
import Sig_Kernel_Package
SigKernel1 = Sig_Kernel_Package.SigKernel
from static_kernels import RBFKernel, LinearKernel
from joblib import Parallel, delayed
import numpy as np
import torch

class SignatureKernelComputer:
    """
    Computes signature kernels for given samples.
    """
    def __init__(self, static_kernel_type='RBF', sigma=0.5,scaling=1.0,subsampling_dual = "uniform",L=2**8,N1=12,truncation = 4):
        self.static_kernel_type = static_kernel_type
        self.sigma = sigma
        self.scaling = scaling
        self.subsampling_dual = subsampling_dual
        self.L = L
        self.N1 = N1
        self.truncation = truncation
    def Signature_Kernel_diagonal(self,X,Y):
        if self.static_kernel_type == 'RBF':
            static_kernel = RBFKernel(sigma=self.sigma)
        elif self.static_kernel_type == 'linear':
            static_kernel = LinearKernel()
        else:
            raise ValueError(f"Invalid static_kernel_type: {self.static_kernel_type}")

        signature_kernel = SigKernel1(static_kernel, dyadic_order=0)
        return signature_kernel.compute_kernel(X, Y).numpy()
    def Signature_Kernel_Gram(self, X, Y):
        """
        Compute Signature Kernel Matrix for given Samples.

        Args:
            X (np.ndarray): First set of samples.
            Y (np.ndarray): Second set of samples.

        Returns:
            np.ndarray: Computed Signature Kernel Matrix.
        """
        if self.static_kernel_type == 'RBF':
            static_kernel = RBFKernel(sigma=self.sigma)
        elif self.static_kernel_type == 'linear':
            static_kernel = LinearKernel()
        else:
            raise ValueError(f"Invalid static_kernel_type: {self.static_kernel_type}")

        signature_kernel = SigKernel1(static_kernel, dyadic_order=0)
        return signature_kernel.compute_Gram_diagonal(X, Y, sym=True).numpy()

    def Signature_kernel_parallel_samples(self, Samples):
        """
        Compute in parallel the signature kernel diagonal on sample subsets.

        Args:
            Samples (list): List of  m sample paths for signature kernels, 
                            tuples array of size (M x (N+1), L x (N+1)), 
                            N number of time-steps.

        Returns:
            list: List of m signature kernels matrices, each one is array of size M x L x (N+1).
        """
        m = len(Samples)
        results = Parallel(n_jobs=m)(delayed(self.Signature_Kernel_Gram)(sample[0], sample[1]) for sample in Samples)
        return results
    def index_set_selected_paths(self,X):
        last_point_values = X[:, -1]
        index_ordering = np.argsort(last_point_values)
        W = len(index_ordering)
        return index_ordering[0:W+1:int((W+1)/self.L)]
    def Signature_Kernel_Gram_exercise_dates(self,X_training,X_testing):
        """

        Compute Gram matrix of Signature Kernel at each exercise date for 
        (X^j,X^i), j=1,..,M and i=1,..,L subsamples
        The subsamples are choosen by ordering the samples at each exercise date by the value, and choosing the j*100/(L+1)th percentile
        for j = 1,...,L
        Inputs: - X_training,X_testing independent, time-augmented paths of size Mx(N+1)xdim, where X[:,:,0] is the underlying path
        Outputs: - List of Gram-matrices at each exercise date of size M x L x k for exercise date k for both training and testing data

        """
       
        N = X_training.shape[1]
        N = N-1
        subindex = [int((j+1)*N/self.N1) for j in range(self.N1)]
        subindex2 = [int((j)*N/self.N1) for j in range(self.N1+1)]
        index_selected_training_samples = np.zeros((self.L,self.N1-1))
        for k in reversed(range(1,self.N1)):
            index_selected_training_samples[:,k-1] = self.index_set_selected_paths(X_training[:,0:subindex[k-1]+1,0])
        #compute kernel matrices at each subset
        Subsample_training_sets = [[np.sqrt(self.scaling)*torch.from_numpy(X_training[:,0:subindex[k]+1,:]),np.sqrt(self.scaling)*torch.from_numpy(X_training[[int(j) for j in index_selected_training_samples[:,k]],0:subindex[k]+1,:])] for k in range(self.N1-1)]
        Subsample_training_sets_selected = [[Subsample_training_sets[k][1],Subsample_training_sets[k][1]] for k in range(self.N1-1)]
        Subsample_testing_sets = [[np.sqrt(self.scaling)*torch.from_numpy(X_testing[:,0:subindex[k]+1,:]),np.sqrt(self.scaling)*torch.from_numpy(X_training[[int(j) for j in index_selected_training_samples[:,k]],0:subindex[k]+1,:])] for k in range(self.N1-1)]
        #Next we parallely compute all signature kernels matrix at each exercise date
        Kernel_training = self.Signature_kernel_parallel_samples(Subsample_training_sets)
        Kernel_testing = self.Signature_kernel_parallel_samples(Subsample_testing_sets)
        #Kernel_training_selected = self.Signature_kernel_parallel_samples(Subsample_training_sets_selected)
        return Kernel_training,Kernel_testing
    def Signature_Kernel_Gram_dual(self,X_training,X_testing):
        """
        Compute Gram matrix of Signature kernel at maturity for 
        (X^j,X^i), j=1,..,M and i=1,..,L subsamples, where the subampling method is specified by self.subsampling_dual
        Input: - X_training,X_testing independent, time-augmented paths of size Mx(N+1)xdim, where X[:,:,0] is the underlying path
        Output: - Kernel_training,Kernel_testing Gram matrices as numpy arrays of size M x M x (N+1)
        """
        M,N,_ = X_training.shape
        N = N-1
        subindex = [int((j+1)*N/self.N1) for j in range(self.N1)]
        subindex2 = [int((j)*N/self.N1) for j in range(self.N1+1)]

        if self.subsampling_dual == "uniform":
            index_selected_training_samples_dual = self.index_set_selected_paths(X_training[:,-1,0])
        elif self.subsampling_dual == "non-uniform":
            Kernel_diagonal_dual = np.zeros((M,self.N1))
            for k in range(1,self.N1+1):
                x_test = X_training[:,0:subindex2[k],:]
                Kernel_diagonal_dual[:,k-1] = self.Signature_Kernel_diagonal(torch.from_numpy(x_test),torch.from_numpy(x_test))
            p_kernel_training = 1/np.sum(np.linalg.norm(Kernel_diagonal_dual,axis=1))*np.linalg.norm(Kernel_diagonal_dual,axis=1)
            index_selected_training_samples_dual = np.random.choice(np.arange(0,M),p=p_kernel_training,replace = False, size =self.L)
        Kernel_training_dual = self.Signature_Kernel_Gram(np.sqrt(self.scaling)*torch.from_numpy(X_training),np.sqrt(self.scaling)*torch.from_numpy(X_training[[int(j) for j in index_selected_training_samples_dual]]))
        Kernel_testing_dual = self.Signature_Kernel_Gram(np.sqrt(self.scaling)*torch.from_numpy(X_testing), np.sqrt(self.scaling)*torch.from_numpy(X_training[[int(j) for j in index_selected_training_samples_dual]]))
        return Kernel_training_dual,Kernel_testing_dual


    def truncated_sig_kernel(self,X, Y, num_levels, sigma=1., order=-1):
    """
    Computes the (truncated) signature kernel matrix of a given array of sequences. 
    
    Inputs:
    :X: a numpy array of shape (num_seq_X, len_seq_X, num_feat) of num_seq_X sequences of equal length, len_seq_X, with num_feat coordinates
    :Y: a numpy array of shape (num_seq_Y, len_seq_Y, num_feat) of num_seq_Y sequences of equal length, len_seq_Y, with num_feat coordinates
    :num_levels: the number of signature levels used
    :sigma: a scalar or an np array of shape (num_levels+1); a multiplicative factor for each level
    :order: the order of the signature kernel as per Kiraly and Oberhauser, order=num_levels gives the full signature kernel, while order < num_levels gives a lower order approximation. Defaults to order=-1, which means order=num_levels
    
    Output:
    :K: a numpy array of shape (num_seq_X, num_seq_Y)
    """
    order = num_levels if order < 1 else order
    sigma = sigma * np.ones((num_levels + 1,), dtype=X.dtype)
    
    num_seq_X, len_seq_X, num_feat = X.shape
    num_seq_Y, len_seq_Y, _ = Y.shape
    
    M = np.reshape(X.reshape((-1, num_feat)) @ Y.reshape((-1, num_feat)).T, (num_seq_X, len_seq_X, num_seq_Y, len_seq_Y))
    K = sigma[0] * np.ones((num_seq_X, num_seq_Y), dtype=X.dtype) + sigma[1] * np.sum(M, axis=(1, 3))
    R = M[None, None, ...]
    
    for m in range(1, num_levels):
        d = min(m+1, order)
        R_next = np.empty((d, d, num_seq_X, len_seq_X, num_seq_Y, len_seq_Y), dtype=X.dtype)
        R_next[0, 0] = M * shift(np.cumsum(np.cumsum(np.sum(R, axis=(0, 1)), axis=1), axis=3), shift=(0, 1, 0, 1))
        for j in range(1, d):
            R_next[0, j] = 1./(j+1) * M * shift(np.cumsum(np.sum(R[:, j-1], axis=0), axis=1), shift=(0, 1, 0, 0))
            R_next[j, 0] = 1./(j+1) * M * shift(np.cumsum(np.sum(R[j-1, :], axis=0), axis=3), shift=(0, 0, 0, 1))
            for i in range(1, d):
                R_next[i, j] = 1./((j+1)*(i+1)) * M * R[i-1, j-1]
        R = R_next
        K += sigma[m+1] * np.sum(R, axis=(0, 1, 3, 5))
    return K
    

    
