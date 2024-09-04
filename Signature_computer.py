#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 29 11:05:48 2024

This modul contains a class for computing linear and log signatures for augmented paths, 
for various choices of signature lifts, and polynomial features.

@author: lucapelizzari
"""

import numpy as np
import scipy.special as sc
import iisignature as ii

class SignatureComputer:
    """
    Computes signatures of augmented paths
    Attributes:
           T (float): The time horizon for the option.
           N (int): The number of time steps.
           K (int): The truncation level for signatures.
           signature_spec (str): Specifies 'linear' or 'log' signatures.
           signature_lift (str): The type of signature lift to apply.
           poly_degree (int): The degree of polynomial features to add.
    """
    def __init__(self, T, N, K, signature_spec, signature_lift, poly_degree):
        self.T = T
        self.N = N
        self.K = K
        self.signature_spec = signature_spec
        self.signature_lift = signature_lift
        self.poly_degree = poly_degree
        self.tt = np.linspace(0, T, N+1)
    def compute_signature(self, X, vol, A, Payoff):
        """
        Computes the signature of the augmented path X, vol, A, Payoff
        X = state process, array of Mx(N+1)
        vol = volatility process, array of Mx(N+1)
        A = monoton compononent for augmentation (e.g. time, QV), array of Mx(N+1)
        Payoff = payoff process, array of Mx(N+1)

        Output is linear or log signature of augmented path, array of Mx(N+1)x(K+1), 
        with potentially additional polynomial features.
        """
        print(f"Computing {self.signature_spec} signature with {self.signature_lift} lift")
        if self.signature_spec == "linear":
            result = self._compute_linear_signature(X, vol, A, Payoff)
        elif self.signature_spec == "log":
            result = self._compute_log_signature(X, vol, A, Payoff)
        else:
            raise ValueError(f"Invalid signature_spec: {self.signature_spec}")
        
        
        return result

    def _compute_linear_signature(self, X, vol, A, Payoff):
        """
        Computes the linear signature of the augmented path X, vol, A, Payoff, for differnet choices of signature lift.
        normal: signature of the augmented path (a,X)
        payoff-extended: signature of the augmented path (A,X,Payoff)
        delay: signature of the augmented path (A,X,X_delay)
        polynomial-extended: signature of the augmented path (X,A) + Laguerre polynomials of X
        payoff-and-polynomial-extended: signature of the augmented path (A,X,Payoff) + Laguerre polynomials of (X,vol)
        polynomial-vol: signature of the augmented path (A,vol) + Laguerre polynomials of X
        """
        dX = X[:, 1:] - X[:, :-1]
        dvol = vol[:,1:]-vol[:,:-1]
        if self.signature_lift == "normal":
            return self._signatureQV(self.tt, dX.reshape(X.shape[0], self.N, 1), A)
        elif self.signature_lift == "payoff-extended":
            dP = Payoff[:, 1:] - Payoff[:, :-1]
            dXX = np.stack((dX, dP), axis=-1)
            return self._signatureQV(self.tt, dXX, A)
        elif self.signature_lift == "delay":
            dX_delay = np.zeros_like(dX)
            dX_delay[:, 1:] = dX[:, :-1]
            dXX = np.stack((dX, dX_delay), axis=-1)
            return self._signatureQV(self.tt, dXX, A)
        elif self.signature_lift == "polynomial-extended":
            Sig = self._signatureQV(self.tt, dX.reshape(X.shape[0], self.N, 1), A)
            Poly = self._compute_polynomials(X)
            return np.concatenate((Sig, Poly), axis=-1)
        elif self.signature_lift == "payoff-and-polynomial-extended":
            dP = Payoff[:, 1:] - Payoff[:, :-1]
            dXX = np.stack((dX, dP), axis=-1)
            Sig = self._signatureQV(self.tt, dXX, A)
            Poly = self._compute_polynomials_2dim(X,vol)
            return np.concatenate((Sig, Poly), axis=-1)
        elif self.signature_lift == "polynomial-vol":
            Sig = self._signatureQV(self.tt, dvol.reshape(X.shape[0], self.N, 1), A)
            Poly = self._compute_polynomials(X)
            return np.concatenate((Sig, Poly), axis=-1)
        else:
            raise ValueError(f"Invalid signature_lift for linear signature: {self.signature_lift}")
    def _compute_log_signature(self, X, vol, A, Payoff):
        """
        Computes the log signature of the augmented path X, vol, A, Payoff, for differnet choices of signature lift.
        normal: log signature of the augmented path (a,X)
        payoff-extended: log signature of the augmented path (A,X,Payoff)
        delay: log signature of the augmented path (A,X,X_delay)
        polynomial-extended: log signature of the augmented path (X,A) + Laguerre polynomials of X
        payoff-and-polynomial-extended: log signature of the augmented path (A,X,Payoff) + Laguerre polynomials of (X,vol)
        polynomial-vol: log signature of the augmented path (A,vol) + Laguerre polynomials of X
        """
        dX = X[:, 1:] - X[:, :-1]
        dvol = vol[:, 1:] - vol[:, :-1]
        
        if self.signature_lift == "normal":
            XX = np.stack([A, X], axis=-1)
            return self._full_log_signature(XX)
        elif self.signature_lift == "payoff-extended":
            dP = Payoff[:, 1:] - Payoff[:, :-1]
            XX = np.stack([A, X, Payoff], axis=-1)
            return self._full_log_signature(XX)
        elif self.signature_lift == "delay":
            X_delay = np.zeros_like(X)
            X_delay[:, 1:] = X[:, :-1]
            XX = np.stack([A, X, X_delay], axis=-1)
            return self._full_log_signature(XX)
        elif self.signature_lift == "polynomial-extended":
            XX = np.stack([A, X], axis=-1)
            Sig = self._full_log_signature(XX)
            Poly = self._compute_polynomials(X)
            return np.concatenate((Sig, Poly), axis=-1)
        elif self.signature_lift == "payoff-and-polynomial-extended":
            dP = Payoff[:, 1:] - Payoff[:, :-1]
            XX = np.stack([A, X, Payoff], axis=-1)
            Poly = self._compute_polynomials_2dim(X,vol)
            return np.concatenate((Sig, Poly), axis=-1)

        elif self.signature_lift == "polynomial-vol":
            XX = np.stack([A, vol], axis=-1)
            Sig = self._full_log_signature(XX)
            Poly = self._compute_polynomials(X)
            return np.concatenate((Sig, Poly), axis=-1)
        else:
            raise ValueError(f"Invalid signature_lift for log signature: {self.signature_lift}")

    def _compute_polynomials(self, X):
        """
        Computes the Laguerre polynomials of X
        """
        Polynomials = np.zeros((X.shape[0], X.shape[1], self.poly_degree))
        for k in range(self.poly_degree):
            Polynomials[:,:,k] = sc.laguerre(k+1)(X)
        return Polynomials
    def _compute_polynomials_2dim(self, X,vol):
        """
        Computes the Laguerre polynomials of (X,vol)
        """
        DD_primal = int((self.poly_degree+1)*(self.poly_degree+2)/2) #Number of polynomials 2 dim
        Polynomials = np.zeros((X.shape[0], X.shape[1], DD_primal))
        for k in range(self.poly_degree+1):
            for j in range(0,k+1):
                C = np.zeros((self.poly_degree+1,self.poly_degree+1))
                C[k,j] = 1
                Polynomials[:,:,int(k*(k+1)/2+j)] = np.polynomial.laguerre.lagval2d(X,vol, C)
        return Polynomials
    
    def _signatureQV(self, tGrid, dx, QV):
        """
        Compute the signature of a path (t,x,[x]) up to degree K.

        Parameters
        ----------
        tGrid : numpy array
            Time grid of t, size N+1.
        dx : numpy array
            Increments of the path x, an array of dimension MxNxd.
        QV : numpy array
            Quadratic variation of the path.

        Returns
        -------
        sig : numpy array
            The signature of (t,x,[x]) at all the times, an array of size (M,N,k+1).
        """
        M, d, z = self._prepare_sigQV(tGrid, dx, QV)
        
        # We need to compute the signature of z
        k = ii.siglength(d+1, self.K)
        N = len(tGrid)
        
        sig = np.zeros((M, N, k+1))
        for m in range(M):
            sig[m, 1:N, 1:k+1] = ii.sig(z[m, :, :], self.K, 2)
        sig[:, :, 0] = 1

        return sig
    
    def _prepare_sigQV(self, tGrid, dx, QV):
        """Auxiliary function for computing signatures. See help for signature."""
        N = len(tGrid) - 1
        if len(dx.shape) == 1:
            # assume that M = d = 1
            dx = dx.reshape((1, dx.shape[0], 1))
        if len(dx.shape) == 2:
            # assume that either d = 1 or M = 1
            if dx.shape[0] == N:
                dx = dx.reshape((1, dx.shape[0], dx.shape[1]))
            elif dx.shape[1] == N:
                dx = dx.reshape((dx.shape[0], dx.shape[1], 1))
        assert len(dx.shape) == 3 and dx.shape[1] == N, \
            f"dx is misshaped as {dx.shape}"
        M, _, d = dx.shape
        QV = QV.reshape(M, N+1, 1)
        x = np.zeros((M, N+1, d))
        x[:, 1:(N+1), :] = np.cumsum(dx, axis=1)
        z = np.concatenate((QV, x), axis=2)
        return M, d, z  # d+1 because we added the QV dimension


    

    def _full_log_signature(self, X):
        """
        Compute the full log signature of the given paths.

        Args:
            X (np.ndarray): The paths to compute the log signature for.

        Returns:
            np.ndarray: The computed log signatures.
        """
        m, n, d = X.shape
        
        if (d == 2) and (self.K <= 3):
            return self._full_log_signature_dim_two_level_three(X, self.K)
        else:
            log_sig = np.zeros((m, n, ii.logsiglength(d, self.K)))
            bch = ii.prepare(d, self.K, 'C')  # precalculate the BCH formula
            for i in range(1, n):
                log_sig[:, i] = ii.logsig(X[:,:i+1], bch, 'C')
        
        return log_sig
    def _full_log_signature_dim_two_level_three(self, X, deg):
        """
        Compute the full log signature for 2D paths up to level 3.

        Args:
            X (np.ndarray): The paths to compute the log signature for.
            deg (int): The degree of the log signature (1, 2, or 3).

        Returns:
            np.ndarray: The computed log signatures.

        Raises:
            AssertionError: If the input dimensions are incorrect or deg is out of range.
        """
        m, n, d = X.shape
        
        assert d == 2
        assert (deg >= 1) and (deg <= 3)
        
        log_sig_dim = {1: 2, 2: 3, 3: 5}
        
        log_sig = np.zeros((m, n, log_sig_dim[deg]))
        
        log_sig[:,1:,:2] = X[:, 1:] - X[:, 0].reshape(-1, 1, 2)
            
        if deg >= 2:
            dX = np.diff(X, axis=1)
            
            for i in range(1, n):
                l = log_sig[:, i - 1]
                dx = dX[:, i - 1]
                log_sig[:, i, 2] = l[:, 2] + 0.5 * (l[:, 0] * dx[:, 1] - l[:, 1] * dx[:, 0])
                
        if deg == 3:
            for i in range(1, n):
                l = log_sig[:, i - 1]
                dx = dX[:, i - 1]
                log_sig[:, i, 3] = l[:, 3] - 0.5 * l[:, 2] * dx[:, 0] + (1 / 12) * \
                        (l[:, 0] - dx[:, 0]) * (l[:, 0] * dx[:, 1] - l[:, 1] * dx[:, 0])
                log_sig[:, i, 4] = l[:, 4] + 0.5 * l[:, 2] * dx[:, 1] - (1 / 12) * \
                        (l[:, 1] - dx[:, 1]) * (l[:, 0] * dx[:, 1] - l[:, 1] * dx[:, 0])
                
        return log_sig
    
    
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

