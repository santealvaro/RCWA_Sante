# SPDX-License-Identifier: GPL-3.0-or-later
#
# Copyright (C) 2026 Alvaro Sante Insua
#
# This file is part of RCWA_Sante.

import numpy as np
from scipy.linalg import toeplitz
from . import Utils
from . import TM_matrixes
import pandas as pd

# --------------------------------------------------------------------------
# Module to compute diffraction efficiencies for TE polarization using RCWA
# --------------------------------------------------------------------------

def TransRefl(N, e_m, e_d, f1, f2, period, d, e, wavelength, theta, Num_ord, permittivity_matrix):
    """
    Computes the diffraction efficiencies (reflection and transmission) 
    for a multilayer grating using TM polarization.
    
    Parameters
    ----------
    N : int
        Number of layers in the grating.
    e_m : ndarray of complex
        Permittivity of the metallic material for each layer.
    e_d : ndarray of complex
        Permittivity of the dielectric filling material for each layer.
    f1 : list of ndarray of float
        Left-edge positions of the metallic strips for each layer.
    f2 : list of ndarray of float
        Right-edge positions of the metallic strips for each layer.
    period : ndarray of float
        Grating period for each layer.
    d : ndarray of float
        Thickness of each layer.
    e : list or ndarray of complex
        Permittivities of the incident and substrate media 
        in the form [incident_medium, substrate_medium].
    wavelength : float
        Wavelength of the incident light.
    theta : float
        Angle of incidence in radians.
    Num_ord : int
        Highest Fourier order considered in the expansion.
    permittivity_matrix : ndarray of complex or None
        Optional permittivity profiles for arbitrary grating layers 
        (used when FFT-based convolution matrices are required).
    
    Returns
    -------
    Ref : float
        Total reflected diffraction efficiency (sum over all orders).
    Tran : float
        Total transmitted diffraction efficiency (sum over all orders).
    """
    # -------------------- Define diffraction orders --------------------
    ordMax = Num_ord
    ordMin = -Num_ord
    ordDif = ordMax - ordMin + 1
    
    m = np.arange(ordMin, ordMax + 1)

    # Identity matrix
    I = np.eye(ordDif)

    # Dirac delta vector for incident wave
     
    Dirac_del = Utils.DiracDel(ordDif, Num_ord)
    

    k0 = 2 * np.pi / wavelength # Free-space wave number
    Kxi = k0 * (np.sqrt(e[0]) * np.sin(theta) + m * wavelength / period[0]) # x-component of wavevector for each order


    Q, V, W, X, O = [], [], [], [], [] # Lists to store matrices for each layer
    
    
    if permittivity_matrix is None:
        # Use rectangular grating model
        for ind in range(N):
            # Compute Q, V, W for the current layer
            Q_i, V_i, W_i = TM_matrixes.matrixesQVW_RectangularGrating(
                e_m[ind], e_d[ind], f1[ind], f2[ind],
                period[ind], e[0], wavelength, theta, Num_ord
            )
                
            Q.append(Q_i)
            V.append(V_i)
            W.append(W_i)
            # Compute propagation matrix for layer thickness
            eigvals = np.diag(Q_i)
            X_i = np.diag(np.exp(-k0 * d[ind] * eigvals))
            X.append(X_i)
            # Block matrix O combining W and V for boundary conditions
            O_i = np.block([
                [W_i,  W_i],
                [V_i, -V_i]
            ])
            O.append(O_i)
            
            
    
    Kz1 = np.sqrt(e[0] * k0 ** 2 - Kxi ** 2 + 0j) # incident medium propagation (add 0j to avoid warnings)
    Kz3 = np.sqrt(e[1] * k0 ** 2 - Kxi ** 2 + 0j) # substrate propagation
        
    Y_inc = np.diag(Kz1 / k0 / e[0])
    Y_sub = np.diag(Kz3 / k0 / e[1])

    # Initialize f and g matrices for Enhanced Transmittance Matrix Method recursive propagation
    f, g, a, b = [], [], [], []

    for _ in range(N + 1):
        f.append(None)
        g.append(None)
        
    #condition established in literature
    f[N] = I
    g[N] = 1j * Y_sub

    for ind in range(N-1, -1, -1):
        #procedure from literature and Zhuoming Zang code
        Mat_ab = np.linalg.inv(O[ind]) @ np.vstack((f[ind+1], g[ind+1]))
        
        # Split into submatrices
        a_i = Mat_ab[0:ordDif, :]
        b_i = Mat_ab[ordDif:2*ordDif, :]

        a.append(a_i)
        b.append(b_i)
    
        # Compute f[ind] and g[ind]
        inv_b = np.linalg.inv(b_i)
        f_i = W[ind] @ (I + X[ind] @ a_i @ inv_b @ X[ind]) #a and b are changed (compared to papers) cuz the imag part of Kz is not forced to be positive 
        g_i = V[ind] @ (-I + X[ind] @ a_i @ inv_b @ X[ind])

    
        f[ind] = f_i
        g[ind] = g_i
    
    # Compute transmission vector at input
    T1 = np.linalg.inv(1j * Y_inc @ f[0] + g[0]) @ (
        1j * Y_inc @ Dirac_del + 1j * np.sqrt(e[0]) * np.cos(theta) * Dirac_del/e[0]
    )
    R = f[0] @ T1 - Dirac_del
    
    # Propagate through layers
    T = T1
    for ind in range(N):
        T = np.linalg.inv(b[N-1-ind]) @ X[ind] @ T
    
    # Diffraction efficiencies (TE)
    DEr = (R * np.conj(R)) * np.real(Kz1) / (k0 * np.sqrt(e[0]) * np.cos(theta))
    DEt = (T * np.conj(T)) * np.real(Kz3 / e[1]) / (k0 * np.sqrt(e[0]) * np.cos(theta) / e[0])
    
    Ref = np.sum(DEr)
    Tran = np.sum(DEt)

    return np.real(Ref), np.real(Tran) #Ref and Tran are real, but they may have a very little imaginary part (e^-18 or so)

