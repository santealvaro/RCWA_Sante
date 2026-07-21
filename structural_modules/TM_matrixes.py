# SPDX-License-Identifier: GPL-3.0-or-later
#
# Copyright (C) 2026 Alvaro Sante Insua
#
# This file is part of RCWA_Sante.

import numpy as np
from scipy.linalg import toeplitz
import structural_modules.Utils
import pandas as pd

# --------------------------------------------------------------------------
# Module to calculate Q, V, and W matrices (TM polarization) for computing transmission (T)
# and reflection (R) coefficients using RCWA (Rigorous Coupled-Wave Analysis)
# --------------------------------------------------------------------------

def matrixesQVW_RectangularGrating(e_m, e_d, f_left_end, f_right_end, period, e_inc, wavelength, theta, Num_ord):
    """
    Calculates Q, V, and W matrices for a rectangular grating in TM polarization.
    
    Parameters
    ----------
    e_m : complex
        Permittivity of the metallic material.
    e_d : complex
        Permittivity of the dielectric filling material.
    f_left_end : list or ndarray of float
        Left-edge positions of the metallic strips within one period.
    f_right_end : list or ndarray of float
        Right-edge positions of the metallic strips within one period.
    period : float
        Period of the grating.
    e_inc : complex
        Permittivity of the incident medium.
    wavelength : float
        Wavelength of the incident light.
    theta : float
        Angle of incidence in radians.
    Num_ord : int
        Highest order of the Fourier expansion (number of harmonics).
    
    Returns
    -------
    Q : ndarray of complex
        Diagonal matrix containing the square roots of the eigenvalues.
    V : ndarray of complex
        Matrix of eigenvectors multiplied by Q.
    W : ndarray of complex
        Matrix of eigenvectors of the system matrix.
    """
    eps = 10e-6
    
    if (len(f_right_end) != len(f_left_end)):
        raise("There must be the same number of left and right ends")
    ordMax = Num_ord
    ordMin = -Num_ord
    ordDif = 2*Num_ord + 1
    m = np.arange(ordMin, ordMax + 1)
    
    k0 = 2 * np.pi / wavelength # Free-space wave number
    Kxi = k0 * (np.sqrt(e_inc) * np.sin(theta) + m * wavelength / period) # x-component of wavevector for each diffraction order
    Kx = np.diag(Kxi / k0) 

    # ---------------------- Calculating Fourier coefficients ----------------------
    #These formulas come from integrating permitivitty * e^(-j*m*2pi/period * x) dx
    
    #positive region m = 1, 2, 3...
    m = np.arange(1, ordMax - ordMin + 1)
    a_positive = np.zeros(len(m), dtype = complex)
    a_positive_inverse = np.zeros(len(m), dtype = complex)
    for m_i in range(1, len(m) + 1):           
        tmp = 0.0
        for i in range(len(f_left_end)):
            exp_sum = (
                np.exp(-1j * 2*np.pi * m_i * f_right_end[i] / period)
                - np.exp(-1j * 2*np.pi * m_i * f_left_end[i] / period)
            )
            tmp += exp_sum
    
        a_positive[m_i - 1] = (
            1j / (2 * np.pi * m_i) * (e_m - e_d) * tmp
        )
        a_positive_inverse [m_i - 1] = (
            1j / (2 * np.pi * m_i) * (1/e_m - 1/e_d) * tmp
        )

    #0th order
    zero_sum = 0.0
    for i in range(len(f_left_end)):
        temp = f_right_end[i] - f_left_end[i]
        zero_sum += temp
    a_zero = 1 / period * (e_d * period + (e_m - e_d) * zero_sum)
    a_zero_inverse = 1 / period * (1/e_d * period + (1/e_m - 1/e_d) * zero_sum)

    #negative region m = -N_harmonics // 2, N_harmonics // 2 + 1...
    a_negative = np.zeros(len(m), dtype = complex)
    a_negative_inverse = np.zeros(len(m), dtype = complex)
    
    for m_i in range(1, len(m)):          
        tmp = 0.0
        for i in range(len(f_left_end)):
            exp_sum = (
                np.exp(1j * 2*np.pi * m_i * f_right_end[i] / period)
                - np.exp(1j * 2*np.pi * m_i * f_left_end[i] / period)
            )
            tmp += exp_sum
    
        a_negative[-m_i] = (
            1j / (2 * np.pi * -m_i) * (e_m - e_d) * tmp
        )
        a_negative_inverse[-m_i] = (
            1j / (2 * np.pi * -m_i) * (1/e_m - 1/e_d) * tmp
        )
    # Concatenate negative, zero, and positive coefficients
    epsilonG = np.concatenate((a_negative, a_zero, a_positive), axis = None) #axis = None so that arrays are concatenated into a single array
    epsilonG_inverse = np.concatenate((a_negative_inverse, a_zero_inverse, a_positive_inverse), axis = None)
    
    # compute convolution matrix
    E = np.zeros((ordDif, ordDif), dtype=complex)
    E_inv = np.zeros((ordDif, ordDif), dtype=complex)
    
    for i in range(ordDif):
        start = ordDif + 1 - i - 2  
        stop = start + 2 * Num_ord + 1
        E[i, :] = epsilonG[start:stop]
        E_inv[i, :] = epsilonG_inverse[start:stop]

    # Para E
    col_E = epsilonG[ordDif-1:]
    row_E = epsilonG[:ordDif][::-1]
    E = toeplitz(col_E, row_E)
    
    # Para E_inv (misma estructura, distintos coeficientes)
    col_Einv = epsilonG_inverse[ordDif-1:]
    row_Einv = epsilonG_inverse[:ordDif][::-1]
    E_inv = toeplitz(col_Einv, row_Einv)
    
    I = np.eye(ordDif, dtype = complex)
    #B matrix defined in Gaylord and moharam paper
    B = Kx @ np.linalg.inv(E) @ Kx - I
    #when solving the differential equation for the magnetic field we need to multipy the convolution matrix and B. That's where A comes from
    A = np.linalg.inv(E_inv) @ B 
    # B = Kx @ E_inv @ Kx - I
    # A = E @ B

    #matrixes for RCWA
    eigvals, W = np.linalg.eig(A)
    Q = np.diag(np.sqrt(eigvals))
    V = E_inv @ W @ Q

    return Q, V, W


if __name__ == '__main__':
    e_m = 6.0
    e_d = 1.0
    e_inc = 1.0 
    f_left_end = [0.0]
    f_right_end = [0.5]
    period = 1
    wavelength = 12
    theta = 0.0
    Num_ord = 21
    Q, V, W = matrixesQVW_RectangularGrating(e_m, e_d, f_left_end, f_right_end, period, e_inc, wavelength, theta, Num_ord)
    
    df = pd.DataFrame(Q)
    df.to_csv("Q_nueva.csv", index = False, header = False)
        

    
    
    
