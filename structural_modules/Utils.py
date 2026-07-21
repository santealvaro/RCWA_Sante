# SPDX-License-Identifier: GPL-3.0-or-later
#
# Copyright (C) 2026 Alvaro Sante Insua
#
# This file is part of RCWA_Sante.

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap


# Module used to define some useful functions for the program

def interp_ep(lambda_, wv, n, kapa):
    """
    Linearly interpolates the refractive index (n) and extinction coefficient (kappa)
    as a function of wavelength, and returns the complex permittivity.
    
    Parameters
    ----------
    lambda_ : float
        Wavelength of interest (same units as `wv`).
    wv : array_like
        1D array of wavelengths, sorted in ascending order.
    n : array_like
        1D array of refractive index (n) values corresponding to `wv`.
    kapa : array_like
        1D array of extinction coefficient (kappa) values corresponding to `wv`.
    
    Returns
    -------
    ep : complex
        Complex permittivity interpolated at the given wavelength.
        Rises error if `lambda_` is outside the wavelength range.
    """
    lambda_ = np.float64(lambda_)
    wv = np.asarray(wv, dtype = np.float64) #no crea copia del array (modifica el array original en caso de hacer cambios)
    n = np.asarray(n, dtype = np.float64)
    kapa = np.asarray(kapa, dtype = np.float64)
    
    # Return zero if lambda_ is out of range
    if lambda_ < wv[0] or lambda_ > wv[-1]:
        raise ValueError("Wavelength is out of the CSV bounds.")
    
    # Find first index where wv >= lambda_
    idx = np.searchsorted(wv, lambda_)
    
    if idx == 0:
        n_interp = n[0]
        k_interp = kapa[0]
    else:
        wL, wR = wv[idx - 1], wv[idx]
        nL, nR = n[idx - 1], n[idx]
        kL, kR = kapa[idx - 1], kapa[idx]
    
        # Linear interpolation
        n_interp = nL + (nR - nL) * (lambda_ - wL) / (wR - wL)
        k_interp = kL + (kR - kL) * (lambda_ - wL) / (wR - wL)
    
    # Compute complex permittivity
    ep = np.complex128((n_interp + 1j * k_interp) ** 2)
    return ep

def interp_ep_no_n(lambda_, wv, e1, e2):
    """
    Linearly interpolates the real permitivitty (e1) and the imaginary part (e2)
    as a function of wavelength, and returns the interpolated complex permittivity.
    
    Parameters
    ----------
    lambda_ : float
        Wavelength of interest (same units as `wv`).
    wv : array_like
        1D array of wavelengths, sorted in ascending order.
    e1 : array_like
        1D array of real permitivitty values corresponding to `wv`.
    e2 : array_like
        1D array of extinction coefficient (kappa) values corresponding to `wv`.
    
    Returns
    -------
    ep : complex
        Complex permittivity interpolated at the given wavelength.
        Rises error if `lambda_` is outside the wavelength range.
    """
    lambda_ = np.float64(lambda_)
    wv = np.asarray(wv, dtype = np.float64) #no crea copia del array (modifica el array original en caso de hacer cambios)
    e1 = np.asarray(e1, dtype = np.float64)
    e2 = np.asarray(e2, dtype = np.float64)
    
    # Return zero if lambda_ is out of range
    if lambda_ < wv[0] or lambda_ > wv[-1]:
        raise("Wavelength is out of the csv bounds.")
    
    # Find first index where wv >= lambda_
    idx = np.searchsorted(wv, lambda_)
    
    if idx == 0:
        e1_interp = e1[0]
        e2_interp = e2[0]
    else:
        wL, wR = wv[idx - 1], wv[idx]
        e1L, e1R = e1[idx - 1], e1[idx]
        e2L, e2R = e2[idx - 1], e2[idx]
    
        # Linear interpolation
        e1_interp = e1L + (e1R - e1L) * (lambda_ - wL) / (wR - wL)
        e2_interp = e2L + (e2R - e2L) * (lambda_ - wL) / (wR - wL)
    
    # Compute complex permittivity
    ep = np.complex128(e1_interp + 1j * e2_interp) 
    return ep

def DiracDel(length, pos):
    """
    Returns a discrete Kronecker delta array.

    Parameters:
    ----------
    length : int
        Length of the output array.
    pos : int
        Index position where the delta is located (0 <= pos < length).

    Returns:
    -------
    Del : ndarray of complex
        A 1D complex array of size `length` containing zeros everywhere
        except at index `pos`, where the value is 1.0.

    """
    Del = np.zeros(length, dtype = np.complex128)
    Del[pos] = 1.0

    return Del

def discretize_rugosity_square_staricase(df, e_m = 6.0, e_d = 1.0, slices = 10):
    """
    Returns a serier of data used for the RCWA simulation

    Parameters:
    ----------
    df : dataframe
        2 column dataframe which contains the surface profile.
    e_m : complex
        permittivity of the metallic material.
    e_d : complex
        permittivity of the dielectric material
    slices : int
        Number of slices the profile will be divided into.
    period : float
        Period of the grating
    Returns:
    -------
    f1 : ndarray of float
        Array containing the left-end of the strip for the slices
    f2 : ndarray of float
        Array containing the right-end of the strip for the slices
    d : ndarray of float
        Array which contains the height of each slice  
    periods: ndarray of float
        Array containing the period for the slices
    """

    # profile normalization
    min_y = np.min(df['y'].to_numpy(dtype=np.float64))
    max_y = np.max(df['y'].to_numpy(dtype=np.float64))
    period = np.max(df['x'].to_numpy(dtype=np.float64))
    

    slice_heigth = (max_y - min_y) / slices

    N_points = len(df)
    
    point_width = period / N_points

    
    slices = int(slices)
    height = 1.0 / slices
    real_heigth = (max_y - min_y) / slices

    df['normalized_y'] = (df['y'] - min_y) / (max_y - min_y)
    
    # df.plot(x='x', y='normalized_y')
    # plt.title("Normalized profile")
    # plt.xlabel("x")
    # plt.ylabel("Normalized heigth")
    # plt.show()

    # Initializing values
    
    permittivity_matrix = np.ones((slices, len(df)), dtype = np.complex128) * e_d
    
    f_left_end = [None] * slices #no se hace con np.zeros porque luego se meterán listas en esos elementos y daria value error
    f_right_end = [None] * slices
    # if slices == 1: #needed to correct this case
    #     e_m = e_d
    for i in range(0, slices):
        indexes = df.index[(df['normalized_y'] > height * i)].tolist()
        jumps = np.where(np.diff(indexes) > 1)[0]

        left_end_list = [df["x"][indexes[0]]]
        right_end_list = []
        
        for jump_idx in jumps:
            left_end_list.append(df["x"][indexes[jump_idx + 1]]) #given like this it wouldn't be normalized with respect to period => need to introduce / period in the matrix generator formula
            right_end_list.append(df["x"][indexes[jump_idx]])

        right_end_list.append(df["x"][indexes[-1]])

        f_left_end[-i-1] = np.array(left_end_list)  #because RCWA solves from top to bottom
        f_right_end[-i-1] = np.array(right_end_list) 

        
        permittivity_matrix[-i-1, indexes] = e_m

    d = np.ones(slices) * real_heigth
    periods = np.ones(slices) * period
    #N = len(d)
    #d = d[:N - 1]
    #f_left_end = f_left_end[:N-1] 
    #f_right_end = f_right_end[:N-1]
    #periods = periods[:N-1]
    #permittivity_matrix = permittivity_matrix[:N - 1, :] 


    return np.array(f_left_end, dtype = object), np.array(f_right_end, dtype = object), d, periods, permittivity_matrix

def build_convolution_matrix_1D(eps_orders, Num_ord):
    """
    Args:
        eps_orders : ndarray
            Vector de coeficientes de Fourier centrados (índice 0 = orden 0)
        Num_ord : int
            Highest order of the Fourier expansion
    
    Returns:
        E_conv : ndarray
            Convolution matrix of size (2*Num_ord+1, 2*Num_ord+1)
    """
    ordDif = 2 * Num_ord + 1
    E_conv = np.zeros((ordDif, ordDif), dtype=complex)
    
    # Los coeficientes en eps_orders deben estar ordenados como:
    # [ε_{-Num_ord}, ε_{-Num_ord+1}, ..., ε_{-1}, ε_0, ε_1, ..., ε_{Num_ord}]
    
    for i in range(ordDif):
        for j in range(ordDif):
            # El índice en la matriz de convolución corresponde a i-j
            order_index = (i - j) 
            
            # Mapear el índice al rango de eps_orders
            idx_in_eps = order_index + Num_ord
            
            if 0 <= idx_in_eps < ordDif:
                E_conv[i, j] = eps_orders[idx_in_eps]
    return E_conv




if __name__ == '__main__':
    e_m = 6.0
    e_d = 1.0
    e_inc = 1.0 
    f1 = 0.0
    f2 = 0.5
    period = 1
    wavelength = 12
    theta = 0.0
    N_harmonics = 21
    df = pd.read_csv('../csv/gratings/sample_3_h_first.csv', sep = ";")
    #df = abs(df)
    # print(df.head())
    
    f_l, f_r, d, h, permittivity_matrix = discretize_rugosity_square_staricase(df, e_m = 6.0, e_d = 1.0, slices = 61)
    
     # --- plot the matrix ---
        
    cmap_white_gray = LinearSegmentedColormap.from_list(
        "white_gray", ["white", "gray"]
    )

    x = np.linspace(0, 2 * np.pi, 1000)
    y = np.linspace(0, 1, 61)

    # print(f1[30])
    # print(f2[30])

    plt.figure(figsize=(10, 5))
    plt.imshow(
    np.real(permittivity_matrix),
    cmap=cmap_white_gray,
    aspect='auto',
    interpolation='nearest',
    extent=[x.min(), x.max(), y.min(), y.max()]  # 👈 aquí está la clave
    )
    plt.xlabel("x (µm)")
    plt.ylabel("y (µm)")
    plt.title("Gainazalaren geometria diskretizatuta")
    plt.show()
    
    
    # print(len(f_l))

    

    
    
    
