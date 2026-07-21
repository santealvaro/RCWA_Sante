import numpy as np
import matplotlib.pyplot as plt
from structural_modules import Matrix_Gen_TE, Matrix_Gen_TM, RCWA_Multi_TE, RCWA_multi_TM, Utils, RCWA_multi_TM_Cambios

import pandas as pd
from scipy.interpolate import UnivariateSpline 
from tqdm import tqdm
import time
#UnivariateSpline does not do a good interpolation if the given csv got very different wavelength values

def drudeAg(lambda_):
    import numpy as np

    lambda_ = np.float64(lambda_)

    C0 = np.float64(2.99792458e8)
    w = 2.0 * np.pi * C0 / (lambda_ * 1e-6)

    e_inf = np.float64(3.4)
    w_p1 = np.float64(1.39e16)
    r1 = np.float64(2.70e13)

    epsilon = np.complex128(
        e_inf - (w_p1**2) / (w * (w + 1j * r1))
    )

    return epsilon

#Let's make an example in Zhuoming Zhang's program's documentation

if __name__ == '__main__':
    #Define parameters
    C0 = np.float64(299792458.0)      # speed of light in vacuum (float64)
    theta = np.float64(0.0)
    
    wavelength = np.linspace(
        np.float64(2.0),
        np.float64(4.0),
        200,
        dtype=np.float64
    )
    
    d = np.array([0.2], dtype=np.float64)
    N = d.size
    
    period = np.full(N, np.float64(0.4), dtype=np.float64)
    
    # vectores columna (mejor para RCWA)
    f1 = np.array([[0.0]], dtype=np.float64)
    f2 = np.array([[0.395]], dtype=np.float64)

    Num_ord = 100 #can't be larger than profileLength // 2
    e_d = np.full(N, 1+1e-12 * 1j, dtype = np.complex128)

 
    E = []
    R = []
    T = []
    A = []

    for wv in tqdm(wavelength, desc="Wavelength loop"):

        e_m = np.full(N, drudeAg(wv))
        e = [1, e_m]
        
        R_wv, T_wv = RCWA_multi_TM_Cambios.TransRefl(N, e_m, e_d, f1, f2, period, d, e, wv, theta, Num_ord, permittivity_matrix = None)
        A_wv = 1 - R_wv - T_wv
        E_wv = 1 - R_wv
        R.append(R_wv)
        T.append(T_wv)
        A.append(A_wv)
        E.append(E_wv)       
    
    plt.figure(figsize=(8,5))
    # plt.plot(wavelength, R, label = "Reflectance (R)", color = "blue")
    plt.plot(wavelength, E, label = "Emittance (E)", color = "pink")
    # plt.plot(wavelength, T, label = "Transmission (T)", color = "green")
    # plt.plot(wavelength, A, label = "Absorption (R)", color = "red")
    plt.xlabel("wavelength")
    plt.ylabel("Coefficients")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


