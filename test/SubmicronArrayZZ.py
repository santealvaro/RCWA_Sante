import numpy as np
import matplotlib.pyplot as plt
from structural_modules import Matrix_Gen_TE, Matrix_Gen_TM, RCWA_Multi_TE, RCWA_multi_TM, Utils, RCWA_multi_TM, RCWA_multi_TM_Cambios, RCWA_Multi_TE_Cambios

import pandas as pd
from scipy.interpolate import UnivariateSpline 
from tqdm import tqdm
import time
#UnivariateSpline does not do a good interpolation if the given csv got very different wavelength values


#Let's make an example in Zhuoming Zhang's program's documentation

if __name__ == '__main__':
    #Define parameters
    C0 = np.float64(299792458.0)      # speed of light in vacuum (float64)
    theta = np.float64(0.0)
    
    wavelength = np.linspace(
        np.float64(1.67),
        np.float64(9.0),
        300,
        dtype=np.float64
    )
    
    d = np.array([0.3, 0.2], dtype=np.float64)
    N = d.size
    
    period = np.full(N, np.float64(0.8), dtype=np.float64)
    
    # vectores columna (mejor para RCWA)
    f1 = np.array([[0.0], [0.0]], dtype=np.float64)
    f2 = np.array([[0.504], [0.504]], dtype=np.float64)

    Num_ord = 50 #can't be larger than profileLength // 2
    e_d = np.full(N, 1+1e-12 * 1j, dtype = np.complex128)


    rIndex_raw_Au = pd.read_csv("csv/refractive_indexes/gold_n_k_palik.csv", sep = ",")
    rIndex_raw_Ti = pd.read_csv("csv/refractive_indexes/titanium_n_k_mash.csv", sep = ",")
    rIndex_raw_Si = pd.read_csv("csv/refractive_indexes/Li_293K_Si.csv", sep = ",")
    n = len(rIndex_raw_Si["n"])  
    rIndex_raw_Si["k"] = np.zeros(n)   
    
    R = []
    T = []
    A = []

    for wv in tqdm(wavelength, desc="Wavelength loop"):
        refractive_index_Au = Utils.interp_ep(wv, rIndex_raw_Au["wl"], rIndex_raw_Au["n"], rIndex_raw_Au["k"])
        refractive_index_Ti = Utils.interp_ep(wv, rIndex_raw_Ti["wl"], rIndex_raw_Ti["n"], rIndex_raw_Ti["k"])
        refractive_index_Si = Utils.interp_ep(wv, rIndex_raw_Si["wl"], rIndex_raw_Si["n"], rIndex_raw_Si["k"])

        global_refr_index = np.array([refractive_index_Au, refractive_index_Ti])

        e = [1, refractive_index_Si]
        
        R_wv, T_wv = RCWA_multi_TM_Cambios.TransRefl(N, global_refr_index, e_d, f1, f2, period, d, e, wv, theta, Num_ord, permittivity_matrix = None)
        A_wv = 1 - R_wv - T_wv
        R.append(R_wv)
        T.append(T_wv)
        A.append(A_wv)

    
    plt.figure(figsize=(8,5))
    plt.plot(wavelength, R, label = "Reflectance (R)", color = "blue")
    plt.plot(wavelength, T, label = "Transmission (T)", color = "green")
    # plt.plot(wavelength, A, label = "Absorption (R)", color = "red")
    plt.xlabel("wavelength")
    plt.ylabel("Coefficients")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


