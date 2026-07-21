import numpy as np
import matplotlib.pyplot as plt
from structural_modules import Matrix_Gen_TE, RCWA_Multi_TE, RCWA_multi_TM, Utils
import pandas as pd
from scipy.interpolate import UnivariateSpline 
from tqdm import tqdm
import time
#UnivariateSpline does not do a good interpolation if the given csv got very different wavelength values


if __name__ == '__main__':
    #Define parameters

    c0 = 299792458
    theta = 0.0      
    wavelength = np.linspace(0.3,4,200)
    
    d = [0.06, 0.06]         
    N = len(d)
    period = np.full(N, 0.6)
    f1 = [[0.0], [0.0]]
    f2 = [[0.3], [0.0]]
    Num_ord = 25;



    rIndex_raw_W = pd.read_csv("csv/refractive_indexes/palik_W.csv", sep =  r"\s+")
    rIndex_raw_SiO2 = pd.read_csv("csv/refractive_indexes/palik_SiO2.csv", sep =  r"\s+")
    
    
    R = []
    T = []
    A = []
    E = []

    for wv in tqdm(wavelength, desc="Wavelength loop"):
        refractive_index_W = Utils.interp_ep(wv, rIndex_raw_W["wv"], rIndex_raw_W["n"], rIndex_raw_W["k"])
        refractive_index_SiO2 = Utils.interp_ep(wv, rIndex_raw_SiO2["wv"], rIndex_raw_SiO2["n"], rIndex_raw_SiO2["k"])
        
        e_m = np.array([refractive_index_W, refractive_index_W])
        e_d = np.array([1.1+1e-12*1j, refractive_index_SiO2])
        e = [1, refractive_index_W]
        R_wv, T_wv = RCWA_multi_TM.TransRefl(N, e_m, e_d, f1, f2, period, d, e, wv, theta, Num_ord, permittivity_matrix = None)
        A_wv = 1 - R_wv - T_wv
        E_wv = 1 - R_wv #Emittance
        E.append(E_wv)
        R.append(R_wv)
        T.append(T_wv)
        A.append(A_wv)

    
    plt.figure(figsize=(8,5))
    plt.plot(wavelength, E, label = "Emittance (R)", color = "pink")
    # plt.plot(wavelength, R, label = "Reflectance (R)", color = "blue")
    # plt.plot(wavelength, T, label = "Transmission (T)", color = "green")
    # plt.plot(wavelength, A, label = "Absorption (R)", color = "red")
    plt.xlabel("wavelength")
    plt.ylabel("Coefficients")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    
