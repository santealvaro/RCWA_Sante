import numpy as np
import matplotlib.pyplot as plt
from structural_modules import Matrix_Gen_TE, Matrix_Gen_TM, RCWA_Multi_TE, RCWA_multi_TM, Utils
import pandas as pd
from scipy.interpolate import UnivariateSpline 
from tqdm import tqdm
import time
#UnivariateSpline does not do a good interpolation if the given csv got very different wavelength values


#Let's make an example in Zhuoming Zhang's program's documentation

if __name__ == '__main__':
    # -------------------- Define parameters ------------------------------
    
    theta = 0 # incident angle
    wavelength = np.linspace(0.2, 3.0, 200) # Array of wavelengths (in micrometers)
    d = np.array([0.8]) # Thickness of each layer (single-layer structure here)
    N = d.size # Number of layers in the structure
    period = np.full(N, 0.4) # Grating period for each layer (uniform period)
    width = 0.2 #width of the metallic strip
    f1 = [[0]] # Left-edge position of the metallic strip (per layer)
    f2 = [[width]] # Right-edge position of the metallic strip (per layer)
    e = [1, 1] # Permittivities of incident medium and substrate
    Num_ord = 50 # Highest Fourier order considered in the RCWA expansion
    e_d = np.full(N, 1+1e-12 * 1j) # Dielectric permittivity


    # -------------------- Load material data --------------------
    
    rIndex_raw_Ag = pd.read_csv("csv/refractive_indexes/palik_Ag.csv", sep = r"\s+")
    
    # -------------------- Initialize result arrays --------------------
    R = []
    T = []
    A = []

    # -------------------- Wavelength sweep --------------------
    for wv in tqdm(wavelength, desc="Wavelength loop"): #Tqdm used to show the progress bar
        # Interpolate complex permittivity of silver at the current wavelength
        refractive_index_Ag = Utils.interp_ep(wv, rIndex_raw_Ag["wv"], rIndex_raw_Ag["n"], rIndex_raw_Ag["k"])
        
        # Assign metal permittivity to all layers
        e_m = np.full(N, refractive_index_Ag)  

        # Define dielectric permittivity (air)
        e_d = np.full(N, 1.0) 

        # Compute reflection and transmission using RCWA (TE polarization)
        R_wv, T_wv = RCWA_multi_TE.TransRefl(N, e_m, e_d, f1, f2, period, d, e, wv, theta, Num_ord, permittivity_matrix = None)
        
        # Store results
        A_wv = 1 - R_wv - T_wv
        R.append(R_wv)
        T.append(T_wv)
        A.append(A_wv)
     
    
    # -------------------- plot the result  --------------------
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


