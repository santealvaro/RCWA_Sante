"""
RCWA Convergence & Timing Analysis
====================================
Fija: theta=0, wavelength=3.1 µm
Barre: Num_ord (convergencia en orden de Fourier)
       slices  (convergencia en discretización vertical)
Guarda resultados en CSV.
"""
 
import numpy as np
import pandas as pd
import time
from tqdm import tqdm
 
import structural_modules.Utils as Utils
import structural_modules.TM_solver as TM_solver
import structural_modules.TE_solver as TE_solver
 
# ─────────────────────────────────────────────
#  Parámetros fijos
# ─────────────────────────────────────────────
THETA      = 0.0   # radianes
WAVELENGTH = 3.1   # µm
E          = [1, 1]
 
surface_profile  = pd.read_csv("csv/gratings/triangles.csv", sep=",")
rIndex_raw_Ag    = pd.read_csv("csv/refractive_indexes/Unused/palik_Ag.csv", sep=r"\s+")
 
refractive_index = Utils.interp_ep(
    WAVELENGTH,
    rIndex_raw_Ag["wv"],
    rIndex_raw_Ag["n"],
    rIndex_raw_Ag["k"]
)
 
# ─────────────────────────────────────────────
#  Helper: una sola evaluación TM+TE
# ─────────────────────────────────────────────
def evaluate(Num_ord: int, slices: int):
    N = slices - 1
    e_m = np.full(N, refractive_index)
    e_d = np.full(N, 1.0)
 
    f_left, f_right, d, periods, perm_mat = Utils.discretize_rugosity_square_staricase(
        surface_profile, e_m=refractive_index, e_d=1.0, slices=slices
    )
 
    t0 = time.perf_counter()
    R_TM, T_TM = TM_solver.TransRefl(
        N=N, e_m=e_m, e_d=e_d,
        f1=f_left, f2=f_right,
        period=periods, d=d, e=E,
        wavelength=WAVELENGTH, theta=THETA,
        Num_ord=Num_ord, permittivity_matrix=None
    )
    t_TM = time.perf_counter() - t0
 
    t0 = time.perf_counter()
    R_TE, T_TE = TE_solver.TransRefl(
        N=N, e_m=e_m, e_d=e_d,
        f1=f_left, f2=f_right,
        period=periods, d=d, e=E,
        wavelength=WAVELENGTH, theta=THETA,
        Num_ord=Num_ord, permittivity_matrix=None
    )
    t_TE = time.perf_counter() - t0
 
    return R_TM, T_TM, t_TM, R_TE, T_TE, t_TE
 
 
# ═══════════════════════════════════════════════════════════
#  BARRIDO 1: Num_ord variable  (slices fijo en 80)
# ═══════════════════════════════════════════════════════════
SLICES_FIXED  = 40
num_ord_range = list(range(1, 301, 2))   # 1, 3, 5, … 99  (50 puntos)
 
rows_nord = []
print(f"\n{'='*55}")
print(f"  Barrido Num_ord  (slices={SLICES_FIXED}, λ={WAVELENGTH} µm)")
print(f"{'='*55}")
 
for nord in tqdm(num_ord_range, desc="Num_ord sweep"):
    R_TM, T_TM, t_TM, R_TE, T_TE, t_TE = evaluate(nord, SLICES_FIXED)
    rows_nord.append({
        "Num_ord"  : nord,
        "slices"   : SLICES_FIXED,
        "R_TM"     : float(np.real(R_TM)),
        "T_TM"     : float(np.real(T_TM)),
        "A_TM"     : float(1 - np.real(R_TM) - np.real(T_TM)),
        "time_TM_s": t_TM,
        "R_TE"     : float(np.real(R_TE)),
        "T_TE"     : float(np.real(T_TE)),
        "A_TE"     : float(1 - np.real(R_TE) - np.real(T_TE)),
        "time_TE_s": t_TE,
    })
 
df_nord = pd.DataFrame(rows_nord)
df_nord.to_csv("convergence_vs_NumOrd.csv", index=False)
print(f"\n  ✓ Guardado: convergence_vs_NumOrd.csv  ({len(df_nord)} filas)")
 
 
# ═══════════════════════════════════════════════════════════
#  BARRIDO 2: slices variable  (Num_ord fijo en 50)
# ═══════════════════════════════════════════════════════════
NUMORD_FIXED = 50
slices_range = list(range(5, 201, 5))   # 5, 10, 15, … 200  (40 puntos)
 
rows_slices = []
print(f"\n{'='*55}")
print(f"  Barrido slices  (Num_ord={NUMORD_FIXED}, λ={WAVELENGTH} µm)")
print(f"{'='*55}")
 
for sl in tqdm(slices_range, desc="slices sweep"):
    R_TM, T_TM, t_TM, R_TE, T_TE, t_TE = evaluate(NUMORD_FIXED, sl)
    rows_slices.append({
        "Num_ord"  : NUMORD_FIXED,
        "slices"   : sl,
        "R_TM"     : float(np.real(R_TM)),
        "T_TM"     : float(np.real(T_TM)),
        "A_TM"     : float(1 - np.real(R_TM) - np.real(T_TM)),
        "time_TM_s": t_TM,
        "R_TE"     : float(np.real(R_TE)),
        "T_TE"     : float(np.real(T_TE)),
        "A_TE"     : float(1 - np.real(R_TE) - np.real(T_TE)),
        "time_TE_s": t_TE,
    })
 
df_slices = pd.DataFrame(rows_slices)
df_slices.to_csv("convergence_vs_slices.csv", index=False)
print(f"\n  ✓ Guardado: convergence_vs_slices.csv  ({len(df_slices)} filas)")
 
 