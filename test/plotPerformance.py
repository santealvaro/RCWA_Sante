"""
Plot de resultados del análisis de convergencia RCWA
=====================================================
Lee los 3 CSVs generados por convergence_analysis.py y produce:
  - Fig 1 : Convergencia de R y T vs Num_ord
  - Fig 2 : Tiempo de cómputo vs Num_ord
  - Fig 3 : Convergencia de R y T vs slices
  - Fig 4 : Tiempo de cómputo vs slices
  - Fig 5 : Diferencia relativa (convergencia) vs Num_ord
  - Fig 6 : Diferencia relativa (convergencia) vs slices
  - Fig 7 : Mapa de calor 2D — tiempo total en función de Num_ord × slices
  - Fig 8 : Mapa de calor 2D — R_TM en función de Num_ord × slices
Guarda cada figura como PNG de alta resolución.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.colors import LogNorm
import os

# ─────────────────────────────────────────────────────
#  Estética global
# ─────────────────────────────────────────────────────
STYLE = {
    "c_TM"    : "#1f77b4",
    "c_TE"    : "#d62728",
    "c_TM2"   : "#4a9fd4",
    "c_TE2"   : "#e87575",
    "c_total" : "#2ca02c",
    "lw"      : 1.8,
    "ms"      : 5,
    "alpha"   : 0.85,
}

plt.rcParams.update({
    "figure.dpi"        : 150,
    "savefig.dpi"       : 200,
    "font.family"       : "DejaVu Sans",
    "font.size"         : 10,
    "axes.titlesize"    : 11,
    "axes.labelsize"    : 10,
    "legend.fontsize"   : 9,
    "axes.spines.top"   : False,
    "axes.spines.right" : False,
    "axes.grid"         : True,
    "grid.alpha"        : 0.3,
    "grid.linestyle"    : "--",
})

OUTPUT_DIR = "plots"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def save(fig, name):
    path = os.path.join(OUTPUT_DIR, name)
    fig.savefig(path, bbox_inches="tight")
    print(f"  ✓ {path}")
    plt.close(fig)

# ─────────────────────────────────────────────────────
#  Cargar datos
# ─────────────────────────────────────────────────────
print("Cargando CSVs…")
df_nord   = pd.read_csv("convergence_vs_NumOrd.csv")
df_slices = pd.read_csv("convergence_vs_slices.csv")

WV    = 3.1   # µm (referencia)
THETA = 0.0

# ═══════════════════════════════════════════════════════
#  FIG 1 — Convergencia R y T vs Num_ord
# ═══════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=False)
fig.suptitle(f"Konbergentzia vs Num_ord  (slices={df_nord['slices'].iloc[0]}, "
             f"λ={WV} µm, θ={THETA}°)", fontsize=12, y=1.01)

for ax, qty, label in zip(
    axes,
    [("R_TM","R_TE"), ("T_TM","T_TE")],
    ["Islapena R", "Transmisioa T"]
):
    ax.plot(df_nord["Num_ord"], df_nord[qty[0]],
            color=STYLE["c_TM"], lw=STYLE["lw"], label="TM",
            marker="o", ms=STYLE["ms"]-2, markevery=5)
    ax.plot(df_nord["Num_ord"], df_nord[qty[1]],
            color=STYLE["c_TE"], lw=STYLE["lw"], label="TE",
            marker="s", ms=STYLE["ms"]-2, markevery=5, linestyle="--")
    ax.set_xlabel("Num_ord")
    ax.set_ylabel(label)
    ax.legend()
    ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())

plt.tight_layout()
save(fig, "fig1_convergencia_R_T_vs_NumOrd.png")

# ═══════════════════════════════════════════════════════
#  FIG 2 — Tiempo de cómputo vs Num_ord
# ═══════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(df_nord["Num_ord"], df_nord["time_TM_s"],
        color=STYLE["c_TM"], lw=STYLE["lw"], label="TM",
        marker="o", ms=STYLE["ms"]-2, markevery=5)
ax.plot(df_nord["Num_ord"], df_nord["time_TE_s"],
        color=STYLE["c_TE"], lw=STYLE["lw"], label="TE",
        marker="s", ms=STYLE["ms"]-2, markevery=5, linestyle="--")
ax.plot(df_nord["Num_ord"],
        df_nord["time_TM_s"] + df_nord["time_TE_s"],
        color=STYLE["c_total"], lw=STYLE["lw"]+0.5, label="Total",
        marker="^", ms=STYLE["ms"]-2, markevery=5, linestyle=":")
ax.set_xlabel("Num_ord")
ax.set_ylabel("Exekuzio-denbora (s)")
ax.set_title(f"Denbora vs Num_ord  (slices={df_nord['slices'].iloc[0]}, λ={WV} µm)")
ax.legend()
ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())
plt.tight_layout()
save(fig, "fig2_tiempo_vs_NumOrd.png")

# ═══════════════════════════════════════════════════════
#  FIG 3 — Convergencia R y T vs slices
# ═══════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=False)
fig.suptitle(f"Konbergentzia vs slices  (Num_ord={df_slices['Num_ord'].iloc[0]}, "
             f"λ={WV} µm, θ={THETA}°)", fontsize=12, y=1.01)

for ax, qty, label in zip(
    axes,
    [("R_TM","R_TE"), ("T_TM","T_TE")],
    ["Islapena R", "Transmisioa T"]
):
    ax.plot(df_slices["slices"], df_slices[qty[0]],
            color=STYLE["c_TM"], lw=STYLE["lw"], label="TM",
            marker="o", ms=STYLE["ms"]-2, markevery=4)
    ax.plot(df_slices["slices"], df_slices[qty[1]],
            color=STYLE["c_TE"], lw=STYLE["lw"], label="TE",
            marker="s", ms=STYLE["ms"]-2, markevery=4, linestyle="--")
    ax.set_xlabel("Slices")
    ax.set_ylabel(label)
    ax.legend()
    ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())

plt.tight_layout()
save(fig, "fig3_convergencia_R_T_vs_slices.png")

# ═══════════════════════════════════════════════════════
#  FIG 4 — Tiempo de cómputo vs slices
# ═══════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(df_slices["slices"], df_slices["time_TM_s"],
        color=STYLE["c_TM"], lw=STYLE["lw"], label="TM",
        marker="o", ms=STYLE["ms"]-2, markevery=4)
ax.plot(df_slices["slices"], df_slices["time_TE_s"],
        color=STYLE["c_TE"], lw=STYLE["lw"], label="TE",
        marker="s", ms=STYLE["ms"]-2, markevery=4, linestyle="--")
ax.plot(df_slices["slices"],
        df_slices["time_TM_s"] + df_slices["time_TE_s"],
        color=STYLE["c_total"], lw=STYLE["lw"]+0.5, label="Total",
        marker="^", ms=STYLE["ms"]-2, markevery=4, linestyle=":")
ax.set_xlabel("Slices")
ax.set_ylabel("Exekuzio-denbora (s)")
ax.set_title(f"Denbora vs slices  (Num_ord={df_slices['Num_ord'].iloc[0]}, λ={WV} µm)")
ax.legend()
ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())
plt.tight_layout()
save(fig, "fig4_tiempo_vs_slices.png")

# ═══════════════════════════════════════════════════════
#  FIG 5 — Diferencia relativa |ΔR/R| vs Num_ord
#           (variación punto a punto — convergencia numérica)
# ═══════════════════════════════════════════════════════
def rel_diff(series):
    """Diferencia relativa entre pasos consecutivos |x[i]-x[i-1]| / |x[i]|"""
    arr = np.array(series, dtype=float)
    diff = np.abs(np.diff(arr))
    base = np.abs(arr[1:])
    with np.errstate(divide="ignore", invalid="ignore"):
        rd = np.where(base > 1e-15, diff / base, np.nan)
    return rd

fig, axes = plt.subplots(1, 2, figsize=(11, 4))
fig.suptitle(f"Convergencia relativa |ΔR/R|, |ΔT/T| vs Num_ord  "
             f"(slices={df_nord['slices'].iloc[0]})", fontsize=12, y=1.01)

for ax, qty, label in zip(
    axes,
    [("R_TM","R_TE"), ("T_TM","T_TE")],
    ["|ΔR/R|", "|ΔT/T|"]
):
    x = df_nord["Num_ord"].values[1:]
    ax.semilogy(x, rel_diff(df_nord[qty[0]]),
                color=STYLE["c_TM"], lw=STYLE["lw"], label="TM",
                marker="o", ms=STYLE["ms"]-2, markevery=5)
    ax.semilogy(x, rel_diff(df_nord[qty[1]]),
                color=STYLE["c_TE"], lw=STYLE["lw"], label="TE",
                marker="s", ms=STYLE["ms"]-2, markevery=5, linestyle="--")
    ax.set_xlabel("Num_ord")
    ax.set_ylabel(label)
    ax.legend()
    ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())

plt.tight_layout()
save(fig, "fig5_convergencia_relativa_vs_NumOrd.png")

# ═══════════════════════════════════════════════════════
#  FIG 6 — Diferencia relativa vs slices
# ═══════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
fig.suptitle(f"Convergencia relativa |ΔR/R|, |ΔT/T| vs slices  "
             f"(Num_ord={df_slices['Num_ord'].iloc[0]})", fontsize=12, y=1.01)

for ax, qty, label in zip(
    axes,
    [("R_TM","R_TE"), ("T_TM","T_TE")],
    ["|ΔR/R|", "|ΔT/T|"]
):
    x = df_slices["slices"].values[1:]
    ax.semilogy(x, rel_diff(df_slices[qty[0]]),
                color=STYLE["c_TM"], lw=STYLE["lw"], label="TM",
                marker="o", ms=STYLE["ms"]-2, markevery=4)
    ax.semilogy(x, rel_diff(df_slices[qty[1]]),
                color=STYLE["c_TE"], lw=STYLE["lw"], label="TE",
                marker="s", ms=STYLE["ms"]-2, markevery=4, linestyle="--")
    ax.set_xlabel("slices")
    ax.set_ylabel(label)
    ax.legend()
    ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())

plt.tight_layout()
save(fig, "fig6_convergencia_relativa_vs_slices.png")



# ─────────────────────────────────────────────────────
#  Resumen
# ─────────────────────────────────────────────────────
print(f"\n{'='*52}")
print(f"  Figuras guardadas en: ./{OUTPUT_DIR}/")
print(f"{'='*52}")
print("  fig1 — Convergencia R, T  vs Num_ord")
print("  fig2 — Tiempo           vs Num_ord")
print("  fig3 — Convergencia R, T  vs slices")
print("  fig4 — Tiempo           vs slices")
print("  fig5 — Convergencia relativa vs Num_ord")
print("  fig6 — Convergencia relativa vs slices")
print("  fig7 — Heatmap tiempo total  (2D)")
print("  fig8 — Heatmap R_TM / R_TE  (2D)\n")