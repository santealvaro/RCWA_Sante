import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np

# --- Load data ---
df = pd.read_csv("csv/results/1_inconel_temps_corregida_cortada.csv")

# Rename first column for convenience
df.rename(columns={df.columns[0]: "Lambda"}, inplace=True)

temp_columns = [c for c in df.columns if c != "Lambda"]

# --- Color map: cool blue -> hot red across temperatures ---
colors = cm.plasma(np.linspace(0.15, 0.9, len(temp_columns)))

# --- Plot ---
fig, ax = plt.subplots(figsize=(10, 6))

for col, color in zip(temp_columns, colors):
    ax.plot(df["Lambda"], df[col], label=col, color=color, linewidth=1.8)

ax.set_xlabel("Wavelength (µm)", fontsize=13)
ax.set_ylabel("Emissivity", fontsize=13)
ax.set_title("Inconel emissivity vs wavelength at different temperatures", fontsize=14)
ax.legend(title="Temperature", fontsize=10, title_fontsize=11, loc="best")
ax.grid(True, linestyle="--", alpha=0.4)
ax.set_ylim(bottom=0)

plt.tight_layout()
plt.savefig("emissivity_inconel.png", dpi=150)
plt.show()
print("Plot saved to emissivity_inconel.png")