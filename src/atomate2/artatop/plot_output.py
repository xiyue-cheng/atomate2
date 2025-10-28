import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Path to your out_nonlin directory
out_dir = Path("/home/asma/Work/jf_calculations/art_local/test/art")
data_file = out_dir / "d_energy.dat"

# Read the data file
df = pd.read_csv(data_file, delim_whitespace=True, header=None)

# Assign column names
df.columns = ["E", "Value"]

# Split into valence and conduction regions
df_vb = df[df["E"] <= 0]  # Valence band (E < 0)
df_cb = df[df["E"] > 0]   # Conduction band (E > 0)

# Plot
plt.figure(figsize=(8, 4))
plt.plot(df_vb["E"], df_vb["Value"], color="green", label="VB (E < 0)")
plt.plot(df_cb["E"], df_cb["Value"], color="#9932CC", label="CB (E > 0)")

# Add small, smooth scatter points
plt.scatter(df_vb["E"], df_vb["Value"], color="green", s=10, alpha=0.6)
plt.scatter(df_cb["E"], df_cb["Value"], color="#9932CC", s=10, alpha=0.6)

# Axis and labels
plt.axvline(0, color="red", linestyle=":", linewidth=1)
plt.xlabel(r"$E_B$ (eV)")
plt.ylabel(r"$\zeta(E_B)$ (pm/V)")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.3)
plt.tight_layout()
plt.show()
