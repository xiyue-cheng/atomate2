import pandas as pd
import matplotlib.pyplot as plt
from jobflow import SETTINGS

# Connect to your JobStore
store = SETTINGS.JOB_STORE
store.connect()

# Query the stored ARTATOP job
result = store.query_one({"name": "artatop"}, load=True)

# Extract the nonlinear data from the output
output_data = result["output"]["output_data"]
vb_data = output_data["nshgv"]  # valence band
cb_data = output_data["nshgc"]  # conduction band

# Convert to DataFrames
df_vb = pd.DataFrame(vb_data)
df_cb = pd.DataFrame(cb_data)

# Plot
plt.figure(figsize=(8, 4))
plt.plot(df_vb["E_repr"], df_vb["value"], color="green", label="VB")
plt.plot(df_cb["E_repr"], df_cb["value"], color="#9932CC", label="CB")

# Add scatter points for smoothness
plt.scatter(df_vb["E_repr"], df_vb["value"], color="green", s=10, alpha=0.6)
plt.scatter(df_cb["E_repr"], df_cb["value"], color="#9932CC", s=10, alpha=0.6)

# Axis and labels
plt.axvline(0, color="red", linestyle=":", linewidth=1)
plt.xlabel(r"$E_B$ (eV)")
plt.ylabel(r"$\zeta(E_B)$ (pm/V)")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.3)
plt.tight_layout()
plt.show()
