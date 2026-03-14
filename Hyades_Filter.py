import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load data
data_path = "project data\Hyades_Cluster.csv"
df = pd.read_csv(data_path)

print("Columns: \n", df.columns.tolist())
print("\nShape: ", df.shape)

# Double check columns
cols_needed = [
    "source_id", "ra", "dec",
    "parallax", "parallax_error", "parallax_over_error",
    "pmra", "pmra_error", "pmdec", "pmdec_error",
    "phot_g_mean_mag", "bp_rp"
]

df = df[[c for c in cols_needed if c in df.columns]].copy()

# convert columns to numeric
for col in df.columns:
    if col != "source_id":
        df[col] = pd.to_numeric(df[col], errors = "coerce")

# get rid of rows with missing data
df = df.dropna(subset = ["parallax", "pmra", "pmdec", "phot_g_mean_mag", "bp_rp"])

print("Shape after cleaning: ", df.shape)

# Broad plots
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

axes[0].scatter(df["pmra"], df["pmdec"], s=2, alpha=0.4)
axes[0].set_xlabel("pmra (mas/yr)")
axes[0].set_ylabel("pmdec (mas/yr)")
axes[0].set_title("Proper Motion: Full Cone Search")

axes[1].hist(df["parallax"], bins=100)
axes[1].set_xlabel("Parallax (mas)")
axes[1].set_ylabel("Count")
axes[1].set_title("Parallax Distribution")

axes[2].scatter(df["bp_rp"], df["phot_g_mean_mag"], s=2, alpha=0.4)
axes[2].invert_yaxis()
axes[2].set_xlabel("bp_rp")
axes[2].set_ylabel("G magnitude")
axes[2].set_title("Color-Magnitude Diagram: Full Sample")

plt.tight_layout()
plt.show()

# According to `https://sci.esa.int/web/gaia/-/60226-the-hyades-cluster`
# The Hyades star cluster is ~150 light years from Earth
# 150 light years = ~46 parsec
# 46 parsec = ~0.02174 arseconds = ~ 21.74 milliarcseconds parallax
# Further filter parallax values to be closer to true parallax for Hyades Cluster

filter_vals = df[
    (df["parallax"] > 19) & (df["parallax"] < 25)
].copy()

print("Filtered sample size: ", len(filter_vals))

# Plot filtered sample on top of full proper motion diagram
plt.figure(figsize=(7, 6))
plt.scatter(df["pmra"], df["pmdec"], s=2, alpha=0.2, label="All stars")
plt.scatter(filter_vals["pmra"], filter_vals["pmdec"], s=6, alpha=0.8, label="Filtered sample")
plt.xlabel("pmra (mas/yr)")
plt.ylabel("pmdec (mas/yr)")
plt.title("Filter Stars with Similar Parallax to Hyades")
plt.legend()
plt.show()

# Overlay shows a tight cluster between 60-140 pmra and 10-(-60) pmdec.
# Lets filter the original dataset with these additional constraints and recreate the plots

filter_vals = df[
    (df["parallax"] > 20) & (df["parallax"] < 23) &
    (df["pmra"] > 60) & (df["pmra"] < 140) &
    (df["pmdec"] > -60) & (df["pmdec"] < 10)
].copy()

# New plots
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

axes[0].scatter(filter_vals["pmra"], filter_vals["pmdec"], s=2, alpha=0.4)
axes[0].set_xlabel("pmra (mas/yr)")
axes[0].set_ylabel("pmdec (mas/yr)")
axes[0].set_title("Proper Motion")

axes[1].hist(filter_vals["parallax"], bins=100)
axes[1].set_xlabel("Parallax (mas)")
axes[1].set_ylabel("Count")
axes[1].set_title("Parallax Distribution")

axes[2].scatter(filter_vals["bp_rp"], filter_vals["phot_g_mean_mag"], s=2, alpha=0.4)
axes[2].invert_yaxis()
axes[2].set_xlabel("bp_rp")
axes[2].set_ylabel("G magnitude")
axes[2].set_title("Color-Magnitude Diagram: Full Sample")

plt.tight_layout()
plt.show()

# Further filter with sigma clipping in pmra, pmdec, and parallax

# use median and robust standard deviation from filtered values
def robust_sigma(x):
    mad = np.median(np.abs(x - np.median(x)))
    return 1.4826 * mad

pmra_med = filter_vals["pmra"].median()
pmdec_med = filter_vals["pmdec"].median()
parallax_med = filter_vals["parallax"].median()

pmra_sig = robust_sigma(filter_vals["pmra"])
pmdec_sig = robust_sigma(filter_vals["pmdec"])
parallax_sig = robust_sigma(filter_vals["parallax"])

print("\nFiltered medians:")
print("pmra   =", pmra_med)
print("pmdec  =", pmdec_med)
print("parallax =", parallax_med)

print("\nRobust sigmas:")
print("pmra sigma   =", pmra_sig)
print("pmdec sigma  =", pmdec_sig)
print("parallax sigma =", parallax_sig)


# cut off data that is more than 3 standard deviations away from medians
ref_data = filter_vals[
    (np.abs(filter_vals["pmra"] - pmra_med) < 3 * pmra_sig) &
    (np.abs(filter_vals["pmdec"] - pmdec_med) < 3 * pmdec_sig) &
    (np.abs(filter_vals["parallax"] - parallax_med) < 3 * parallax_sig) 
].copy()

print("Filtered dataset size: ", len(filter_vals))
print("Refined dataset size: ", len(ref_data))

# This time refined dataset and filtered dataset have approximately the same number of stars
# so we can just save the refined dataset
# It has 320 stars, which should be plenty for our analysis

ref_data.to_csv("project data\hyades_cluster_320_members.csv", index = False)

print("Files saved.")