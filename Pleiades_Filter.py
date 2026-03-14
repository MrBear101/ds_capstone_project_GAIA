import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load data
data_path = "project data\Pleiades_Cluster.csv"
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
axes[0].set_xlim(-50, 50)
axes[0].set_ylim(-100, 100)
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

# filter based on plots
# According to `https://science.nasa.gov/missions/hubble/hubble-refines-distance-to-pleiades-star-cluster/`
# The Pleiades Star Cluster is approximately 440 light years from Earth
# 440 light years = ~135 parsec
# 135 parsec = ~0.00741 arcseconds = ~7.4 milliarcseconds parallax
# In the Parallax distribution plot we see a slight bump around 7-7.5 mas, so we can focus our filtered values in that region

filter_vals = df[
    (df["parallax"] > 6) & (df["parallax"] < 9)
].copy()

print("Filtered sample size: ", len(filter_vals))

# Plot filtered sample on top of full proper motion diagram
plt.figure(figsize=(7, 6))
plt.scatter(df["pmra"], df["pmdec"], s=2, alpha=0.2, label="All stars")
plt.scatter(filter_vals["pmra"], filter_vals["pmdec"], s=6, alpha=0.8, label="Filtered sample")
plt.xlabel("pmra (mas/yr)")
plt.ylabel("pmdec (mas/yr)")
plt.title("Filter Stars with Similar Parallax to Pleiades")
plt.legend()
plt.show()

# Overlaying stars that have a similar parallax (distance) to Pleiades with all stars in our search returns a wide cluster of stars
# But there is a very tight cluster located between 15 and 25 pmra, and -37 and -53 pmdec
# We can now filter the original dataset with these additional constraints and recreate the plots
filter_vals = df[
    (df["parallax"] > 6) & (df["parallax"] < 9) &
    (df["pmra"] > 15) & (df["pmra"] < 25) &
    (df["pmdec"] > -53) & (df["pmdec"] < -37)
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
fig.suptitle("New Plots with Tight Filters")
fig.subplots_adjust(top = 0.88)
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
ref_data = df[
    (np.abs(df["pmra"] - pmra_med) < 3 * pmra_sig) &
    (np.abs(df["pmdec"] - pmdec_med) < 3 * pmdec_sig) &
    (np.abs(df["parallax"] - parallax_med) < 3 * parallax_sig) 
].copy()

print("Refined dataset size: ", len(ref_data))

# Save dataset after filter and after sigma clipping
# This produces 2 Pleiades datasets, one with 91 stars, and one with 51 stars
# 91 star dataset can be used for RQ3 (machine learning)
# Both datasets can be compared for RQ2 (visual distortions with uncertainty)
# 51 star set can be used for RQ1 (how does distance bias vary)
# ALl of this is a maybe, we might just want to use the 51 star high confidence dataset

filter_vals.to_csv("project data\pleiades_91_members.csv", index = False)

ref_data.to_csv("project data\pleiades_3sd_51_members.csv", index = False)

print("Files saved.")