import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys


# OVERVIEW
# 3 sets of Visuals
#   1. Galactic plane H-R diagrams based on uncertainty (low, med, high)
#   2. Galactic plane vs nearby stars H-R diagrams
#   3. Pleiades 91 star dataset vs Pleiades 51 star dataset H-R diagrams

# Load data
nearby = pd.read_csv("project data\\Nearby_stars.csv")
gplane = pd.read_csv("project data\\Galactic_Plane_Stars.csv")
pleiades_91 = pd.read_csv("project data\\pleiades_91_members.csv")
pleiades_51 = pd.read_csv("project data\\pleiades_3sd_51_members.csv")

# generic function to prepare datasets for HR diagrams
def prep_data(df, name):
    df = df.copy()

    # Only keep rows needed for HR diagrams
    required_cols = ["parallax", "parallax_error", "phot_g_mean_mag", "bp_rp"]
    for col in required_cols:
        if col not in df.columns:
            print(f"dataset {name} is missing required column: {col}")
            sys.exit(1)
        
        df[col] = pd.to_numeric(df[col], errors="coerce")
    
    # Drop missing values
    df = df.dropna(subset = required_cols)

    # make sure all parallax values are positive
    df = df[df["parallax"] > 0].copy()

    # calculate relative parallax error and naive distance
    df["relative_parallax_error"] = df["parallax_error"] / df["parallax"]
    df["naive_distance_pc"] = 1000 / df["parallax"]

    # calculate absolute G magnitude
    df["M_G"] = df["phot_g_mean_mag"] - 5 * np.log10(df["naive_distance_pc"]) + 5

    return df

# prepare data
nearby = prep_data(nearby, "nearby")
gplane = prep_data(gplane, "galactic_plane")
pleiades_91 = prep_data(pleiades_91, "pleiades_91")
pleiades_51 = prep_data(pleiades_51, "pleiades_51")


# Create uncertainty bins for galactic plane
gp = gplane.copy()

gp["uncertainty_bin"] = pd.qcut(gp["relative_parallax_error"], q = 3, labels = ["Low", "Medium", "High"])

gp_low = gp[gp["uncertainty_bin"] == "Low"]
gp_med = gp[gp["uncertainty_bin"] == "Medium"]
gp_high = gp[gp["uncertainty_bin"] == "High"]

# print counts for each bin
print(gp["uncertainty_bin"].value_counts())

# Helper function for HR diagrams
def hr_dgrm(ax, df, title, s = 4, alpha = 0.5):
    ax.scatter(df["bp_rp"], df["M_G"], s = s, alpha = alpha)
    ax.set_xlabel("BP - RP (color)")
    ax.set_ylabel("Absolute G Magnitude (M_G)")
    ax.set_title(title)
    ax.grid(alpha = 0.2)
    ax.set_ylim(20, -10)


# VISUALS 1
# Galactic plane binning
fig, axes = plt.subplots(1, 3, figsize = (15, 5), sharey = True)

hr_dgrm(axes[0], gp_low, "Galactic Plane - Low Uncertainty")
hr_dgrm(axes[1], gp_med, "Galactic Plane - Medium Uncertainty")
hr_dgrm(axes[2], gp_high, "Galactic Plane - High Uncertainty")

plt.tight_layout()
plt.show()

# VISUALS 2
# Galactic plane vs Nearby stars
fig, axes = plt.subplots(1, 2, figsize = (10, 5), sharey = True)

hr_dgrm(axes[0], gplane, "Galactic Plane Stars")
hr_dgrm(axes[1], nearby, "Nearby Stars")

plt.tight_layout()
plt.show()

#### VISUAL 3 Showed bery little difference between the two plots so we won't use it.

# # VISUALS 3
# # Pleiades 91 star vs 51 star
# fig, axes = plt.subplots(1, 2, figsize = (10, 5), sharey = True)

# hr_dgrm(axes[0], pleiades_91, "Pleiades 91 Star Sample")
# hr_dgrm(axes[1], pleiades_51, "Pleiades 51 Star Sample")

# plt.tight_layout()
# plt.show()