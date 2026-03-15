import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load datasets
nearby = pd.read_csv("project data\\Nearby_stars.csv")
gplane = pd.read_csv("project data\\Galactic_Plane_Stars.csv")
pleiades = pd.read_csv("project data\\pleiades_3sd_51_members.csv")
hyades = pd.read_csv("project data\\hyades_cluster_320_members.csv")

# label groups and add derived variables
def prep_data(df, group_name):
    df = df.copy()
    df["group"] = group_name

    # double check that parallax values are positive
    df = df[df["parallax"] > 0].copy()

    df["relative_parallax_error"] = df["parallax_error"] / df["parallax"]
    df["naive_distance_pc"] = 1000 / df["parallax"]

    return df

nearby = prep_data(nearby, "Nearby")
gplane = prep_data(gplane, "Galactic Plane")
pleiades = prep_data(pleiades, "Pleiades")
hyades = prep_data(hyades, "Hyades")

all_data = pd.concat([nearby, gplane, pleiades, hyades], ignore_index = True)


# Initial summary table
summary = all_data.groupby("group").agg(
    n_stars = ("source_id", "count"),
    median_parallax = ("parallax", "median"),
    median_parallax_error = ("parallax_error", "median"),
    median_relative_error = ("relative_parallax_error", "median"),
    median_distance = ("naive_distance_pc", "median"),
    std_distance = ("naive_distance_pc", "std"),
    iqr_distance=("naive_distance_pc", lambda x: x.quantile(0.75) - x.quantile(0.25))
).reset_index()

print(summary)

# create boxplot for each group for relative parallax error
groups = ["Nearby", "Galactic Plane", "Pleiades", "Hyades"]
data_for_box = [all_data[all_data["group"] == g]["relative_parallax_error"] for g in groups]

plt.figure(figsize=(8, 6))
plt.boxplot(data_for_box, tick_labels=groups, showfliers=False)
plt.ylabel("Relative Parallax Error")
plt.title("Relative Parallax Error by Group")
plt.show()

# Create histograms of naive distances by group
fig, axes = plt.subplots(2, 2, figsize = (12,10))

for ax, group in zip(axes.flatten(), groups):
    subset = all_data[all_data["group"] == group]
    # Filter out extremely large distance for galactic plane data
    subset_dist = subset["naive_distance_pc"]
    subset_dist = [dist for dist in subset_dist if dist < 20000]
    ax.hist(subset_dist, bins = 50)
    ax.set_title(group)
    ax.set_xlabel("Naive Distance (pc)")
    ax.set_ylabel("Count")

plt.tight_layout()
plt.show()

# Plot comparing difference between naive distances  and median distances in clusters with relative parallax error
# if larger parallax error produces larger distance difference, that points towards uncertainty causing distortion

for cluster_name, cluster_df in [("Pleiades", pleiades), ("Hyades", hyades)]:
    cluster_df["distance_residual"] = cluster_df["naive_distance_pc"] - cluster_df["naive_distance_pc"].median()

    plt.figure(figsize = (12, 10))
    plt.scatter(cluster_df["relative_parallax_error"], cluster_df["distance_residual"], alpha = 0.7)
    plt.axhline(0, color = "black", linestyle = "--")
    plt.xlabel("Relative Parallax Error")
    plt.ylabel("Distance Residual from Cluster Median (pc)")
    plt.title(f"{cluster_name}: Distance Residual vs Relative Parallax Error")
    plt.show()


# Plot magnitude vs relative error
# If there is a strong corrletaion,  shows error is systematic not random

plt.figure(figsize = (12, 10))

for group in groups:
    subset = all_data[all_data["group"] == group]
    plt.scatter(subset["phot_g_mean_mag"], subset["relative_parallax_error"], s = 8, alpha = 0.4, label = group)

plt.xlabel("G Magnitude (Lower is Brighter)")
plt.ylabel("Relative Parallax Error")
plt.title("Relative Parallax Error vs Apparent Brightness")
plt.legend()
plt.show()