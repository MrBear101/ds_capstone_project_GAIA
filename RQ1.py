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