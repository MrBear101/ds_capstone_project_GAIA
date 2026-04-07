# This file contains code to use TaiPy to create the interactive dashboard

# Install TaiPy and Plotly if needed:
#       pip install taipy plotly pandas numpy

import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from taipy.gui import Gui, navigate
import os

# global variables
MAX_DATA_SIZE = 30_000

# dataset processing function (samples of data are randomly selected to avoid overloading dashboard with 100k samples)
def prep_data(df, group_name, sample_n = None):
    df = df.copy()
    df["group"] = group_name

    numeric_cols = [c for c in df.columns if c not in ("source_id", "group")]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors = "coerce")

    # ensure that all data in the required columns is not NA
    required = ["parallax", "parallax_error", "phot_g_mean_mag", "bp_rp"]
    df = df.dropna(subset = [c for c in required if c in df.columns])

    # drop negative parallax values
    df = df[df["parallax"] > 0].copy()

    df["relative_parallax_error"] = df["parallax_error"] / df["parallax"]
    df["naive_distance_pc"] = 1000.0 / df["parallax"]
    df["M_G"] = df["phot_g_mean_mag"] - 5 * np.log10(df["naive_distance_pc"]) + 5
    df["distance_residual"] = df["naive_distance_pc"] - df["naive_distance_pc"].median()
    
    # select random sample of data if theres too much
    if sample_n and len(df) > sample_n:
        df = df.sample(n=sample_n, random_state=42)
 
    return df

# Load + prep datasets using path join so it works on windows and linux
nearby      = pd.read_csv(os.path.join("project data", "Nearby_Stars.csv"))
gplane      = pd.read_csv(os.path.join("project data", "Galactic_Plane_Stars.csv"))
pleiades    = pd.read_csv(os.path.join("project data", "pleiades_3sd_51_members.csv"))
hyades      = pd.read_csv(os.path.join("project data", "hyades_cluster_320_members.csv"))

# Cap sample size to 30k for performance
nearby      = prep_data(nearby, "Nearby", MAX_DATA_SIZE)
gplane      = prep_data(gplane, "Galactic Plane", MAX_DATA_SIZE)
pleiades    = prep_data(pleiades, "Pleiades")
hyades      = prep_data(hyades, "Hyades")

# Create full datasets for certain metrics
nearby_full = pd.read_csv(os.path.join("project data", "Nearby_Stars.csv"))
nearby_full = prep_data(nearby_full, "Nearby")

gplane_full = pd.read_csv(os.path.join("project data", "Galactic_Plane_Stars.csv"))
gplane_full = prep_data(gplane_full, "Galactic Plane")

all_data = pd.concat([nearby, gplane, pleiades, hyades], ignore_index = True)
all_data_full = pd.concat([nearby_full, gplane_full, pleiades, hyades], ignore_index = True)

# Initial summary table
summary = all_data_full.groupby("group").agg(
    n_stars = ("source_id", "count"),
    median_parallax = ("parallax", "median"),
    median_parallax_error = ("parallax_error", "median"),
    median_relative_error = ("relative_parallax_error", "median"),
    median_distance = ("naive_distance_pc", "median"),
    std_distance = ("naive_distance_pc", "std"),
    iqr_distance=("naive_distance_pc", lambda x: x.quantile(0.75) - x.quantile(0.25))
).reset_index().round(4)

# colors and base layout

COLORS = {
    "Nearby":        "#4FC3F7",
    "Galactic Plane":"#FF8A65",
    "Pleiades":      "#81C784",
    "Hyades":        "#CE93D8",
}
 
BG      = "rgba(12,12,30,0.97)"
GRID    = "#1e2040"
ZERO    = "#555580"
TEXT    = "#D8D8F0"
 
def _base_layout(title="", height=500):
    return dict(
        title       = dict(text=title, font=dict(size=16, color=TEXT)),
        paper_bgcolor = "rgba(0,0,0,0)",
        plot_bgcolor  = BG,
        font          = dict(color=TEXT, family="'Segoe UI', Inter, sans-serif", size=12),
        height        = height,
        margin        = dict(l=65, r=30, t=55, b=65),
        legend        = dict(bgcolor="rgba(0,0,0,0)", bordercolor=GRID),
        xaxis         = dict(gridcolor=GRID, zerolinecolor=ZERO, linecolor=GRID),
        yaxis         = dict(gridcolor=GRID, zerolinecolor=ZERO, linecolor=GRID),
    )
 
# Overview charts

# overview histogram
def make_hist_fig(group = "All"):
    title = f"Naive Distance Distribution - {group}"

    # use full datasets for histograms
    df = all_data_full

    fig = go.Figure()
    groups_to_plot = [group]

    if group == "All":
        groups_to_plot = list(COLORS.keys())
    else:
        groups_to_plot = [group]
    
    for g in groups_to_plot:
        # get distance of stars in selected group
        subset = df[df["group"] == g]["naive_distance_pc"]
        # filter out massive outliers
        subset = subset[subset < 20_000]

        # create histograms
        fig.add_trace(go.Histogram(
            x = subset,
            name = g,
            marker = dict(color = COLORS[g]),
            opacity = 0.75,
            nbinsx = 60
        ))

    layout = _base_layout(title, height = 650)
    
    if group == "All":
        layout["barmode"] = "overlay"
    
    layout["xaxis"]["title"] = "Naive Distance (pc)"
    layout["yaxis"]["title"] = "Count"

    # ** operator unpacks dict in the format `key=value`
    fig.update_layout(**layout)
    return fig

# overview HR diagram
def make_hr_overview_fig(group = "All"):
    title = f"Color-Magnitude (H-R) Diagram - {group}"

    fig = go.Figure()
    groups_to_plot = [group]

    if group == "All":
        groups_to_plot = list(COLORS.keys())
    else:
        groups_to_plot = [group]

    for g in groups_to_plot:
        subset = all_data[all_data["group"] == g]
        # adjust marker size depending on how many points
        size = 6
        if len(subset) > 500:
            size = 3
        fig.add_trace(go.Scatter(
            x = subset["bp_rp"],
            y = subset["M_G"],
            mode = "markers",
            name = g,
            marker = dict(color = COLORS[g], size = size, opacity = 0.5)
        ))

    layout = _base_layout(title, height = 800)
    layout["xaxis"]["title"] = "Color Index (BP - RP)"
    layout["yaxis"]["title"] = "Absolute G Magnitude (M_G)"

    # reverse y axis (lower magnitude is brighter)
    layout["yaxis"]["autorange"] = "reversed"

    fig.update_layout(**layout)
    return fig


#  Uncertainty charts

# Boxplot figure
def make_boxplot_fig():
    fig = go.Figure()
    for g, color in COLORS.items():
        # select just relative error from all data full where the group is g
        subset = all_data_full[all_data_full["group"] == g]["relative_parallax_error"]
        # filter out top 1% extreme values for greater interpretability 
        p99 = subset.quantile(0.99)
        fig.add_trace(go.Box(
            y = subset[subset <= p99],
            name = g,
            fillcolor = color,
            boxmean = "sd",
            line = dict(width = 1.5)
        ))
    layout = _base_layout("Relative Parallax Error by Stellar Group", height = 650)
    layout["yaxis"]["title"] = "Relative Parallax Error (Error / Parallax)"

    fig.update_layout(**layout)
    return fig

# brightness vs relative error figure (filtering extreme errors)
def make_brightness_error_fig(max_relative_error = 2.0):
    df = all_data[all_data["relative_parallax_error"] <= max_relative_error]
    fig = go.Figure()

    for g, color in COLORS.items():
        subset = df[df["group"] == g]
        fig.add_trace(go.Scatter(
            x = subset["phot_g_mean_mag"],
            y = subset["relative_parallax_error"],
            mode = "markers",
            name = g,
            marker = dict(color = color, size = 3, opacity = 0.50)
        ))
    
    layout = _base_layout(f"Relative Parallax Error vs Apparent Brightness (Max relative error shown: {max_relative_error:.2f})", height = 700)
    layout["xaxis"]["title"] = "G Magnitude (lower is brighter)"
    layout["yaxis"]["title"] = "Relative Parallax Error"

    fig.update_layout(**layout)
    return fig
    

# RQ1 Charts - distance bias in the dataset

# cluster distance residual figure
def make_residual_fig(cluster = "Pleiades"):
    df = hyades

    if cluster == "Pleiades":
        df = pleiades
    
    color = COLORS[cluster]
    mean_dist = df["distance_residual"].mean()

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x = df["relative_parallax_error"],
        y = df["distance_residual"],
        mode = "markers",
        marker = dict(color = df["relative_parallax_error"],
                      colorscale = "Geyser",
                      size = 9,
                      opacity = 0.75,
                      colorbar = dict(title = "Rel. Error", thickness = 15),
                      showscale = True,
                    ),
        hovertemplate = ("Rel. Error:\t%{x:.4f}<br>", "Residual:\t%{y:.2f} pc<extra></extra>"),
        name = cluster
    ))

    # dashed line at zero
    fig.add_hline(
        y = 0,
        line_dash = "dash",
        line_color = "#FFFFFF",
        annotation_text = "Zero Residual"
    )

    # dotted line at mean residual
    fig.add_hline(
        y = mean_dist,
        line_dash = "dot",
        line_color = "#FFD700",
        annotation_text = f"Mean Residual: {mean_dist:.2f}"
    )

    layout = _base_layout(f"{cluster } Cluster - Distance Residual vs Relative Parallax Error", height = 700)
    layout["xaxis"]["title"] = "Relative Parallax Error (Error / Parallax)"
    layout["yaxis"]["title"] = "Distance Residual from Cluster Median (pc)"

    fig.update_layout(**layout)
    return fig

# RQ2 Charts

# 3 panel binned HR diagram
def make_hr_bins_fig():
    gp = gplane_full.copy()
    gp["uncertainty_bin"] = pd.qcut(gp["relative_parallax_error"], q=3, labels=["Low", "Medium", "High"])
    bin_colors = {"Low": "#81C784", "Medium": "#FFD54F", "High": "#E57373"}
    titles     = [
        "<br></br>Low Uncertainty<br><sub>(bottom third of rel. error)</sub>",
        "<br>Medium Uncertainty<br><sub>(middle third)</sub>",
        "<br>High Uncertainty<br><sub>(top third)</sub>",
    ]

    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=titles,
        shared_yaxes=True,
        horizontal_spacing=0.04,
    )
    for i, label in enumerate(["Low", "Medium", "High"]):
        sub = gp[gp["uncertainty_bin"] == label].sample(n=min(8_000, (gp["uncertainty_bin"]==label).sum()), random_state=42)
        fig.add_trace(
            go.Scatter(
                x      = sub["bp_rp"],
                y      = sub["M_G"],
                mode   = "markers",
                name   = label,
                marker = dict(color=bin_colors[label], size=3, opacity=0.40),
                showlegend = True,
            ),
            row=1, col=i+1,
        )
        fig.update_xaxes(
            title_text  = "BP − RP",
            gridcolor   = GRID, linecolor=GRID,
            row=1, col=i+1,
        )
        fig.update_yaxes(
            title_text  = "M_G" if i == 0 else "",
            autorange   = "reversed",
            gridcolor   = GRID, linecolor=GRID,
            row=1, col=i+1,
        )
 
    fig.update_layout(
        title       = dict(
            text="Galactic Plane H-R Diagrams - Binned by Relative Parallax Error<br><br>",
            font=dict(size=15, color=TEXT),
            pad=dict(b=500),
        ),
        paper_bgcolor = "rgba(0,0,0,0)",
        plot_bgcolor  = BG,
        font          = dict(color=TEXT, family="'Segoe UI', Inter, sans-serif", size=12),
        height        = 650,
        margin        = dict(l=65, r=30, t=80, b=65),
        legend        = dict(bgcolor="rgba(0,0,0,0)"),
    )
    return fig

# HR diagrams with a max relative error slider
def make_hr_filter_fig(dataset="Galactic Plane", max_relative_error=1.0):
    _map = {
        "Galactic Plane": gplane_full,
        "Nearby":         nearby_full,
        "Pleiades":       pleiades,
        "Hyades":         hyades,
    }
    df_full = _map[dataset]
    df      = df_full[df_full["relative_parallax_error"] <= max_relative_error].copy()
 
    # Sample for large datasets
    if len(df) > 50_000:
        df = df.sample(n=50_000, random_state=42)
 
    n_total = len(df_full)
    n_shown = len(df)
 
    fig = go.Figure()
    if len(df) == 0:
        fig.add_annotation(text="No stars pass this filter threshold.",
                           xref="paper", yref="paper", x=0.5, y=0.5,
                           showarrow=False, font=dict(size=16, color=TEXT))
    else:
        fig.add_trace(go.Scatter(
            x    = df["bp_rp"],
            y    = df["M_G"],
            mode = "markers",
            marker = dict(
                color      = df["relative_parallax_error"],
                colorscale = "Geyser",
                reversescale= True,
                size       = 4 if len(df) > 1000 else 7,
                opacity    = 0.55,
                colorbar   = dict(title="Rel. Error", thickness=14),
                showscale  = True,
            ),
            hovertemplate=(
                "BP−RP: %{x:.2f}<br>"
                "M_G:   %{y:.2f}<br>"
                "Rel. Err: %{marker.color:.4f}<extra></extra>"
            ),
            name = dataset,
        ))
 
    layout = _base_layout(
        f"H-R Diagram - {dataset}  "
        f"({n_shown:,} / {n_total:,} stars shown,  max rel. error ≤ {max_relative_error:.3f})",
        height=800,
    )
    layout["xaxis"]["title"] = "BP − RP  (color index)"
    layout["yaxis"]["title"] = "Absolute G Magnitude (M_G)"
    layout["yaxis"]["autorange"] = "reversed" 
    fig.update_layout(**layout)
    return fig


# Initial State


# Overview
groups_list             = ["All", "Nearby", "Galactic Plane", "Pleiades", "Hyades"]
selected_group          = "All"
hist_fig                = make_hist_fig("All")
hr_overview_fig         = make_hr_overview_fig("All")
 
# Uncertainty
boxplot_fig             = make_boxplot_fig()
max_relative_error_bright          = 2.0
bright_err_fig          = make_brightness_error_fig(max_relative_error_bright)
 
# RQ1
clusters_list           = ["Pleiades", "Hyades"]
selected_cluster        = "Pleiades"
residual_fig            = make_residual_fig("Pleiades")
 
# RQ2
rq2_datasets_list       = ["Galactic Plane", "Nearby", "Pleiades", "Hyades"]
selected_rq2_dataset    = "Galactic Plane"
max_relative_error_hr      = 0.50
hr_bins_fig             = make_hr_bins_fig()
hr_filter_fig           = make_hr_filter_fig("Galactic Plane", 0.50)
 

# Callbacks
  
def on_group_change(state):
    state.hist_fig        = make_hist_fig(state.selected_group)
    state.hr_overview_fig = make_hr_overview_fig(state.selected_group)
 
def on_max_relative_error_bright_change(state):
    state.bright_err_fig = make_brightness_error_fig(state.max_relative_error_bright)
 
def on_cluster_change(state):
    state.residual_fig = make_residual_fig(state.selected_cluster)
 
def on_rq2_dataset_change(state):
    state.hr_filter_fig = make_hr_filter_fig(state.selected_rq2_dataset, state.max_relative_error_hr)
 
def on_max_relative_error_hr_change(state):
    state.hr_filter_fig = make_hr_filter_fig(state.selected_rq2_dataset, state.max_relative_error_hr)

def go_overview(state):
    navigate(state, "Overview")

def go_uncertainty(state):
    navigate(state, "Uncertainty")

def go_rq1(state):
    navigate(state, "RQ1")
    
def go_rq2(state):
    navigate(state, "RQ2")

# Create TaiPy table for home page
home_table = pd.DataFrame({
    "Population":   ["Nearby Stars", "Galactic Plane Stars", "Pleiades Cluster", "Hyades Cluster"],
    "Description":  [
        "High quality parallax measurements",
        "Low galactic latitude (|b| < 5°), crowded and dusty",
        "Open cluster ~440 light years away, proper-motion filtered",
        "Open cluster ~150 light years away, proper-motion filtered",
    ],
    "N Stars": ["~100,000", "~100,000", "51", "320"],
})

# Create markdowns for each page so TaiPy can use them

root_md = """
<|navbar|>

<audio id="bg-music" loop controls style="
    position: fixed;
    bottom: 10px;
    right: 10px;
    opacity: 0.15;
">
    <source src="/assets/OutroM83.mp3" type="audio/mpeg"/>
</audio>
"""

home_md = """

<br/>
### Navigation

<|layout|columns=1 1 1 1|
<|Overview|button|on_action=go_overview|>
 
<|Uncertainty|button|on_action=go_uncertainty|>
 
<|RQ1 - Distance Bias|button|on_action=go_rq1|>
 
<|RQ2 - H-R Degradation|button|on_action=go_rq2|>
|>
<br/>

<|
# GAIA Parallax Uncertainty Dashboard

**Data Science Capstone Project**

---

## The Core Question

Gaia measures stellar distances through **parallax** - the apparent shift of a star's position as Earth orbits the Sun. Every measurement comes with an associated unceratinty. This project investigates how that uncertainty causes bias within derived quantities such as distance and absolute magnitude.

---

### Four Stellar Populations

<|{home_table}|table|show_all=True|>
---

Use the **tabs at the top** to navigate between sections.
|>
"""

overview_md = """
# Dataset Overview

Explore the distribution of distances and stellar classifications across the four populations.
Use the dropdown to focus on one group, or compare all at once.

<|{selected_group}|selector|lov={groups_list}|dropdown|label=Stellar Group|on_change=on_group_change|>

---

## Summary Statistics

<|{summary}|table|show_all=True|rebuild|>

---

## Distance Distribution

Histograms of the **naively** derived distance (d=1000 / parllax). Notw the wide spread of the Galactic Plane group compared to the tight clusters.

<|chart|figure={hist_fig}|>

---

## Color-Magnitude (H-R) Diagram

Stars colored by group. The nearby star sample reveals the clearest stellar sequences:
Main Sequence, Giant Branch, and White Dwarf Branch

<|chart|figure={hr_overview_fig}|>
"""

uncertainty_md = """
# Measurement Uncertainty

Parallax measurement uncertainty varies strongly with distance, brightness, and observational variables. The **relative parallax error** is the key metric. When this value is large, derived distances become unreliable.

---

## Relative Parllax Error by Group

Groups closer to Earth or with fewer observational hazards (dust, crowding, etc.) show substantially smaller relative errors. Whiskers extend to ±2 standard deviations.

<|chart|figure={boxplot_fig}|>

---

## Relative Error vs Apparent Brightness

Faitner stars (higher G-band magnitude) systematically have larger measurement uncertainty. This is a fundamental limit of GAIA instrumentation. Dimmer targets produce noisier data. The spread also widens at high magnitudes, indicating increased variability in precision for faint sources.

**Slide to set maximum relative error displayed:**
\n
<|{max_relative_error_bright}|slider|min=0.1|max=3.0|step=0.05|on_change=on_max_relative_error_bright_change|>
\n

<|chart|figure={bright_err_fig}|>
"""

rq1_md = """
# Research Question 1 - Distance Bias

**Research Question:** To what extent does distance bias vary across the Gaia dataset?

Since all stars in a cluster are physically close, they should all have approximately the same derived distance. Any spread around the cluster median is therefor likely a consequence of measurement uncertainty.

---

**Select Cluster:**
<|{selected_cluster}|selector|lov = {clusters_list}|on_change=on_cluster_change|>

- **White dashed line** - zero residual
- **Gold dotted line** - mean residual

Stars with hgiher relative error show greater spread above and blow zero. Implying that uncertainty translates into distance estimation error.

<|chart|figure={residual_fig}|>
"""

rq2_md = """
# Research Question 2 - Derived Visual Degradation

**Research Question:** Are there noticeable shifts or degradation in derived visuals as distance uncertainty increases?

Absolute magnitude (M_G) is calculated using distance, so parallax uncertainty propagates into the vertical axis of the H-R diagrams. As uncertainty inreases, segments of the diagram blur and dissapear.

---

## Galactic Plane - Binned by Uncertainty

The galactic plane dataset is split into equal-sized thirds based on relative parallax error. The progression from low to high uncertainty visibly degrades the main sequence band.

<|chart|figure={hr_bins_fig}|>

---

## Interactive Filter

Select a dataset and drag the slider to restrict whcih stars are shown. Lower thresholds reveal cleaner stellar structure. Stars are **colored by their individual relative error** (darker = higher error, lighter = lower).

**Dataset:**
<|{selected_rq2_dataset}|selector|lov={rq2_datasets_list}|dropdown|on_change=on_rq2_dataset_change|>

**Max Relative Parallax Error Shown:**
<|{max_relative_error_hr}|slider|min=0.005|max=2.0|step=0.005|on_change=on_max_relative_error_hr_change|>


<|chart|figure={hr_filter_fig}|>

"""


# PUT IT TOGETHER AND RUN

pages = {
    "/":            root_md,
    "Home":         home_md,
    "Overview":     overview_md,
    "Uncertainty":  uncertainty_md,
    "RQ1":          rq1_md,
    "RQ2":          rq2_md,
}
 
gui = Gui(pages=pages)
 
if __name__ == "__main__":
    gui.run(
        title           = "Gaia Parallax Uncertainty",
        host            = "0.0.0.0",
        port            = int(os.environ.get("PORT", 5000)),
        css_file        = "style.css",
        use_reloader    = False,
        debug           = False,
    )
 