# Gaia Parallax Uncertainty Dashboard

An interactive data dashboard built for a Data Science capstone project exploring how measurement uncertainty in the ESA Gaia DR3 dataset propagates into downstream astronomical analyses.

---

## Getting Started

All you need to do is clone the repo and run `run.py`:

```bash
git clone <repo-url>
cd <repo-folder>
python run.py
```

This will automatically install all required dependencies and launch the dashboard at **http://localhost:5000**.

---

## Project Data

The `project data` folder contains all datasets used in this project — no additional downloads are required. This includes:

- **Nearby_Stars.csv** — High quality Gaia DR3 stars with strong parallax measurements
- **Galactic_Plane_Stars.csv** — Stars along the galactic plane with high observational noise
- **pleiades_3sd_51_members.csv** — Filtered Pleiades cluster members (51 stars)
- **hyades_cluster_320_members.csv** — Filtered Hyades cluster members (320 stars)

The filtering scripts used to produce the cluster datasets from raw Gaia cone search results are also included (`Pleiades_Filter.py`, `Hyades_Filter.py`) for reference.

---

## Requirements

- Python 3.9+
- All Python dependencies are handled automatically by `run.py`