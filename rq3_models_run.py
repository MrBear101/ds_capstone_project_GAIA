# This script will run the models for RQ3.
# It needs to be run ONCE before launching the dashboard
# Results are saved as CSV files into the "project data folder"
# Ideally the results will be in the github so nobody should need to run this

import os
import pandas as pd
import numpy as np
from sklearn.linear_model import LassoCV
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
 
DATA = "project data"
 
# ── Load datasets ────────────────────────────────────────────────────────
 
nearby      = pd.read_csv(os.path.join(DATA, "nearby stars extra attributes.csv")).dropna()
gplane      = pd.read_csv(os.path.join(DATA, "galactic plane stars extra attributes.csv")).dropna()
pleiades    = pd.read_csv(os.path.join(DATA, "pleiades_extra_attributes.csv")).dropna()
hyades      = pd.read_csv(os.path.join(DATA, "hyades_extra_attributes.csv")).dropna()
 
# ── Feature columns ──────────────────────────────────────────────────────
# These are the extra-attribute columns available in all four datasets.
# l and b are only in nearby and galactic plane so they are handled per-dataset.
 
BASE_FEATURES = [
    "ra", "dec", "pm", "pmra", "pmdec", "phot_g_mean_mag", "bp_rp",
    "astrometric_excess_noise", "astrometric_excess_noise_sig",
    "visibility_periods_used", "ruwe",
]
 
FEATURES = {
    "Nearby":         BASE_FEATURES + ["l", "b"],
    "Galactic Plane": BASE_FEATURES + ["l", "b"],
    "Pleiades":       BASE_FEATURES,
    "Hyades":         BASE_FEATURES,
}
 
DATASETS = {
    "Nearby":         nearby,
    "Galactic Plane": gplane,
    "Pleiades":       pleiades,
    "Hyades":         hyades,
}
 
TARGET = "parallax_error"
 
# ── Model functions ───────────────────────────────────────────────────────
 
def get_xy(df, feature_cols):
    cols = [c for c in feature_cols if c in df.columns]
    d    = df[cols + [TARGET]].dropna().copy()
    # Cap large datasets to 20k rows for reasonable training time
    if len(d) > 20_000:
        d = d.sample(n=20_000, random_state=42)
    X = d[cols]
    y = d[TARGET]
    return X, y
 
def run_lasso(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    scaler     = StandardScaler()
    X_train_s  = scaler.fit_transform(X_train)
    X_test_s   = scaler.transform(X_test)
 
    model = LassoCV(cv=5, n_alphas=100, max_iter=3000, random_state=42)
    model.fit(X_train_s, y_train)
    y_pred = model.predict(X_test_s)
 
    r2   = round(r2_score(y_test, y_pred), 4)
    rmse = round(np.sqrt(mean_squared_error(y_test, y_pred)), 6)
 
    importance = pd.DataFrame({
        "feature":    X.columns.tolist(),
        "importance": np.abs(model.coef_),
    })
    return importance, r2, rmse
 
def run_gbr(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    scaler    = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)
 
    model = GradientBoostingRegressor(
        n_estimators=200, learning_rate=0.05,
        max_depth=3, subsample=0.8, random_state=42
    )
    model.fit(X_train_s, y_train)
    y_pred = model.predict(X_test_s)
 
    r2   = round(r2_score(y_test, y_pred), 4)
    rmse = round(np.sqrt(mean_squared_error(y_test, y_pred)), 6)
 
    importance = pd.DataFrame({
        "feature":    X.columns.tolist(),
        "importance": model.feature_importances_,
    })
    return importance, r2, rmse
 
# ── Run all models ────────────────────────────────────────────────────────
 
all_importances = []
all_metrics     = []
 
for group, df in DATASETS.items():
    feats    = FEATURES[group]
    X, y     = get_xy(df, feats)
 
    print(f"\nTraining {group} ({len(X)} rows, {len(X.columns)} features)...")
 
    print(f"  Running Lasso...")
    lasso_imp, lasso_r2, lasso_rmse = run_lasso(X, y)
    lasso_imp["group"]  = group
    lasso_imp["model"]  = "Lasso"
    all_importances.append(lasso_imp)
    all_metrics.append({"Group": group, "Model": "Lasso", "R²": lasso_r2, "RMSE": lasso_rmse})
    print(f"  Lasso done — R²={lasso_r2}, RMSE={lasso_rmse}")
 
    print(f"  Running GBR...")
    gbr_imp, gbr_r2, gbr_rmse = run_gbr(X, y)
    gbr_imp["group"] = group
    gbr_imp["model"] = "GBR"
    all_importances.append(gbr_imp)
    all_metrics.append({"Group": group, "Model": "GBR", "R²": gbr_r2, "RMSE": gbr_rmse})
    print(f"  GBR done — R²={gbr_r2}, RMSE={gbr_rmse}")
 
# ── Save results ──────────────────────────────────────────────────────────
 
importances_df = pd.concat(all_importances, ignore_index=True)
metrics_df     = pd.DataFrame(all_metrics)
 
importances_df.to_csv(os.path.join(DATA, "rq3_importances.csv"), index=False)
metrics_df.to_csv(os.path.join(DATA, "rq3_metrics.csv"),         index=False)
 
print("\nDone. Files saved:")
print(f"  {os.path.join(DATA, 'rq3_importances.csv')}")
print(f"  {os.path.join(DATA, 'rq3_metrics.csv')}")