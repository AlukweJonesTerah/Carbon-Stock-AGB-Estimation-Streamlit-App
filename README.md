# Carbon Stock & AGB Estimation — Streamlit App

A Streamlit port of the Google Earth Engine notebook for estimating
above-ground biomass (AGB) carbon stock across selected Kenyan counties,
using Sentinel-1/2, SRTM, WorldClim, PALSAR, soil, and canopy-height data
as predictors, with Random Forest, Gradient Tree Boosting, and SVM
regression models.

## What it does

- Lets you pick which Kenyan counties (ADM1) to include in the study area
- Builds the full predictor stack on Earth Engine (spectral indices, radar,
  terrain, climate, soil, canopy height, land-surface temperature)
- Samples training/testing points and trains RF, GTB, and SVM regressors
- Shows an interactive map where you can switch between model outputs
  (carbon stock or AGB) — this replaces the notebook's `ipywidgets` dropdown
- Compares RF vs. GTB with a difference map
- Computes RMSE / MAE / R² and actual-vs-predicted scatter plots on the
  held-out test set
- Computes per-county zonal statistics (mean/min/max/sum), downloadable as CSV
- Shows variable importance for RF and GTB

## 1. Prerequisites

- Python 3.10+
- A Google account with **Earth Engine access** enabled
  (sign up free at https://earthengine.google.com)
- A **Google Cloud project** registered for Earth Engine use
  (any project ID works, e.g. one created at https://console.cloud.google.com)

## 2. Install

```bash
pip install -r requirements.txt
```

## 3. Authenticate with Earth Engine (one-time, per machine)

Run this once in your terminal before launching the app:

```bash
earthengine authenticate
```

This opens a browser window, asks you to sign in with the Google account
that has Earth Engine access, and stores a local credentials token. After
this, `ee.Initialize(project=...)` will work without prompting again.

> **Note on hosted/headless deployments** (e.g. Streamlit Community Cloud):
> the interactive OAuth flow in `ee.Authenticate()` won't work in a headless
> environment. For those, use a **service account**:
> 1. Create a service account in Google Cloud Console and enable the Earth
>    Engine API for your project.
> 2. Download its JSON key.
> 3. Replace the `init_earth_engine()` call in `app.py` with:
>    ```python
>    credentials = ee.ServiceAccountCredentials(service_account_email, key_path)
>    ee.Initialize(credentials, project=project_id)
>    ```
> 4. Store the key contents in Streamlit secrets rather than committing the
>    file to source control.

## 4. Run

```bash
streamlit run app.py
```

Then in the sidebar:
1. Enter your GEE Cloud project ID and click **Initialize Earth Engine**.
2. Choose counties, sampling size, train/test split, and model hyperparameters.
3. Click **Run analysis**.

## Notes on performance

- All Earth Engine calls in this app are *lazy* (server-side) until you
  call `.getInfo()` or render a layer — building the predictor stack is fast,
  but sampling and training run actual computation on Google's servers and
  can take anywhere from several seconds to a couple of minutes depending on
  `num_pixels` and the size of your selected counties.
- Results are cached per parameter combination (`st.cache_resource`), so
  re-running with the same settings is instant; changing any slider or
  county selection triggers a recompute.
- The validation and zonal-statistics tabs are computed on demand (via
  buttons) since they trigger additional Earth Engine reductions.

## Differences from the original notebook

- The `ee.batch.Export.*` tasks (exporting training points, rasters, and
  zonal stats to Google Drive/Assets) were **not ported** — they're
  one-off batch jobs, not something a live app should kick off automatically.
  If you need those exports, run the original notebook's export cells, or
  add a button that calls `ee.batch.Export...` and `.start()` explicitly.
- The `ipywidgets` model-switcher dropdown became a `st.radio` control tied
  to a `geemap.foliumap` map rendered via `streamlit-folium`.
- Validation metrics and zonal stats are computed on demand via buttons,
  rather than automatically on every run, to avoid unnecessary Earth Engine
  calls.
