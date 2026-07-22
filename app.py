"""
Carbon Stock & Above-Ground Biomass (AGB) Estimation — Streamlit App
======================================================================
A Streamlit port of the Google Earth Engine notebook that estimates
above-ground biomass carbon stock across selected Kenyan counties using
Sentinel-1/2, SRTM, WorldClim, PALSAR, soil, and canopy-height predictors,
trained with Random Forest, Gradient Tree Boosting, and SVM regressors.

Run with:
    streamlit run app.py

Requires a Google Earth Engine account + a registered Cloud project.
See the "Setup" section in the sidebar / README for authentication steps.
"""

import os
import json
import time
import io
import base64
import urllib.error
import urllib.request
os.environ.setdefault("USE_FOLIUM", "1")  # ensure geemap.__init__ uses folium branch

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.backends.backend_pdf import PdfPages

import ee
import geemap.foliumap as geemap  # folium-backed geemap -> works in Streamlit
from streamlit_folium import st_folium

# ----------------------------------------------------------------------------
# PAGE CONFIG
# ----------------------------------------------------------------------------
st.set_page_config(
    page_title="Carbon Stock & AGB Estimation",
    page_icon="🌳",
    layout="wide",
)

# ----------------------------------------------------------------------------
# THEME / CSS
# ----------------------------------------------------------------------------
st.markdown("""
<style>
@import url('https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.5.1/css/all.min.css');
/* ── Global tokens ───────────────────────────────────────── */
:root {
    --green-dark:   #1a472a;
    --green-mid:    #2d6a4f;
    --green-light:  #52b788;
    --green-pale:   #eaf6ec;
    --surface:      #ffffff;
    --text-primary: #1c2826;
    --text-muted:   #5a6e63;
    --border:       #c8e6cc;
    --shadow:       rgba(26,71,42,0.10);
}

/* ── Hide default Streamlit chrome ───────────────────────── */
#MainMenu, footer { visibility: hidden; }

/* ── App background ──────────────────────────────────────── */
.stApp { background: #f4faf5; }

/* ── Sidebar ─────────────────────────────────────────────── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #1a472a 0%, #2d6a4f 55%, #1b5e35 100%);
}
[data-testid="stSidebar"] section { padding-top: 0 !important; }

/* All sidebar text white */
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] span,
[data-testid="stSidebar"] div { color: #dff0e3 !important; }

[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 { color: #ffffff !important; }

/* Sidebar inputs */
[data-testid="stSidebar"] input {
    background: rgba(255,255,255,0.12) !important;
    color: #ffffff !important;
    border: 1px solid rgba(255,255,255,0.25) !important;
    border-radius: 8px !important;
}
[data-testid="stSidebar"] .stSelectbox > div,
[data-testid="stSidebar"] .stMultiSelect > div {
    background: rgba(255,255,255,0.12) !important;
    border: 1px solid rgba(255,255,255,0.25) !important;
    border-radius: 8px !important;
}

/* Sidebar dividers */
[data-testid="stSidebar"] hr {
    border: none;
    border-top: 1px solid rgba(255,255,255,0.18) !important;
    margin: 0.75rem 0;
}

/* Sidebar primary button */
[data-testid="stSidebar"] [data-testid="baseButton-primary"] {
    background: linear-gradient(135deg, #52b788, #40916c) !important;
    border: none !important;
    border-radius: 10px !important;
    font-weight: 700 !important;
    letter-spacing: 0.3px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.3) !important;
}
[data-testid="stSidebar"] [data-testid="baseButton-secondary"] {
    background: rgba(255,255,255,0.12) !important;
    border: 1px solid rgba(255,255,255,0.30) !important;
    color: #fff !important;
    border-radius: 10px !important;
}

/* ── Hero banner ─────────────────────────────────────────── */
.hero {
    background: linear-gradient(135deg, #1a472a 0%, #2d6a4f 50%, #40916c 100%);
    border-radius: 18px;
    padding: 2.25rem 2.5rem 2rem;
    margin-bottom: 1.5rem;
    position: relative;
    overflow: hidden;
}
.hero::after {
    font-family: "Font Awesome 6 Free";
    font-weight: 900;
    content: "\\f1bb";
    position: absolute;
    right: 2.5rem; top: 1.25rem;
    font-size: 7rem;
    opacity: 0.13;
    line-height: 1;
}
.hero h1 {
    font-size: 1.9rem;
    font-weight: 800;
    color: #ffffff !important;
    margin: 0 0 0.4rem;
    line-height: 1.2;
}
.hero p {
    color: rgba(255,255,255,0.78);
    font-size: 0.97rem;
    max-width: 680px;
    margin: 0 0 1rem;
}
.hero-tags { display: flex; flex-wrap: wrap; gap: 0.45rem; }
.hero-tag {
    background: rgba(255,255,255,0.15);
    border: 1px solid rgba(255,255,255,0.28);
    border-radius: 20px;
    padding: 0.18rem 0.7rem;
    font-size: 0.76rem;
    color: rgba(255,255,255,0.88);
    font-weight: 500;
}

/* ── Metric cards ────────────────────────────────────────── */
[data-testid="stMetric"] {
    background: #ffffff;
    border: 1px solid var(--border);
    border-radius: 14px;
    padding: 1.1rem 1.4rem !important;
    box-shadow: 0 2px 8px var(--shadow);
}
[data-testid="stMetricLabel"] p {
    color: var(--text-muted) !important;
    font-size: 0.80rem !important;
    font-weight: 600 !important;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}
[data-testid="stMetricValue"] {
    color: var(--green-dark) !important;
    font-size: 1.9rem !important;
    font-weight: 800 !important;
}

/* ── Tabs ────────────────────────────────────────────────── */
[data-testid="stTabs"] [data-baseweb="tab-list"],
[data-testid="stTabs"] [role="tablist"] {
    gap: 4px;
    background: #dceee0;
    border-radius: 12px;
    padding: 5px;
    border: 1px solid #b7dbbe;
}
[data-testid="stTabs"] [data-baseweb="tab"],
[data-testid="stTabs"] button[role="tab"] {
    border-radius: 9px;
    padding: 0.45rem 1.1rem;
    font-weight: 600;
    font-size: 0.88rem;
    color: #274b39 !important;
    background: transparent !important;
    border: none !important;
    transition: all 0.15s ease;
}
[data-testid="stTabs"] [data-baseweb="tab"] *,
[data-testid="stTabs"] button[role="tab"] * {
    color: #274b39 !important;
    -webkit-text-fill-color: #274b39 !important;
    opacity: 1 !important;
}
[data-testid="stTabs"] [data-baseweb="tab"]:hover,
[data-testid="stTabs"] button[role="tab"]:hover {
    background: #c6e5cb !important;
    color: #123621 !important;
}
[data-testid="stTabs"] [aria-selected="true"] {
    background: #ffffff !important;
    color: var(--green-dark) !important;
    box-shadow: 0 1px 5px var(--shadow) !important;
}
[data-testid="stTabs"] [aria-selected="true"] *,
[data-testid="stTabs"] [aria-selected="true"] p {
    color: #1a472a !important;
    -webkit-text-fill-color: #1a472a !important;
    font-weight: 700 !important;
}
[data-testid="stTabs"] [data-baseweb="tab-panel"] {
    padding-top: 1.25rem;
}

/* ── Section headers ─────────────────────────────────────── */
.sec-hdr {
    font-size: 1.05rem;
    font-weight: 700;
    color: var(--green-dark);
    display: flex;
    align-items: center;
    gap: 0.5rem;
    padding-bottom: 0.5rem;
    border-bottom: 2px solid var(--border);
    margin-bottom: 1rem;
}
.sec-hdr i {
    color: var(--green-light);
    font-size: 1rem;
    width: 1.1rem;
    text-align: center;
}

/* ── Cards / panels ──────────────────────────────────────── */
.info-card {
    background: #ffffff;
    border: 1px solid var(--border);
    border-radius: 14px;
    padding: 1.4rem 1.6rem;
    box-shadow: 0 2px 8px var(--shadow);
    margin-bottom: 1rem;
}

/* ── Welcome / empty state ───────────────────────────────── */
.welcome {
    background: linear-gradient(135deg, #eaf6ec, #f4faf5);
    border: 2px dashed #90c49a;
    border-radius: 18px;
    padding: 3.5rem 2rem;
    text-align: center;
}
.welcome .icon { font-size: 3rem; display: block; margin-bottom: 0.75rem; color: var(--green-light); }
.welcome h3 { color: var(--green-dark); font-size: 1.3rem; margin: 0 0 0.5rem; }
.welcome p  { color: var(--text-muted); max-width: 440px; margin: 0 auto; }

/* ── Footer ──────────────────────────────────────────────── */
.app-footer {
    background: #dceee0;
    border-radius: 12px;
    padding: 0.8rem 1.25rem;
    font-size: 0.76rem;
    color: var(--text-muted);
    margin-top: 2rem;
    line-height: 1.6;
}

/* ── Dataframe ───────────────────────────────────────────── */
[data-testid="stDataFrame"] { border-radius: 10px; overflow: hidden; }

/* â”€â”€ Learning Guide chat â”€â”€ */
[data-testid="stChatMessage"] {
    background: #ffffff !important;
    border: 1px solid #c8e6cc;
    border-radius: 14px;
    padding: 0.85rem 1rem;
    margin-bottom: 0.7rem;
}
[data-testid="stChatMessage"] p,
[data-testid="stChatMessage"] li,
[data-testid="stChatMessage"] span,
[data-testid="stChatMessage"] div,
[data-testid="stChatMessage"] strong,
[data-testid="stChatMessage"] code {
    color: #1c2826 !important;
}
[data-testid="stChatInput"] textarea,
[data-testid="stChatInput"] textarea::placeholder {
    color: #1c2826 !important;
    background: #ffffff !important;
}
.guide-question-card {
    background: #ffffff;
    border: 1px solid #c8e6cc;
    border-radius: 14px;
    padding: 1rem 1.1rem 0.35rem;
    margin-top: 1rem;
}
[data-testid="stTextArea"] textarea {
    color: #1c2826 !important;
    background: #ffffff !important;
    border: 1px solid #9bcba4 !important;
    border-radius: 10px !important;
}
[data-testid="stTextArea"] textarea::placeholder { color: #62776c !important; }
[data-testid="stTextArea"] textarea:disabled {
    color: #1c2826 !important;
    -webkit-text-fill-color: #1c2826 !important;
    opacity: 1 !important;
    background: #ffffff !important;
}
.validation-guide {
    background: #ffffff;
    border: 1px solid #90c49a;
    border-left: 5px solid #2d6a4f;
    border-radius: 12px;
    padding: 1rem 1.2rem;
    color: #1c2826 !important;
    margin-bottom: 1rem;
}
.validation-guide, .validation-guide * { color: #1c2826 !important; }
.validation-guide strong { color: #1a472a !important; }

/* ── Expander ────────────────────────────────────────────── */
[data-testid="stSidebar"] [data-testid="stExpander"] {
    background: rgba(255,255,255,0.08) !important;
    border: 1px solid rgba(255,255,255,0.15) !important;
    border-radius: 10px !important;
}

/* ── Warning / info banner ───────────────────────────────── */
[data-testid="stAlert"] {
    border-radius: 10px !important;
}

/* ── Main-content widget label visibility ────────────────── */
[data-testid="stMain"] label,
[data-testid="stRadio"] label,
[data-testid="stCheckbox"] label,
[data-testid="stSelectbox"] label,
[data-testid="stSlider"] label,
[data-testid="stTextInput"] label,
[data-testid="stNumberInput"] label,
[data-testid="stMultiSelect"] label,
[data-testid="stRadio"] p,
[data-testid="stCheckbox"] p {
    color: var(--text-primary) !important;
}
[data-testid="stRadio"] div[role="radiogroup"] label span,
[data-testid="stCheckbox"] label span {
    color: var(--text-primary) !important;
}
</style>
""", unsafe_allow_html=True)

# ----------------------------------------------------------------------------
# CONSTANTS
# ----------------------------------------------------------------------------
CARBON_TO_BIOMASS_FACTOR = 0.47

VIS_PARAMS_CARBON = {
    "min": 0,
    "max": 200,
    "palette": ["#d7191c", "#fdae61", "#ffffbf", "#a6d96a", "#1a9641"],
}
VIS_PARAMS_BIOMASS = {
    "min": 0,
    "max": 425,
    "palette": ["#fde725", "#90d743", "#35b779", "#21918c", "#31688e", "#443983", "#440154"],
}
VIS_PARAMS_DIFF = {
    "min": 0,
    "max": 50,
    "palette": ["000080", "008080", "FFFF00", "FF0000"],
}

ALL_KENYA_COUNTIES_OPTIONS = [
    "Baringo", "Bomet", "Bungoma", "Busia", "Embu",
    "Garissa", "Homa Bay", "Isiolo", "Kajiado", "Kakamega", "Kericho",
    "Kiambu", "Kilifi", "Kirinyaga", "Kisii", "Kisumu", "Kitui", "Kwale",
    "Meru", "Migori", "Mombasa", "Murang'a", "Nakuru", "Nyandarua", "Nyamira", "Nyeri", "Samburu", "Siaya",
]

DATE_START_S2    = "2015-01-01"
DATE_END_S2      = "2024-01-01"  # inclusive coverage through 2023
AGB_YEARS_AVAIL  = ["2010", "2015", "2016", "2017", "2018", "2019", "2020", "2021", "2022"]
ESA_AGB_COLLECTION = "ESA/CCI/Above_Ground_Biomass/V6_0"
ESA_AGB_BAND       = "agb"

ANALYSIS_PRESETS = {
    "Quick preview": {"num_pixels": 1000, "rf_trees": 60, "gtb_trees": 60},
    "Balanced": {"num_pixels": 3000, "rf_trees": 100, "gtb_trees": 100},
    "High accuracy": {"num_pixels": 6000, "rf_trees": 250, "gtb_trees": 250},
}

VIS_PARAMS_SPREAD = {
    "min": 0,
    "max": 30,
    "palette": ["#f7fbff", "#9ecae1", "#2171b5", "#08306b"],
}

# ----------------------------------------------------------------------------
# MATPLOTLIB THEME
# ----------------------------------------------------------------------------
mpl.rcParams.update({
    "figure.facecolor": "#ffffff",
    "axes.facecolor":   "#f4faf5",
    "axes.edgecolor":   "#c8e6cc",
    "axes.grid":        True,
    "grid.color":       "#d0e8d4",
    "grid.linewidth":   0.8,
    "font.family":      "sans-serif",
    "font.size":        10,
    "axes.titlesize":   11,
    "axes.titleweight": "bold",
    "axes.titlecolor":  "#1a472a",
    "axes.labelcolor":  "#2d6a4f",
    "xtick.color":      "#5a6e63",
    "ytick.color":      "#5a6e63",
})

# ----------------------------------------------------------------------------
# EARTH ENGINE INITIALIZATION
# ----------------------------------------------------------------------------
def _ee_credentials_from_file():
    """Load OAuth2 credentials from the earthengine credentials file (no gcloud needed)."""
    import pathlib, json as _json
    cred_file = pathlib.Path.home() / ".config" / "earthengine" / "credentials"
    if not cred_file.exists():
        return None
    try:
        import google.oauth2.credentials
        data = _json.loads(cred_file.read_text())
        return google.oauth2.credentials.Credentials(
            token=None,
            refresh_token=data["refresh_token"],
            token_uri="https://oauth2.googleapis.com/token",
            client_id=data["client_id"],
            client_secret=data["client_secret"],
        )
    except Exception:
        return None


def _is_cloud_deployment() -> bool:
    """Return True when running on Streamlit Cloud or any headless server."""
    cloud_signals = [
        os.environ.get("STREAMLIT_SHARING_MODE"),
        os.environ.get("IS_CLOUD_ENV"),
        os.environ.get("STREAMLIT_SERVER_HEADLESS"),
    ]
    return any(v for v in cloud_signals)


_STREAMLIT_CLOUD_SETUP = """
**Production deployment detected — service account credentials required.**

Add the following to your app's **Streamlit Cloud → Settings → Secrets**:

```toml
[gee]
credentials = '''
{
  "type": "service_account",
  "project_id": "your-project-id",
  "private_key_id": "...",
  "private_key": "-----BEGIN RSA PRIVATE KEY-----\\n...\\n-----END RSA PRIVATE KEY-----\\n",
  "client_email": "your-sa@your-project.iam.gserviceaccount.com",
  ...
}
'''
```

Steps:
1. Go to [Google Cloud Console](https://console.cloud.google.com/) → IAM & Admin → Service Accounts
2. Create a service account and grant it the **Earth Engine Resource Viewer** role
3. Create a JSON key and paste the full JSON as the `credentials` value above
4. Redeploy the app
"""


def init_earth_engine(project_id: str) -> tuple[bool, str]:
    import json as _json

    # 1. Service-account path — Streamlit Cloud deployment via st.secrets
    _sa_creds = st.secrets.get("gee", {}).get("credentials", "").strip()
    if _sa_creds:
        try:
            cred_dict = _json.loads(_sa_creds)
            credentials = ee.ServiceAccountCredentials(
                email=cred_dict["client_email"],
                key_data=_sa_creds,
            )
            ee.Initialize(credentials=credentials, project=project_id)
            return True, "Earth Engine initialized via service account."
        except Exception as e:
            return False, f"Service account auth failed: {e}"

    # On Streamlit Cloud there is no home directory with credentials and no
    # browser to complete OAuth — skip straight to a clear setup message.
    if _is_cloud_deployment():
        return False, _STREAMLIT_CLOUD_SETUP

    # 2. Explicit credentials file — written by `earthengine authenticate` (no gcloud needed)
    local_creds = _ee_credentials_from_file()
    if local_creds is not None:
        try:
            ee.Initialize(credentials=local_creds, project=project_id)
            return True, "Earth Engine initialized."
        except Exception:
            pass  # fall through to browser auth

    # 3. Browser OAuth — only attempted on local machines
    try:
        ee.Authenticate(auth_mode="localhost", force=False)
        local_creds = _ee_credentials_from_file()
        if local_creds is not None:
            ee.Initialize(credentials=local_creds, project=project_id)
            return True, "Earth Engine authenticated and initialized."
        ee.Initialize(project=project_id)
        return True, "Earth Engine authenticated and initialized."
    except OSError as e:
        if getattr(e, "errno", None) == 98 or "Address already in use" in str(e):
            return False, (
                "Could not start the local OAuth server (port already in use). "
                "Run `earthengine authenticate --auth_mode notebook` in a terminal, "
                "then restart the app."
            )
        return False, f"Authentication failed: {e}"
    except Exception as e:
        return False, (
            f"Authentication failed ({e}). "
            "Run `earthengine authenticate` in a terminal and restart the app."
        )


# ----------------------------------------------------------------------------
# DATA / FEATURE ENGINEERING
# ----------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def get_study_area(_project_id: str, county_selection: tuple[str, ...]):
    all_counties = ee.FeatureCollection("WM/geoLab/geoBoundaries/600/ADM1")
    kenya_counties = all_counties.filter(ee.Filter.eq("shapeGroup", "KEN"))
    selected_fc = kenya_counties.filter(ee.Filter.inList("shapeName", list(county_selection)))
    return kenya_counties, selected_fc


@st.cache_resource(show_spinner=False)
def build_predictor_stack(_project_id: str, county_selection: tuple[str, ...], agb_year: str):
    kenya_counties, selected_fc = get_study_area(_project_id, county_selection)
    geom = selected_fc.geometry()

    esa_agb = (
        ee.ImageCollection(ESA_AGB_COLLECTION)
        .filterDate(f"{agb_year}-01-01", f"{agb_year}-12-31")
        .first()
        .select(ESA_AGB_BAND)
    )
    biomass_data = esa_agb.multiply(CARBON_TO_BIOMASS_FACTOR).rename("carbon_tonnes_per_ha")

    s2 = (
        ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
        .select("B.*")
        .filterBounds(geom)
        .filterDate(DATE_START_S2, DATE_END_S2)
        .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 10))
        .median()
        .multiply(0.0001)
    )
    ndvi = s2.normalizedDifference(["B8", "B4"]).rename("NDVI")
    evi  = s2.expression(
        "2.5 * ((NIR - RED) / (NIR + 6 * RED - 7.5 * BLUE + 1))",
        {"NIR": s2.select("B8"), "RED": s2.select("B4"), "BLUE": s2.select("B2")},
    ).rename("EVI")
    savi = s2.expression(
        "((NIR - RED) / (NIR + RED + 0.5)) * (1.5)",
        {"NIR": s2.select("B8"), "RED": s2.select("B4")},
    ).rename("SAVI")
    ndmi = s2.normalizedDifference(["B8", "B11"]).rename("NDMI")
    ndre = s2.normalizedDifference(["B5", "B8"]).rename("NDRE")

    dynamic_world_label = (
        ee.ImageCollection("GOOGLE/DYNAMICWORLD/V1")
        .select("label")
        .filterDate(DATE_START_S2, DATE_END_S2)
        .filterBounds(geom)
        .mode()
    )
    landcover = dynamic_world_label.eq(1)
    masked_biomass = biomass_data.updateMask(landcover)
    biomass_mask   = masked_biomass.mask().gt(0)

    dem       = ee.Image("USGS/SRTMGL1_003")
    elevation = dem.select("elevation")
    slope     = ee.Terrain.slope(dem).rename("slope")
    aspect    = ee.Terrain.aspect(dem).rename("aspect")

    worldclim    = ee.Image("WORLDCLIM/V1/BIO")
    mean_temp    = worldclim.select("bio01").rename("mean_temp")
    annual_precip = worldclim.select("bio12").rename("annual_precip")

    s1 = (
        ee.ImageCollection("COPERNICUS/S1_GRD")
        .filterBounds(geom)
        .filterDate(DATE_START_S2, DATE_END_S2)
        .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VH"))
        .filter(ee.Filter.eq("instrumentMode", "IW"))
        .select("VH")
        .median()
        .rename("VH")
    )
    vh_int   = s1.multiply(100).toInt32()
    glcm     = vh_int.glcmTexture(size=4)
    contrast = glcm.select("VH_contrast").rename("S1_contrast")

    soil_carbon = (
        ee.Image("OpenLandMap/SOL/SOL_ORGANIC-CARBON_USDA-6A1C_M/v02")
        .select("b0")
        .rename("soil_carbon")
    )
    canopy_height = (
        ee.ImageCollection("projects/meta-forest-monitoring-okw37/assets/CanopyHeight")
        .mosaic()
        .rename("canopy_height")
    )
    modis_lst = ee.ImageCollection("MODIS/061/MOD11A2").filterDate(DATE_START_S2, DATE_END_S2).median()
    lst = modis_lst.select("LST_Day_1km").multiply(0.02).subtract(273.15).rename("LST")

    palsar_composite = (
        ee.ImageCollection("JAXA/ALOS/PALSAR/YEARLY/SAR_EPOCH")
        .filterDate(DATE_START_S2, DATE_END_S2)
        .filterBounds(geom)
        .median()
    )
    hh = palsar_composite.select("HH").rename("PALSAR_HH")
    hv = palsar_composite.select("HV").rename("PALSAR_HV")

    predictors_all = (
        ee.Image.constant(1)
        .addBands(s2).addBands(ndvi).addBands(evi).addBands(savi)
        .addBands(elevation).addBands(slope).addBands(aspect)
        .addBands(s1).addBands(canopy_height)
        .addBands(mean_temp).addBands(annual_precip)
        .addBands(soil_carbon)
        .addBands(hh).addBands(hv)
        .addBands(contrast).addBands(ndmi).addBands(ndre).addBands(lst)
    )

    predictors_mask      = predictors_all.mask().reduce(ee.Reducer.min())
    final_combined_mask  = biomass_mask.And(predictors_mask)
    predictor_variables  = predictors_all.updateMask(final_combined_mask)
    final_biomass        = masked_biomass.updateMask(final_combined_mask)

    return {
        "selected_fc": selected_fc,
        "kenya_counties": kenya_counties,
        "biomass_data": biomass_data,
        "dynamic_world_label": dynamic_world_label,
        "predictor_variables": predictor_variables,
        "final_biomass": final_biomass,
    }


@st.cache_resource(show_spinner=False)
def sample_and_split(_project_id, county_selection, num_pixels, train_split, seed, agb_year):
    stack = build_predictor_stack(_project_id, county_selection, agb_year)
    predictor_variables = stack["predictor_variables"]
    final_biomass       = stack["final_biomass"]
    selected_fc         = stack["selected_fc"]

    dependent_variable   = "carbon_tonnes_per_ha"
    predictor_band_names = predictor_variables.bandNames().getInfo()
    if "constant" in predictor_band_names:
        predictor_band_names.remove("constant")

    combined_dataset = predictor_variables.addBands(final_biomass)
    all_sampled = combined_dataset.sample(
        region=selected_fc.geometry(), scale=300,
        numPixels=num_pixels, geometries=True, tileScale=16,
    )
    all_sampled  = all_sampled.randomColumn(seed=seed)
    training_set = all_sampled.filter(ee.Filter.lt("random", train_split))
    testing_set  = all_sampled.filter(ee.Filter.gte("random", train_split))

    n_train = round(num_pixels * train_split)
    return {
        "dependent_variable":   dependent_variable,
        "predictor_band_names": predictor_band_names,
        "training_set":  training_set,
        "testing_set":   testing_set,
        "n_total": num_pixels,
        "n_train": n_train,
        "n_test":  num_pixels - n_train,
    }


@st.cache_resource(show_spinner=False)
def train_models(_project_id, county_selection, num_pixels, train_split, seed,
                 rf_trees, rf_vars_per_split, rf_min_leaf,
                 svm_gamma, svm_cost,
                 gtb_trees, gtb_shrinkage, gtb_sampling_rate, gtb_max_nodes,
                 agb_year):
    sample = sample_and_split(_project_id, county_selection, num_pixels, train_split, seed, agb_year)
    training_set         = sample["training_set"]
    dependent_variable   = sample["dependent_variable"]
    predictor_band_names = sample["predictor_band_names"]

    rf_classifier = ee.Classifier.smileRandomForest(
        numberOfTrees=rf_trees, variablesPerSplit=rf_vars_per_split,
        minLeafPopulation=rf_min_leaf, seed=seed,
    ).setOutputMode("REGRESSION")
    rf_model      = rf_classifier.train(
        features=training_set, classProperty=dependent_variable,
        inputProperties=predictor_band_names,
    )

    svm_classifier = ee.Classifier.libsvm(
        svmType="EPSILON_SVR", kernelType="RBF", gamma=svm_gamma, cost=svm_cost,
    ).setOutputMode("REGRESSION")
    svm_model = svm_classifier.train(
        features=training_set, classProperty=dependent_variable,
        inputProperties=predictor_band_names,
    )

    gtb_classifier = ee.Classifier.smileGradientTreeBoost(
        numberOfTrees=gtb_trees, shrinkage=gtb_shrinkage,
        samplingRate=gtb_sampling_rate, maxNodes=gtb_max_nodes, seed=seed,
    ).setOutputMode("REGRESSION")
    gtb_model      = gtb_classifier.train(
        features=training_set, classProperty=dependent_variable,
        inputProperties=predictor_band_names,
    )

    return {
        "rf_model":  rf_model,  "svm_model": svm_model, "gtb_model": gtb_model,
    }


def compute_validation_metrics(testing_set, model, dependent_variable):
    predicted = testing_set.classify(model)

    def add_errors(feature):
        actual = ee.Number(feature.get(dependent_variable))
        pred   = ee.Number(feature.get("classification"))
        diff   = actual.subtract(pred)
        return feature.set("sq_diff", diff.pow(2), "abs_diff", diff.abs())

    with_errors = predicted.map(add_errors)
    rmse = ee.Number(
        with_errors.reduceColumns(ee.Reducer.mean(), ["sq_diff"]).get("mean")
    ).sqrt().getInfo()
    mae = ee.Number(
        with_errors.reduceColumns(ee.Reducer.mean(), ["abs_diff"]).get("mean")
    ).getInfo()
    r2 = ee.Number(
        predicted.reduceColumns(
            ee.Reducer.pearsonsCorrelation(), [dependent_variable, "classification"]
        ).get("correlation")
    ).pow(2).getInfo()
    actual_vals = predicted.aggregate_array(dependent_variable).getInfo()
    pred_vals   = predicted.aggregate_array("classification").getInfo()
    return {"rmse": rmse, "mae": mae, "r2": r2, "actual": actual_vals, "predicted": pred_vals}


def make_scatter_plot(actual, predicted, title, color="#2d6a4f"):
    fig, ax = plt.subplots(figsize=(5, 4.5))
    ax.scatter(actual, predicted, alpha=0.45, s=22, color=color, edgecolors="none")
    max_val = max(max(actual, default=1), max(predicted, default=1)) * 1.08
    ax.plot([0, max_val], [0, max_val], color="#e63946", linewidth=1.2, linestyle="--", label="1:1 line")
    ax.set_xlim(0, max_val)
    ax.set_ylim(0, max_val)
    ax.set_xlabel("Actual (t C/ha)")
    ax.set_ylabel("Predicted (t C/ha)")
    ax.set_title(title)
    ax.legend(fontsize=8)
    fig.tight_layout()
    return fig


def build_report_pdf(params, validation_results=None, zonal_df=None):
    """Create a compact, downloadable report without adding a PDF dependency."""
    buffer = io.BytesIO()
    with PdfPages(buffer) as pdf:
        fig = plt.figure(figsize=(8.27, 11.69))
        fig.text(0.08, 0.94, "Carbon Stock & AGB Estimation", fontsize=20,
                 fontweight="bold", color="#1a472a")
        fig.text(0.08, 0.90, "Analysis summary", fontsize=12, color="#2d6a4f")
        summary = [
            f"Reference year: {params['agb_year']}",
            f"Counties: {', '.join(params['county_selection'])}",
            f"Sample pixels: {params['num_pixels']:,}",
            f"Training split: {params['train_split']:.0%}",
            f"Preset: {params.get('preset', 'Custom')}",
        ]
        fig.text(0.08, 0.82, "\n".join(summary), fontsize=11, va="top", linespacing=1.6)
        if validation_results:
            metrics_df = pd.DataFrame(
                {name: {"RMSE": r["rmse"], "MAE": r["mae"], "R²": r["r2"]}
                 for name, r in validation_results.items()}
            ).T.apply(pd.to_numeric, errors="coerce")
            fig.text(0.08, 0.55, "Held-out validation", fontsize=13,
                     fontweight="bold", color="#1a472a")
            table_ax = fig.add_axes((0.08, 0.38, 0.84, 0.14))
            table_ax.axis("off")
            table_ax.table(cellText=metrics_df.round(3), loc="center")
        if zonal_df is not None and not zonal_df.empty:
            fig.text(0.08, 0.30, "County estimates", fontsize=13,
                     fontweight="bold", color="#1a472a")
            table_ax = fig.add_axes((0.08, 0.06, 0.84, 0.20))
            table_ax.axis("off")
            report_rows = zonal_df.head(10)[["County", "Mean (t/ha)", "Sum (t)"]].copy()
            table_ax.table(cellText=report_rows.set_index("County").round(2), loc="center")
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)
    return buffer.getvalue()


def build_learning_context(params):
    """Return only the current, non-sensitive analysis facts needed by the guide."""
    context = {
        "reference_year": params.get("agb_year"),
        "selected_counties": list(params.get("county_selection", [])),
        "sample_pixels": params.get("num_pixels"),
        "training_split": params.get("train_split"),
        "models": ["Random Forest", "Gradient Tree Boosting", "Support Vector Machine", "Ensemble"],
        "carbon_to_biomass_factor": CARBON_TO_BIOMASS_FACTOR,
    }
    validation = st.session_state.get("validation_results")
    if validation:
        context["validation"] = {
            name: {key: result.get(key) for key in ("rmse", "mae", "r2")}
            for name, result in validation.items()
        }
    if st.session_state.get("mean_spread") is not None:
        context["mean_model_spread_tC_ha"] = st.session_state["mean_spread"]
    return json.dumps(context, default=str)


def ask_learning_guide(provider, model, user_message, chat_history, context, gemini_key="",
                       language="English", audio_bytes=None, audio_mime_type="audio/wav"):
    """Ask Gemini, with a local Ollama fallback option, without persisting credentials."""
    instructions = """
You are the Environmental & Carbon Learning Guide inside a Kenya carbon-stock and above-ground
biomass app. Teach someone with no technical background using warm, plain language and short
paragraphs. You can explain: climate change, greenhouse gases, carbon emissions and emitters,
carbon footprints, forests and nature-based solutions, carbon stock and AGB, offsets and carbon
credits, voluntary and compliance carbon markets, additionality, leakage, permanence, monitoring,
verification, and responsible ways people, organisations, and governments can reduce emissions.

Explain maps, colours, units, uncertainty, validation metrics, data sources, and model outputs.
Use the supplied app context only for current run details. Clearly distinguish estimates from
measurements. Never present an output as a verified carbon credit, market price, investment
recommendation, legal conclusion, or certification decision. For current carbon prices, laws,
policies, named project claims, or recent events, say that they change over time and recommend
checking an authoritative current source. Do not invent statistics or sources. For questions about
major emitters, discuss sectors and drivers without shaming individuals or making unsupported
claims about a company or community. End map interpretations with one practical next step.
For beginner questions, give a complete answer in 4–6 short paragraphs.
Define unfamiliar terms, use a simple analogy, and finish with a practical next step.
Never end mid-sentence. If the topic has several parts, use concise headings or bullets.
""".strip()
    language_instruction = {
        "English": "Reply in clear English.",
        "Kiswahili": "Jibu kwa Kiswahili rahisi na wazi.",
        "English + Kiswahili": "Give each key explanation first in clear English, then in clear Kiswahili.",
    }.get(language, "Reply in clear English.")
    messages = [{"role": "system", "content": instructions + "\n\n" + language_instruction + "\n\nCurrent app context:\n" + context}]
    messages.extend({"role": item["role"], "content": item["content"]} for item in chat_history[-10:])
    messages.append({"role": "user", "content": user_message})

    if provider == "Gemini":
        if not gemini_key:
            raise ValueError("Add a Gemini API key in the guide settings or Streamlit secrets.")
        contents = []
        for message in messages[1:]:
            contents.append({
                "role": "model" if message["role"] == "assistant" else "user",
                "parts": [{"text": message["content"]}],
            })
        if audio_bytes:
            contents[-1]["parts"].append({
                "inline_data": {
                    "mime_type": audio_mime_type,
                    "data": base64.b64encode(audio_bytes).decode("ascii"),
                }
            })
        payload = json.dumps({
            "system_instruction": {"parts": [{"text": messages[0]["content"]}]},
            "contents": contents,
            "generationConfig": {"temperature": 0.35, "maxOutputTokens": 1200},
        }).encode("utf-8")
        request = urllib.request.Request(
            f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent",
            data=payload,
            headers={"Content-Type": "application/json", "x-goog-api-key": gemini_key},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=45) as response:
                data = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as error:
            detail = error.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"Gemini request failed ({error.code}): {detail[:300]}") from error
        except urllib.error.URLError as error:
            raise RuntimeError(f"Could not reach Gemini: {error.reason}") from error
        try:
            return data["candidates"][0]["content"]["parts"][0]["text"]
        except (KeyError, IndexError) as error:
            raise RuntimeError("Gemini returned no text response for this question.") from error

    if audio_bytes:
        raise RuntimeError("Voice questions require Gemini. Use a typed question with the local Ollama fallback.")
    payload = json.dumps({"model": model, "messages": messages, "stream": False}).encode("utf-8")
    request = urllib.request.Request(
        "http://localhost:11434/api/chat", data=payload,
        headers={"Content-Type": "application/json"}, method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=90) as response:
            data = json.loads(response.read().decode("utf-8"))
        return data["message"]["content"]
    except urllib.error.URLError as error:
        raise RuntimeError(
            "Could not reach Ollama at http://localhost:11434. Start Ollama and pull the selected model first."
        ) from error


def offline_learning_response(question):
    """Provide a useful, no-network explanation while AI services are unavailable."""
    query = question.lower()
    if any(word in query for word in ("feature", "used", "predictor", "data source")):
        return (
            "**This app combines several kinds of information to estimate carbon.**\n\n"
            "- **Sentinel-1 radar** helps describe vegetation structure, even through clouds.\n"
            "- **Sentinel-2 imagery** provides colour-based vegetation measures such as NDVI.\n"
            "- **Terrain, rainfall, temperature, soil, canopy height, and land-surface temperature** help explain why biomass differs from place to place.\n"
            "- **Random Forest, Gradient Tree Boosting, and SVM** are three different prediction methods. The ensemble is their average.\n\n"
            "These are estimates, not direct field measurements. A good next step is to open **Variable Importance** to see which inputs mattered most in this run."
        )
    if any(word in query for word in ("map", "colour", "color", "legend")):
        return (
            "**How to read the map:** use the colour bar beside the map. Each colour represents a range of estimated carbon stock or AGB. "
            "Read the number and unit on the legend before deciding whether a colour is high or low. County outlines only show boundaries; they do not change the estimate. "
            "Turn on **confidence classes** to see where the three models agree most."
        )
    if any(word in query for word in ("credit", "market", "offset")):
        return (
            "A **carbon credit** normally represents one tonne of verified carbon dioxide equivalent reduced or removed under a recognised method. "
            "This app estimates carbon stock; it does **not** create, verify, price, or certify credits. A credible project also needs a baseline, additionality, monitoring, verification, and checks for leakage and permanence."
        )
    if any(word in query for word in ("climate", "emission", "emitter", "greenhouse")):
        return (
            "Climate change is driven mainly by greenhouse gases accumulating in the atmosphere. Key sources include energy, transport, industry, agriculture, land-use change, and waste. "
            "The most reliable action is usually to reduce emissions at the source first; protecting or restoring ecosystems can complement, but not replace, those reductions."
        )
    return (
        "The online guide is not connected yet, but I can still help with the app basics. Try asking about **map colours**, **features used**, **validation**, **carbon credits**, or **climate emissions**. "
        "For open-ended questions, add a Gemini key in the guide settings or start Ollama locally."
    )


def render_voice_player(text, language):
    """Offer browser-native text-to-speech without an additional cloud service."""
    language_code = "sw-KE" if language == "Kiswahili" else "en-KE"
    safe_text = json.dumps(text)
    st.iframe(
        f"""
        <style>
          body {{ margin: 0; font-family: sans-serif; background: transparent; }}
          button {{ background: #2d6a4f; color: white; border: 0; border-radius: 8px; padding: 8px 14px; font-weight: 700; cursor: pointer; }}
        </style>
        <button onclick='window.speechSynthesis.cancel(); const speech = new SpeechSynthesisUtterance({safe_text}); speech.lang = "{language_code}"; window.speechSynthesis.speak(speech);'>🔊 Listen to the latest answer</button>
        """,
        height=48,
    )


def build_portfolio_pdf(entries):
    """Create a concise combined report from saved in-session project snapshots."""
    buffer = io.BytesIO()
    with PdfPages(buffer) as pdf:
        fig = plt.figure(figsize=(11.69, 8.27))
        fig.text(0.07, 0.93, "Carbon Project Portfolio", fontsize=20,
                 fontweight="bold", color="#1a472a")
        fig.text(0.07, 0.89, "Comparison of saved analysis snapshots", fontsize=11, color="#2d6a4f")
        table_rows = [
            [
                entry["Project"], entry["Counties"], entry["Reference year"],
                entry["Samples"], entry["Best model"], entry["Best RMSE"],
                entry["Mean spread"],
            ]
            for entry in entries
        ]
        table_ax = fig.add_axes((0.07, 0.22, 0.86, 0.56))
        table_ax.axis("off")
        table_ax.table(
            cellText=table_rows,
            colLabels=["Project", "Counties", "Year", "Samples", "Best model", "Best RMSE", "Mean spread"],
            loc="center", cellLoc="left",
        )
        fig.text(
            0.07, 0.10,
            "Caution: these are model-analysis snapshots. They are not verified carbon-credit, valuation, or certification records.",
            fontsize=9, color="#5a6e63",
        )
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)
    return buffer.getvalue()


def build_map_briefing(params, validation_results=None, mean_spread=None, zonal_df=None):
    """Create a transparent plain-language briefing from already computed results."""
    counties = list(params.get("county_selection", []))
    county_phrase = ", ".join(counties[:4])
    if len(counties) > 4:
        county_phrase += f", and {len(counties) - 4} more"

    strongest_model = None
    strongest_rmse = None
    if validation_results:
        metric_rows = []
        for name, result in validation_results.items():
            try:
                metric_rows.append((name, float(result.get("rmse"))))
            except (TypeError, ValueError):
                continue
        if metric_rows:
            strongest_model, strongest_rmse = min(metric_rows, key=lambda item: item[1])

    briefing = [
        "### Plain-language map briefing",
        f"This run estimates above-ground carbon across **{len(counties)} selected county/counties**: {county_phrase}. "
        f"It uses the **{params.get('agb_year')} AGB reference year** and {params.get('num_pixels', 0):,} sampled pixels.",
    ]
    if strongest_model:
        briefing.append(
            f"**Strongest tested model:** {strongest_model} had the lowest held-out RMSE "
            f"({strongest_rmse:.2f} t C/ha) in this run. This makes it the most accurate of the tested models here, "
            "not a guarantee that every map pixel is correct."
        )
    else:
        briefing.append(
            "**Model strength:** validation has not been computed yet. Open the Validation tab and run its first step "
            "before choosing a preferred model."
        )
    if mean_spread is not None:
        try:
            spread = float(mean_spread)
            confidence_note = "fairly close agreement" if spread < 10 else "moderate disagreement" if spread < 20 else "substantial disagreement"
            briefing.append(
                f"**Uncertainty:** the average model spread is {spread:.2f} t C/ha, indicating {confidence_note} between the models. "
                "Use the confidence layer to see where that agreement changes across the map."
            )
        except (TypeError, ValueError):
            pass
    else:
        briefing.append(
            "**Uncertainty:** model spread has not been calculated yet. Use Model Comparison → Compute mean model spread "
            "for an uncertainty statement."
        )
    if zonal_df is not None and not zonal_df.empty and "Mean (t/ha)" in zonal_df:
        ranked = zonal_df.dropna(subset=["Mean (t/ha)"]).sort_values("Mean (t/ha)")
        if not ranked.empty:
            low = ranked.iloc[0]
            high = ranked.iloc[-1]
            try:
                briefing.append(
                    f"**County hotspots:** {high['County']} has the highest estimated mean carbon stock "
                    f"({float(high['Mean (t/ha)']):.1f} t C/ha), while {low['County']} has the lowest "
                    f"({float(low['Mean (t/ha)']):.1f} t C/ha) for the selected zonal model."
                )
            except (TypeError, ValueError):
                briefing.append("**County hotspots:** county statistics are available, but their numeric values could not be interpreted for this briefing.")
    else:
        briefing.append(
            "**County hotspots:** zonal statistics have not been computed yet. Run them to identify the highest and lowest "
            "county estimates rather than judging by colour alone."
        )
    briefing.append(
        "**Caution:** this is a satellite-and-model estimate, not a field inventory or verified carbon-credit assessment. "
        "Use it to prioritise investigation, then confirm important decisions with local knowledge and field data."
    )
    return "\n\n".join(briefing)


def compute_watchlist_alerts(counties, baseline_year, comparison_year, threshold):
    """Compare available ESA reference maps for watched counties on demand."""
    baseline = (
        ee.ImageCollection(ESA_AGB_COLLECTION)
        .filterDate(f"{baseline_year}-01-01", f"{baseline_year}-12-31")
        .first().select(ESA_AGB_BAND).multiply(CARBON_TO_BIOMASS_FACTOR)
    )
    comparison = (
        ee.ImageCollection(ESA_AGB_COLLECTION)
        .filterDate(f"{comparison_year}-01-01", f"{comparison_year}-12-31")
        .first().select(ESA_AGB_BAND).multiply(CARBON_TO_BIOMASS_FACTOR)
    )
    watched_counties = get_study_area("watchlist", tuple(counties))[1]
    county_changes = comparison.subtract(baseline).rename("carbon_change_tC_ha").reduceRegions(
        collection=watched_counties,
        reducer=ee.Reducer.mean(), scale=1000, tileScale=16,
    ).getInfo()["features"]
    rows = []
    for feature in county_changes:
        value = feature["properties"].get("mean")
        if value is None:
            continue
        change = float(value)
        if change <= -threshold:
            status = "Alert: decline"
        elif change >= threshold:
            status = "Alert: gain"
        else:
            status = "No material change"
        rows.append({
            "County": feature["properties"].get("shapeName", "Unknown"),
            f"Change {baseline_year}–{comparison_year} (t C/ha)": change,
            "Status": status,
        })
    return pd.DataFrame(rows).sort_values(
        f"Change {baseline_year}–{comparison_year} (t C/ha)"
    ) if rows else pd.DataFrame(columns=["County", f"Change {baseline_year}–{comparison_year} (t C/ha)", "Status"])


def assess_data_quality(params, validation_results=None, mean_spread=None):
    """Return an explainable screening score from results already computed in the app."""
    counties = max(1, len(params.get("county_selection", [])))
    samples = int(params.get("num_pixels", 0))
    samples_per_county = samples / counties
    if samples >= 3000 and samples_per_county >= 300:
        sample_score, sample_note = 25, "Strong sample volume for this selected area."
    elif samples >= 1500 and samples_per_county >= 150:
        sample_score, sample_note = 18, "Usable sample volume; more samples may improve stability."
    else:
        sample_score, sample_note = 8, "Limited sample volume; increase samples before relying on comparisons."

    validation_score, validation_note = 0, "Validation has not been computed."
    if validation_results:
        r2_values = []
        for result in validation_results.values():
            try:
                r2_values.append(float(result.get("r2")))
            except (TypeError, ValueError):
                continue
        if r2_values:
            best_r2 = max(r2_values)
            if best_r2 >= 0.70:
                validation_score, validation_note = 25, f"Strong held-out validation signal (best R² {best_r2:.2f})."
            elif best_r2 >= 0.40:
                validation_score, validation_note = 17, f"Moderate held-out validation signal (best R² {best_r2:.2f})."
            else:
                validation_score, validation_note = 8, f"Weak held-out validation signal (best R² {best_r2:.2f})."

    uncertainty_score, uncertainty_note = 0, "Model agreement has not been computed."
    if mean_spread is not None:
        try:
            spread = float(mean_spread)
            if spread < 10:
                uncertainty_score, uncertainty_note = 25, f"High model agreement (mean spread {spread:.1f} t C/ha)."
            elif spread < 20:
                uncertainty_score, uncertainty_note = 17, f"Moderate model agreement (mean spread {spread:.1f} t C/ha)."
            else:
                uncertainty_score, uncertainty_note = 8, f"Low model agreement (mean spread {spread:.1f} t C/ha)."
        except (TypeError, ValueError):
            pass

    if samples_per_county >= 500:
        coverage_score, coverage_note = 25, f"Strong coverage proxy: about {samples_per_county:.0f} samples per selected county."
    elif samples_per_county >= 200:
        coverage_score, coverage_note = 17, f"Moderate coverage proxy: about {samples_per_county:.0f} samples per selected county."
    else:
        coverage_score, coverage_note = 8, f"Sparse coverage proxy: about {samples_per_county:.0f} samples per selected county."

    total = sample_score + validation_score + uncertainty_score + coverage_score
    label = "Strong screening run" if total >= 75 else "Usable with cautions" if total >= 50 else "Needs improvement"
    return {
        "total": total, "label": label,
        "components": [
            ("Sampling strength", sample_score, sample_note),
            ("Held-out validation", validation_score, validation_note),
            ("Model agreement", uncertainty_score, uncertainty_note),
            ("Geographic coverage proxy", coverage_score, coverage_note),
        ],
    }


# ============================================================================
# SIDEBAR
# ============================================================================
st.sidebar.markdown("""
<div style="background:rgba(0,0,0,0.22);border-radius:12px;padding:1.1rem 1rem 0.9rem;
            margin-bottom:1rem;text-align:center;">
    <i class="fa-solid fa-tree"
       style="font-size:2.4rem;color:#74c69d;line-height:1;"></i>
    <div style="font-weight:800;font-size:1.05rem;color:#ffffff;margin-top:0.55rem;">
        Carbon Stock &amp; AGB
    </div>
    <div style="font-size:0.75rem;color:rgba(255,255,255,0.60);margin-top:0.2rem;">
        Kenya &middot; Satellite ML Estimation
    </div>
</div>
""", unsafe_allow_html=True)

# Auto-connect via service account when running on Streamlit Cloud
if "gee" in st.secrets and not st.session_state.get("ee_ready"):
    _pid = st.secrets["gee"].get("project_id", "")
    _ok, _msg = init_earth_engine(_pid)
    if _ok:
        st.session_state["ee_ready"] = True
        st.session_state["ee_project_id"] = _pid
    else:
        st.session_state["ee_ready"] = False
        st.session_state["ee_init_error"] = _msg
elif _is_cloud_deployment() and not st.session_state.get("ee_ready"):
    # Cloud deployment but no secrets configured — surface setup guide immediately.
    st.session_state["ee_ready"] = False
    st.session_state["ee_init_error"] = _STREAMLIT_CLOUD_SETUP

with st.sidebar.expander("Earth Engine Setup", expanded=not st.session_state.get("ee_ready", False)):
    if st.session_state.get("ee_ready") and "gee" in st.secrets:
        st.success("Connected via service account.")
    elif _is_cloud_deployment():
        err = st.session_state.get("ee_init_error", _STREAMLIT_CLOUD_SETUP)
        st.warning(err)
    else:
        st.markdown(
            "Requires a **Google Earth Engine** account with a registered Cloud project. "
            "Run `earthengine authenticate` once in a terminal, then enter your project ID below."
        )
        if st.session_state.get("ee_init_error"):
            st.error(st.session_state["ee_init_error"])
        default_project = st.session_state.get("ee_project_id", "")
        project_id = st.text_input("GEE Cloud Project ID", value=default_project,
                                   placeholder="my-gee-project")
        if st.button("Connect to Earth Engine", width='stretch'):
            if not project_id:
                st.error("Please enter a GEE Cloud Project ID.")
            else:
                with st.spinner("Connecting…"):
                    ok, msg = init_earth_engine(project_id)
                if ok:
                    st.session_state["ee_ready"] = True
                    st.session_state["ee_project_id"] = project_id
                    st.session_state.pop("ee_init_error", None)
                    st.success(msg)
                else:
                    st.session_state["ee_ready"] = False
                    st.session_state["ee_init_error"] = msg
                    st.error(msg)

ee_ready = st.session_state.get("ee_ready", False)

st.sidebar.markdown("---")
with st.sidebar.form("analysis_configuration", border=False):
    st.markdown("**Analysis configuration**")
    st.caption("Changes are applied only when you run the analysis.")
    analysis_preset = st.selectbox(
        "Analysis preset", ["Custom", *ANALYSIS_PRESETS],
        help="Quick preview is best for exploration; high accuracy takes longer.",
    )
    agb_year = st.select_slider(
        "AGB reference year", options=AGB_YEARS_AVAIL, value="2020",
        help="ESA CCI AGB years available from 2010 to 2022. The predictor composite uses satellite data through 2023.",
    )
    county_selection = st.multiselect(
        "Kenyan counties (ADM1)", options=ALL_KENYA_COUNTIES_OPTIONS,
        default=ALL_KENYA_COUNTIES_OPTIONS,
    )
    _extra_raw = st.text_input(
        "Add unlisted counties", placeholder="e.g. Nairobi, Kisumu",
        help="Comma-separated names must match the geoBoundaries shapeName field exactly.",
    )

    st.markdown("**Sampling**")
    num_pixels  = st.slider("Sample pixels", 500, 10000, 3000, step=500)
    train_split = st.slider("Training split", 0.5, 0.9, 0.7, step=0.05,
                            help="Fraction of pixels used for training")
    seed        = st.number_input("Random seed", value=0, step=1)

    with st.expander("Advanced model settings"):
        st.markdown("**Random Forest**")
        rf_trees          = st.slider("Trees", 50, 500, 100, step=10, key="rf_trees")
        rf_vars_per_split = st.slider("Variables per split", 1, 15, 6, key="rf_vars")
        rf_min_leaf       = st.slider("Min leaf population", 1, 50, 10, key="rf_leaf")

        st.markdown("**Gradient Tree Boosting**")
        gtb_trees         = st.slider("Trees", 50, 500, 100, step=10, key="gtb_trees")
        gtb_shrinkage     = st.slider("Shrinkage", 0.001, 0.1, 0.005, step=0.001,
                                       format="%.3f", key="gtb_shrink")
        gtb_sampling_rate = st.slider("Sampling rate", 0.1, 1.0, 0.6, step=0.05, key="gtb_rate")
        gtb_max_nodes     = st.slider("Max nodes", 2, 32, 8, key="gtb_nodes")

        st.markdown("**Support Vector Machine**")
        svm_gamma = st.slider("Gamma", 0.01, 2.0, 0.6, step=0.01, key="svm_gamma")
        svm_cost  = st.slider("Cost", 1.0, 100.0, 10.0, step=1.0, key="svm_cost")

    run_clicked = st.form_submit_button(
        "Run analysis", type="primary", width="stretch", disabled=not ee_ready
    )

if _extra_raw:
    _extra = [c.strip() for c in _extra_raw.split(",") if c.strip()]
    county_selection = list(dict.fromkeys(county_selection + _extra))

if not ee_ready:
    st.sidebar.warning("Connect to Earth Engine above first.")

with st.sidebar.expander("Load saved configuration"):
    saved_config_file = st.file_uploader("Load a saved run", type="json", key="saved_run_file")
    load_saved_run = st.button(
        "Run saved configuration", width="stretch",
        disabled=saved_config_file is None or not ee_ready,
    )

if load_saved_run and saved_config_file is not None:
    try:
        saved_params = json.loads(saved_config_file.getvalue().decode("utf-8"))
        saved_params["county_selection"] = tuple(saved_params["county_selection"])
        required_params = {
            "county_selection", "agb_year", "num_pixels", "train_split", "seed",
            "rf_trees", "rf_vars_per_split", "rf_min_leaf", "svm_gamma", "svm_cost",
            "gtb_trees", "gtb_shrinkage", "gtb_sampling_rate", "gtb_max_nodes",
        }
        if not required_params.issubset(saved_params):
            raise ValueError("The file does not contain a complete analysis configuration.")
        st.session_state["params"] = saved_params
        st.session_state["analysis_ready"] = True
        for _key in ("validation_results", "zonal_df", "mean_spread", "importance_results"):
            st.session_state.pop(_key, None)
        st.rerun()
    except Exception as _e:
        st.sidebar.error(f"Could not load the configuration: {_e}")

with st.sidebar.expander("Display options"):
    high_contrast = st.checkbox("High contrast mode")
    large_text = st.checkbox("Larger text")
if high_contrast or large_text:
    st.markdown(
        "<style>"
        + (".stApp { background:#ffffff !important; color:#000 !important; } " if high_contrast else "")
        + ("html { font-size: 18px; } " if large_text else "")
        + "</style>",
        unsafe_allow_html=True,
    )


# ============================================================================
# MAIN — HERO
# ============================================================================
st.markdown("""
<div class="hero">
    <h1><i class="fa-solid fa-tree" style="font-size:1.6rem;margin-right:0.4rem;opacity:0.9;"></i>Carbon Stock &amp; Biomass Estimation</h1>
    <p>
        Estimates above-ground biomass carbon stock across selected Kenyan counties
        using multi-source satellite data and ensemble machine learning regressors.
    </p>
    <div class="hero-tags">
        <span class="hero-tag">Sentinel-1 SAR</span>
        <span class="hero-tag">Sentinel-2 MSI</span>
        <span class="hero-tag">PALSAR</span>
        <span class="hero-tag">SRTM DEM</span>
        <span class="hero-tag">WorldClim</span>
        <span class="hero-tag">MODIS LST</span>
        <span class="hero-tag">Random Forest</span>
        <span class="hero-tag">Gradient Tree Boosting</span>
        <span class="hero-tag">SVM</span>
        <span class="hero-tag">Ensemble</span>
    </div>
</div>
""", unsafe_allow_html=True)

# ----------------------------------------------------------------------------
# GUARD STATES
# ----------------------------------------------------------------------------
if not ee_ready:
    st.markdown("""
    <div class="welcome">
        <i class="fa-solid fa-satellite icon"></i>
        <h3>Connect to Earth Engine to get started</h3>
        <p>Enter your Google Earth Engine Cloud Project ID in the sidebar and click
        <strong>Connect to Earth Engine</strong>.</p>
    </div>
    """, unsafe_allow_html=True)
    st.stop()

if not county_selection:
    st.warning("Select at least one county in the sidebar.")
    st.stop()

if run_clicked:
    # Results below depend on the active model run; discard prior on-demand
    # outputs so the interface never presents stale maps or statistics.
    for _key in (
        "validation_results", "zonal_df", "mean_spread", "show_diff_map",
        "show_spread_map", "importance_results", "map_briefing", "watchlist_alerts",
        "watchlist_check", "watchlist_counties", "restoration_scenario", "restoration_draw_geometry",
        "data_quality_assessment",
    ):
        st.session_state.pop(_key, None)
    st.session_state["analysis_ready"] = True
    submitted_params = dict(
        county_selection=tuple(county_selection),
        agb_year=agb_year,
        num_pixels=num_pixels, train_split=train_split, seed=seed,
        rf_trees=rf_trees, rf_vars_per_split=rf_vars_per_split, rf_min_leaf=rf_min_leaf,
        svm_gamma=svm_gamma, svm_cost=svm_cost,
        gtb_trees=gtb_trees, gtb_shrinkage=gtb_shrinkage,
        gtb_sampling_rate=gtb_sampling_rate, gtb_max_nodes=gtb_max_nodes,
    )
    if analysis_preset != "Custom":
        submitted_params.update(ANALYSIS_PRESETS[analysis_preset])
    submitted_params["preset"] = analysis_preset
    st.session_state["params"] = submitted_params
    st.session_state["analysis_started_at"] = time.perf_counter()

if not st.session_state.get("analysis_ready"):
    st.markdown("""
    <div class="welcome">
        <i class="fa-solid fa-sliders icon"></i>
        <h3>Configure and run the analysis</h3>
        <p>Choose counties, adjust sampling and model parameters in the sidebar,
        then click <strong>Run Analysis</strong>.</p>
    </div>
    """, unsafe_allow_html=True)
    st.stop()

p          = st.session_state["params"]
project_id = st.session_state["ee_project_id"]

with st.sidebar.expander("Save current configuration"):
    st.download_button(
        "Download current configuration",
        data=json.dumps(p, indent=2),
        file_name="carbon_agb_run_config.json",
        mime="application/json",
        width="stretch",
    )

# ----------------------------------------------------------------------------
# BUILD PREDICTOR STACK
# ----------------------------------------------------------------------------
run_status = st.status("Preparing analysis", expanded=False)
run_status.update(label="1/3 Building predictor stack", state="running")
with st.spinner("Building predictor stack on Earth Engine…"):
    stack = build_predictor_stack(project_id, p["county_selection"], p["agb_year"])

with st.spinner(f"Sampling {p['num_pixels']:,} pixels and splitting train / test…"):
    run_status.update(label="2/3 Sampling training and testing pixels", state="running")
    sample = sample_and_split(
        project_id, p["county_selection"],
        p["num_pixels"], p["train_split"], p["seed"], p["agb_year"],
    )

col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("AGB Reference Year",   p["agb_year"])
col2.metric("Selected counties",    len(p["county_selection"]))
col3.metric("Sampled points",       f"{sample['n_total']:,}")
col4.metric("Training points",      f"{sample['n_train']:,}")
col5.metric("Testing points",       f"{sample['n_test']:,}")

with st.spinner("Training Random Forest, GTB, and SVM models on Earth Engine…"):
    run_status.update(label="3/3 Preparing model outputs", state="running")
    models = train_models(
        project_id, p["county_selection"],
        p["num_pixels"], p["train_split"], p["seed"],
        p["rf_trees"], p["rf_vars_per_split"], p["rf_min_leaf"],
        p["svm_gamma"], p["svm_cost"],
        p["gtb_trees"], p["gtb_shrinkage"], p["gtb_sampling_rate"], p["gtb_max_nodes"],
        p["agb_year"],
    )

if run_clicked:
    st.session_state["last_preparation_seconds"] = (
        time.perf_counter() - st.session_state["analysis_started_at"]
    )
run_status.update(label="Analysis ready", state="complete", expanded=False)
if st.session_state.get("last_preparation_seconds") is not None:
    st.caption(
        f"Run profile: {p.get('preset', 'Custom')} preset · "
        f"prepared in {st.session_state['last_preparation_seconds']:.1f}s · "
        "validation and feature importance load only when requested."
    )

predictor_variables  = stack["predictor_variables"]
selected_fc          = stack["selected_fc"]

estimated_carbon_rf  = predictor_variables.classify(models["rf_model"]).rename("Estimated Carbon Stock RF")
estimated_carbon_gtb = predictor_variables.classify(models["gtb_model"]).rename("Estimated Carbon Stock GTB")
estimated_carbon_svm = predictor_variables.classify(models["svm_model"]).rename("Estimated Carbon Stock SVM")

estimated_carbon_ensemble = (
    estimated_carbon_rf.add(estimated_carbon_gtb).add(estimated_carbon_svm)
    .divide(3)
    .rename("Estimated Carbon Stock Ensemble")
)

model_spread = (
    estimated_carbon_rf.rename("rf")
    .addBands(estimated_carbon_gtb.rename("gtb"))
    .addBands(estimated_carbon_svm.rename("svm"))
    .reduce(ee.Reducer.stdDev())
    .rename("Model_Spread")
)

# Simplified geometry used for clipping display images — complex county
# polygon vertices slow down getMapId(); 1 km tolerance is fine for tiles.
_display_geom = selected_fc.geometry().simplify(maxError=1000)


def safe_add_layer(map_obj, image, vis, label, shown=True):
    """Add an EE image layer, showing a warning instead of crashing on timeout."""
    try:
        map_obj.addLayer(image, vis, label, shown)
    except Exception as _e:
        st.warning(
            f"Map layer **{label}** could not be rendered "
            f"(Earth Engine timed out). Try a smaller study area or fewer sample pixels. "
            f"Detail: `{_e}`"
        )


def safe_center_map(map_obj, geometry, zoom=9):
    """Center on an EE geometry without letting transient API failures crash the app."""
    try:
        map_obj.centerObject(geometry, zoom)
    except Exception as _e:
        # The map is created with a Kenya-wide default view. Keep that usable
        # fallback when Earth Engine cannot calculate the geometry centroid.
        st.info(
            "Earth Engine could not center the map on the selected counties. "
            "Showing the Kenya-wide view instead; retry shortly if the issue persists."
        )


def render_gee_map(map_obj, height, bidirectional=False):
    """Render Folium maps through Streamlit-Folium.

    ``st.iframe`` puts the map in a sandboxed document, which can stop Earth
    Engine tile requests in desktop previews. Streamlit-Folium keeps the map
    component's working tile loader and avoids geemap's retired HTML renderer.
    """
    map_obj.add_layer_control()
    returned_objects = ["last_clicked", "last_active_drawing"] if bidirectional else []
    return st_folium(
        map_obj,
        height=height,
        use_container_width=True,
        returned_objects=returned_objects,
    )


MODEL_IMAGES = {
    "Random Forest":           estimated_carbon_rf,
    "Gradient Tree Boosting":  estimated_carbon_gtb,
    "Support Vector Machine":  estimated_carbon_svm,
    "Ensemble (RF+GTB+SVM)":   estimated_carbon_ensemble,
}

# ============================================================================
# TABS
# ============================================================================
tab_map, tab_briefing, tab_guide, tab_compare, tab_restoration, tab_tools, tab_validation, tab_quality, tab_zonal, tab_importance, tab_report = st.tabs([
    "Interactive Map",
    "Map Briefing",
    "Environmental Guide",
    "Model Comparison",
    "Restoration Scenario",
    "Decision Tools",
    "Validation",
    "Data Quality",
    "Zonal Statistics",
    "Variable Importance",
    "Report & Export",
])

# ── TAB: INTERACTIVE MAP ─────────────────────────────────────────────────────
with tab_map:
    st.markdown('<div class="sec-hdr"><i class="fa-solid fa-map"></i>Estimated Carbon Stock Map</div>', unsafe_allow_html=True)

    ctrl_col, map_col = st.columns([1, 3], gap="medium")
    with ctrl_col:
        selected_model_name   = st.radio("Model", list(MODEL_IMAGES.keys()), index=0)
        show_agb              = st.checkbox("Show AGB instead of carbon")
        show_counties_outline = st.checkbox("Show county boundaries", value=True)
        enable_map_inspector  = st.checkbox(
            "Enable map inspector",
            help="Click the map to inspect model output and selected predictors at that location.",
        )
        show_confidence = st.checkbox("Show confidence classes")
        show_land_cover = st.checkbox("Show Dynamic World land cover")
        st.caption(
            "**Map help / Msaada wa ramani:** choose a **Model** to change the prediction shown; "
            "use **Show AGB** for biomass instead of carbon; turn on **confidence** to see where models agree. "
            "Chagua **Model**, tumia **Show AGB** kuona biomasi, na washa **confidence** kuona maeneo ambayo modeli zinakubaliana."
        )
        st.caption(f"Carbon → biomass factor: **{CARBON_TO_BIOMASS_FACTOR}** (IPCC default)")

    with map_col:
        m             = geemap.Map(center=[0.3, 36.0], zoom=7)
        display_image = MODEL_IMAGES[selected_model_name]
        vis           = VIS_PARAMS_CARBON
        layer_label   = f"Carbon — {selected_model_name}"
        if show_agb:
            display_image = display_image.divide(CARBON_TO_BIOMASS_FACTOR)
            vis           = VIS_PARAMS_BIOMASS
            layer_label   = f"AGB — {selected_model_name}"

        safe_center_map(m, selected_fc.geometry(), 9)
        if show_counties_outline:
            safe_add_layer(m, selected_fc, {"color": "FF4444"}, "Selected Counties", True)
        safe_add_layer(m, display_image.clip(_display_geom), vis, layer_label, True)
        if show_confidence:
            confidence_classes = (
                model_spread.lt(10).multiply(2)
                .where(model_spread.gte(10).And(model_spread.lt(20)), 1)
                .where(model_spread.gte(20), 0)
                .rename("Confidence")
            )
            safe_add_layer(
                m, confidence_classes.clip(_display_geom),
                {"min": 0, "max": 2, "palette": ["#d73027", "#fee08b", "#1a9850"]},
                "Confidence: low / medium / high", False,
            )
        if show_land_cover:
            safe_add_layer(
                m, stack["dynamic_world_label"].clip(_display_geom),
                {"min": 0, "max": 8, "palette": [
                    "#419BDF", "#397D49", "#88B053", "#7A87C6", "#E49635",
                    "#DFC35A", "#C4281B", "#A59B8F", "#B39FE1",
                ]},
                "Dynamic World land cover", False,
            )
        m.add_colorbar(vis, label="t C/ha" if not show_agb else "Mg/ha AGB")
        map_event = render_gee_map(m, height=560, bidirectional=enable_map_inspector)
        if enable_map_inspector:
            clicked = (map_event or {}).get("last_clicked")
            if clicked:
                latitude = round(float(clicked["lat"]), 5)
                longitude = round(float(clicked["lng"]), 5)
                inspection_key = (selected_model_name, latitude, longitude)
                if st.session_state.get("inspection_key") != inspection_key:
                    with st.spinner("Inspecting selected location…"):
                        try:
                            inspection_image = (
                                MODEL_IMAGES[selected_model_name].rename("Carbon_t_ha")
                                .addBands(
                                    MODEL_IMAGES[selected_model_name]
                                    .divide(CARBON_TO_BIOMASS_FACTOR)
                                    .rename("AGB_Mg_ha")
                                )
                                .addBands(model_spread.rename("Model_spread"))
                                .addBands(
                                    predictor_variables.select(
                                        ["NDVI", "canopy_height", "elevation"]
                                    )
                                )
                            )
                            st.session_state["inspection_values"] = inspection_image.reduceRegion(
                                reducer=ee.Reducer.first(),
                                geometry=ee.Geometry.Point([longitude, latitude]),
                                scale=300, maxPixels=1e7,
                            ).getInfo()
                            st.session_state["inspection_key"] = inspection_key
                        except Exception as _e:
                            st.warning(f"Could not inspect this point: {_e}")

                values = st.session_state.get("inspection_values")
                if values and st.session_state.get("inspection_key") == inspection_key:
                    st.markdown(f"**Map inspector** · {latitude:.5f}, {longitude:.5f}")
                    i1, i2, i3 = st.columns(3)
                    i1.metric("Carbon", f"{float(values.get('Carbon_t_ha', 0)):.1f} t C/ha")
                    i2.metric("AGB", f"{float(values.get('AGB_Mg_ha', 0)):.1f} Mg/ha")
                    i3.metric("Model spread", f"{float(values.get('Model_spread', 0)):.1f} t C/ha")
                    st.caption(
                        f"NDVI: {float(values.get('NDVI', 0)):.3f} · "
                        f"Canopy height: {float(values.get('canopy_height', 0)):.1f} m · "
                        f"Elevation: {float(values.get('elevation', 0)):.0f} m"
                    )

# ── TAB: MODEL COMPARISON ────────────────────────────────────────────────────
with tab_briefing:
    st.markdown('<div class="sec-hdr"><i class="fa-solid fa-clipboard-check"></i>Automated Map Briefing</div>', unsafe_allow_html=True)
    st.markdown(
        "Generate a plain-language explanation of the selected counties, strongest tested model, "
        "uncertainty, county hotspots, and important cautions."
    )
    if st.button("Generate map briefing", type="primary", key="btn_map_briefing"):
        st.session_state["map_briefing"] = build_map_briefing(
            p,
            validation_results=st.session_state.get("validation_results"),
            mean_spread=st.session_state.get("mean_spread"),
            zonal_df=st.session_state.get("zonal_df"),
        )
    briefing = st.session_state.get("map_briefing")
    if briefing:
        st.success("Briefing generated. Read it below or download it as a text file.")
        st.text_area(
            "Your plain-language map briefing", value=briefing, height=430,
            disabled=True, key="generated_map_briefing",
        )
        st.download_button(
            "Download briefing as text", briefing.encode("utf-8"),
            "map_briefing.txt", "text/plain", key="download_map_briefing",
        )
    else:
        st.info(
            "For the fullest briefing, first compute validation metrics, mean model spread, and zonal statistics. "
            "You can still generate a briefing now; it will identify what is missing."
        )

with tab_guide:
    st.markdown('<div class="sec-hdr"><i class="fa-solid fa-graduation-cap"></i>Ask the Environmental &amp; Carbon Guide</div>', unsafe_allow_html=True)
    st.markdown(
        "Ask in everyday language about this analysis, climate change, emissions, carbon credits, "
        "carbon markets, forests, or practical climate action. The guide sees the active app settings and results."
    )

    try:
        stored_gemini_key = st.secrets.get("ai", {}).get("gemini_api_key", "")
    except Exception:
        stored_gemini_key = ""

    if not stored_gemini_key:
        st.info(
            "**To enable full AI chat:** open **Guide connection settings** below and paste a Gemini key for this session, "
            "or add it to Streamlit secrets. Until then, common questions use the built-in offline guide."
        )

    with st.expander("Guide connection settings", expanded=not bool(stored_gemini_key)):
        guide_language = st.radio(
            "Explanation language", ["English", "Kiswahili", "English + Kiswahili"], horizontal=True,
        )
        guide_provider = st.radio("Provider", ["Gemini", "Ollama (local)"], horizontal=True)
        gemini_model = st.text_input("Gemini model", value="gemini-3.6-flash")
        ollama_model = st.text_input("Ollama model", value="llama3.2")
        session_gemini_key = st.text_input(
            "Gemini API key (session only)", type="password",
            help="Not saved to the project. Prefer Streamlit secrets for a deployed app.",
        )
        use_ollama_fallback = st.checkbox("Use Ollama if Gemini is unavailable", value=True)
        st.caption(
            "For deployment, set `ai.gemini_api_key` in Streamlit secrets. "
            "For local fallback, install Ollama, run it, and pull the chosen model."
        )

    if "learning_chat" not in st.session_state:
        st.session_state["learning_chat"] = []
    clear_chat_col, prompt_col = st.columns([1, 4])
    with clear_chat_col:
        if st.button("Clear chat", key="clear_learning_chat"):
            st.session_state["learning_chat"] = []
            st.rerun()
    with prompt_col:
        st.caption("The guide explains estimates; it does not replace field verification, project certification, or local expertise.")

    quick_questions = st.columns(4)
    quick_prompts = [
        "Explain the map colours and legend to a complete beginner.",
        "What is the difference between carbon stock and AGB?",
        "How should I interpret the validation and uncertainty results?",
        "What is a carbon credit, and how is it different from reducing emissions?",
    ]
    for column, quick_prompt in zip(quick_questions, quick_prompts):
        with column:
            if st.button(quick_prompt.split("?")[0][:36], key=f"guide_{quick_prompts.index(quick_prompt)}"):
                st.session_state["pending_guide_question"] = quick_prompt

    for message in st.session_state["learning_chat"]:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    latest_answer = next(
        (message["content"] for message in reversed(st.session_state["learning_chat"])
         if message["role"] == "assistant"),
        None,
    )
    if latest_answer:
        render_voice_player(latest_answer, guide_language)

    st.markdown('<div class="guide-question-card">', unsafe_allow_html=True)
    st.markdown("**Ask the guide**")
    st.caption("Try: “What makes a carbon credit trustworthy?” or “How do I read this map?”")
    with st.form("learning_question_form", border=False, clear_on_submit=True):
        typed_question = st.text_area(
            "Your question", height=85,
            placeholder="Write your question here. You can ask about climate change, emissions, carbon markets, maps, or the environment.",
            label_visibility="collapsed",
        )
        submitted_question = st.form_submit_button("Ask the guide", type="primary", width="content")
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("**Ask by voice**")
    st.caption(
        "1. Click the microphone, allow access, record a short question, then stop it. "
        "2. Click **Ask recorded question**. Voice transcription uses Gemini."
    )
    recorded_question = st.audio_input(
        "Record a question", sample_rate=None, key="guide_voice_recording",
        help="Uses your browser's native microphone sample rate for better device compatibility.",
    )
    uploaded_voice_question = st.file_uploader(
        "If recording fails, upload a WAV recording instead", type=["wav"], key="guide_voice_upload",
        help="This fallback is useful when a browser or embedded preview blocks microphone access.",
    )
    if uploaded_voice_question is not None:
        st.audio(uploaded_voice_question, format="audio/wav")
    st.caption(
        "If the recorder shows an error, open the app in Chrome or Edge (not the VS Code embedded preview), "
        "allow microphone access for localhost, then try again."
    )
    ask_recorded_question = st.button("Ask recorded question", key="ask_recorded_question", type="primary")

    question = (
        typed_question.strip() if submitted_question and typed_question.strip() else
        st.session_state.pop("pending_guide_question", None)
    )
    voice_bytes = None
    voice_mime_type = "audio/wav"
    if ask_recorded_question:
        selected_voice_question = (
            recorded_question if recorded_question is not None else uploaded_voice_question
        )
        if selected_voice_question is None:
            st.warning("Record a question or upload a WAV file first, then click Ask recorded question.")
            question = None
        else:
            question = "Please listen to my recorded question and answer it as a beginner-friendly environmental guide."
            voice_bytes = selected_voice_question.getvalue()
            voice_mime_type = getattr(selected_voice_question, "type", "audio/wav") or "audio/wav"
    if question:
        st.session_state["learning_chat"].append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)
        with st.chat_message("assistant"):
            with st.spinner("The guide is thinking…"):
                try:
                    active_key = session_gemini_key or stored_gemini_key
                    # Ollama fallback is text-only. A voice turn must go to Gemini
                    # so that the audio can be transcribed before it is answered.
                    active_provider = "Gemini" if voice_bytes else guide_provider
                    active_model = gemini_model if active_provider == "Gemini" else ollama_model
                    answer = ask_learning_guide(
                        active_provider, active_model, question,
                        st.session_state["learning_chat"][:-1], build_learning_context(p), active_key,
                        language=guide_language, audio_bytes=voice_bytes, audio_mime_type=voice_mime_type,
                    )
                except Exception as primary_error:
                    if guide_provider == "Gemini" and use_ollama_fallback and not voice_bytes:
                        try:
                            answer = ask_learning_guide(
                                "Ollama (local)", ollama_model, question,
                                st.session_state["learning_chat"][:-1], build_learning_context(p),
                                language=guide_language,
                            )
                            answer = "*Gemini was unavailable, so this answer came from local Ollama.*\n\n" + answer
                        except Exception as fallback_error:
                            answer = (
                                "*The online guide is not connected, so this is a built-in explanation. "
                                "Open Guide connection settings to enable full AI chat.*\n\n"
                                + offline_learning_response(question)
                            )
                    else:
                        answer = (
                            "*The online guide is not connected, so this is a built-in explanation.*\n\n"
                            + offline_learning_response(question)
                        )
            st.markdown(answer)
        st.session_state["learning_chat"].append({"role": "assistant", "content": answer})

with tab_compare:
    st.markdown(
        '<div class="sec-hdr"><i class="fa-solid fa-right-left"></i>Model Comparison &amp; Agreement</div>',
        unsafe_allow_html=True,
    )

    # ── Agreement metric (on demand) ─────────────────────────────────────────
    if st.button("Compute mean model spread across study area"):
        with st.spinner("Reducing model spread on Earth Engine…"):
            try:
                spread_result = model_spread.reduceRegion(
                    reducer=ee.Reducer.mean(),
                    geometry=selected_fc.geometry(),
                    scale=5000,
                    maxPixels=1e9,
                    tileScale=16,
                    bestEffort=True,
                ).getInfo()
                st.session_state["mean_spread"] = spread_result.get("Model_Spread")
            except Exception as _e:
                st.error(
                    f"Could not compute model spread ({_e}). "
                    "Try selecting fewer counties or reducing sample pixels."
                )

    mean_spread = st.session_state.get("mean_spread")
    if mean_spread is not None:
        ma, mb = st.columns(2)
        ma.metric("Mean model spread (±σ, t C/ha)",
                  f"{mean_spread:.2f}",
                  help="Pixel-wise standard deviation across RF, GTB, and SVM predictions, averaged over the study area. Lower = higher agreement.")
        mb.metric("Coefficient of variation",
                  f"{mean_spread / 50 * 100:.1f}%",
                  help="Spread relative to a typical carbon stock of 50 t C/ha.")

    st.markdown("---")

    # ── RF vs GTB difference map ─────────────────────────────────────────────
    st.markdown("**RF vs. GTB — Absolute Difference**")
    if st.button("Render RF vs GTB difference map", key="btn_diff_map"):
        st.session_state["show_diff_map"] = True
    if st.session_state.get("show_diff_map"):
        carbon_difference = (
            estimated_carbon_rf.subtract(estimated_carbon_gtb).abs().rename("Carbon_Difference_RF_GTB")
        )
        with st.spinner("Rendering difference map…"):
            m2 = geemap.Map(center=[0.3, 36.0], zoom=7)
            safe_center_map(m2, selected_fc.geometry(), 9)
            safe_add_layer(m2, carbon_difference.clip(_display_geom), VIS_PARAMS_DIFF, "|RF − GTB| (t C/ha)", True)
            m2.add_colorbar(VIS_PARAMS_DIFF, label="|RF − GTB| (t C/ha)")
            render_gee_map(m2, height=420)
        st.caption("Larger values = greater disagreement between Random Forest and Gradient Tree Boosting.")

    st.markdown("---")

    # ── 3-model spread map ───────────────────────────────────────────────────
    st.markdown("**3-Model Spread — Pixel-wise Standard Deviation (RF, GTB, SVM)**")
    if st.button("Render 3-model spread map", key="btn_spread_map"):
        st.session_state["show_spread_map"] = True
    if st.session_state.get("show_spread_map"):
        with st.spinner("Rendering spread map…"):
            m3 = geemap.Map(center=[0.3, 36.0], zoom=7)
            safe_center_map(m3, selected_fc.geometry(), 9)
            safe_add_layer(m3, model_spread.clip(_display_geom), VIS_PARAMS_SPREAD, "Model Spread σ (t C/ha)", True)
            m3.add_colorbar(VIS_PARAMS_SPREAD, label="σ (t C/ha)")
            render_gee_map(m3, height=420)
        st.caption(
            "Per-pixel standard deviation across all three model predictions. "
            "Dark blue = high uncertainty; light = strong model agreement."
        )

# ── TAB: VALIDATION ──────────────────────────────────────────────────────────
with tab_restoration:
    st.markdown('<div class="sec-hdr"><i class="fa-solid fa-seedling"></i>Restoration Scenario Map</div>', unsafe_allow_html=True)
    st.markdown(
        "Create a screening scenario for selected counties or a hand-drawn site. It estimates potential biomass, carbon, "
        "CO₂e, and model uncertainty over time; it is not a project design or crediting calculation."
    )
    scenario_col, scenario_map_col = st.columns([1, 2], gap="medium")
    with scenario_col:
        scenario_area_mode = st.radio("Area to assess", ["Selected counties", "Draw an area on the map"])
        restoration_type = st.selectbox(
            "Restoration type",
            ["Assisted natural regeneration", "Agroforestry", "Native tree planting", "Mangrove/coastal restoration"],
        )
        default_growth = {
            "Assisted natural regeneration": 4.0,
            "Agroforestry": 3.0,
            "Native tree planting": 5.0,
            "Mangrove/coastal restoration": 6.0,
        }[restoration_type]
        annual_growth = st.number_input(
            "Expected biomass gain (Mg/ha/year)", min_value=0.0, value=default_growth, step=0.5,
            help="An adjustable planning assumption, not a prediction from this app.",
        )
        scenario_years = st.slider("Restoration period (years)", 1, 30, 10)
        calculate_scenario = st.button("Calculate restoration scenario", type="primary", key="calculate_restoration")

    with scenario_map_col:
        restoration_map = geemap.Map(center=[0.3, 36.0], zoom=7, plugin_Draw=True)
        safe_center_map(restoration_map, selected_fc.geometry(), 8)
        safe_add_layer(
            restoration_map, estimated_carbon_ensemble.clip(_display_geom), VIS_PARAMS_CARBON,
            "Current ensemble carbon stock", True,
        )
        if scenario_area_mode == "Selected counties":
            safe_add_layer(restoration_map, selected_fc, {"color": "#ff4444"}, "Selected counties", True)
        elif st.session_state.get("restoration_draw_geometry"):
            safe_add_layer(
                restoration_map, ee.Geometry(st.session_state["restoration_draw_geometry"]),
                {"color": "#ff4444"}, "Drawn restoration area", True,
            )
        draw_event = render_gee_map(restoration_map, height=500, bidirectional=True)
        latest_drawing = (draw_event or {}).get("last_active_drawing")
        if latest_drawing and latest_drawing.get("geometry", {}).get("type") in {"Polygon", "MultiPolygon"}:
            st.session_state["restoration_draw_geometry"] = latest_drawing["geometry"]
            st.success("Custom restoration area saved. You can now calculate the scenario.")
        elif scenario_area_mode == "Draw an area on the map":
            st.caption("Use the polygon tool in the map's upper-left corner to draw a site, then wait for the confirmation message.")

    if calculate_scenario:
        if scenario_area_mode == "Selected counties":
            scenario_geometry = selected_fc.geometry()
        else:
            drawn_geometry = st.session_state.get("restoration_draw_geometry")
            if not drawn_geometry:
                scenario_geometry = None
                st.warning("Draw a polygon on the map before calculating a custom-area scenario.")
            else:
                scenario_geometry = ee.Geometry(drawn_geometry)
        if scenario_geometry is not None:
            with st.spinner("Calculating restoration screening scenario…"):
                try:
                    area_ha = ee.Number(scenario_geometry.area()).divide(10000).getInfo()
                    baseline_carbon = estimated_carbon_ensemble.reduceRegion(
                        reducer=ee.Reducer.mean(), geometry=scenario_geometry, scale=1000,
                        maxPixels=1e10, tileScale=16, bestEffort=True,
                    ).getInfo().get("Estimated Carbon Stock Ensemble")
                    uncertainty = model_spread.reduceRegion(
                        reducer=ee.Reducer.mean(), geometry=scenario_geometry, scale=1000,
                        maxPixels=1e10, tileScale=16, bestEffort=True,
                    ).getInfo().get("Model_Spread")
                    added_biomass_per_ha = annual_growth * scenario_years
                    added_carbon_per_ha = added_biomass_per_ha * CARBON_TO_BIOMASS_FACTOR
                    total_biomass = (area_ha or 0) * added_biomass_per_ha
                    total_carbon = (area_ha or 0) * added_carbon_per_ha
                    total_co2e = total_carbon * (44 / 12)
                    st.session_state["restoration_scenario"] = {
                        "area_ha": area_ha or 0, "baseline_carbon": baseline_carbon,
                        "uncertainty": uncertainty, "added_biomass_per_ha": added_biomass_per_ha,
                        "added_carbon_per_ha": added_carbon_per_ha, "total_biomass": total_biomass,
                        "total_carbon": total_carbon, "total_co2e": total_co2e,
                        "restoration_type": restoration_type, "years": scenario_years,
                    }
                except Exception as _e:
                    st.error(f"Could not calculate the restoration scenario: {_e}")

    scenario = st.session_state.get("restoration_scenario")
    if scenario:
        st.markdown("#### Scenario results")
        r1, r2, r3, r4 = st.columns(4)
        r1.metric("Area", f"{scenario['area_ha']:,.1f} ha")
        r2.metric("Potential biomass gain", f"{scenario['total_biomass']:,.0f} Mg")
        r3.metric("Potential carbon gain", f"{scenario['total_carbon']:,.0f} t C")
        r4.metric("Indicative CO₂e", f"{scenario['total_co2e']:,.0f} t CO₂e")
        baseline_text = "not available" if scenario["baseline_carbon"] is None else f"{float(scenario['baseline_carbon']):.1f} t C/ha"
        uncertainty_text = "not available" if scenario["uncertainty"] is None else f"± {float(scenario['uncertainty']):.1f} t C/ha"
        st.info(
            f"**{scenario['restoration_type']}** over **{scenario['years']} years**. Current mean carbon: {baseline_text}; "
            f"mean model uncertainty: {uncertainty_text}. The growth assumption is {scenario['added_biomass_per_ha']:.1f} Mg/ha "
            f"({scenario['added_carbon_per_ha']:.1f} t C/ha) over the full scenario. Confirm land tenure, survival, species suitability, baseline, "
            "leakage, and permanence before making any project or carbon-credit claim."
        )

with tab_tools:
    st.markdown('<div class="sec-hdr"><i class="fa-solid fa-compass"></i>Decision Tools</div>', unsafe_allow_html=True)
    change_col, calculator_col = st.columns(2)
    with change_col:
        st.markdown("#### Carbon change explorer")
        change_years = st.select_slider(
            "Compare reference years", options=AGB_YEARS_AVAIL,
            value=("2010", "2022"), key="change_years",
        )
        if st.button("Render carbon change map", key="btn_change_map"):
            st.session_state["show_change_map"] = True
        if st.session_state.get("show_change_map"):
            start_carbon = (
                ee.ImageCollection(ESA_AGB_COLLECTION)
                .filterDate(f"{change_years[0]}-01-01", f"{change_years[0]}-12-31")
                .first().select(ESA_AGB_BAND).multiply(CARBON_TO_BIOMASS_FACTOR)
            )
            end_carbon = (
                ee.ImageCollection(ESA_AGB_COLLECTION)
                .filterDate(f"{change_years[1]}-01-01", f"{change_years[1]}-12-31")
                .first().select(ESA_AGB_BAND).multiply(CARBON_TO_BIOMASS_FACTOR)
            )
            change_image = end_carbon.subtract(start_carbon).rename("Carbon_change")
            change_map = geemap.Map(center=[0.3, 36.0], zoom=7)
            safe_center_map(change_map, selected_fc.geometry(), 9)
            safe_add_layer(
                change_map, change_image.clip(_display_geom),
                {"min": -50, "max": 50, "palette": ["#b2182b", "#f7f7f7", "#1a9850"]},
                f"Carbon change {change_years[0]} to {change_years[1]} (t C/ha)", True,
            )
            render_gee_map(change_map, height=380)
            st.caption("Red indicates a lower reference carbon estimate; green indicates a higher estimate.")

    with calculator_col:
        st.markdown("#### Carbon-project calculator")
        project_area = st.number_input("Project area (ha)", min_value=1.0, value=100.0, step=10.0)
        annual_biomass_gain = st.number_input(
            "Expected annual biomass gain (Mg/ha/year)", min_value=0.0, value=5.0, step=0.5,
        )
        project_years = st.slider("Project duration (years)", 1, 30, 10)
        projected_co2e = project_area * annual_biomass_gain * project_years * CARBON_TO_BIOMASS_FACTOR * (44 / 12)
        st.metric("Indicative sequestration", f"{projected_co2e:,.0f} t CO₂e")
        st.caption("Screening estimate only: excludes baseline, leakage, permanence, and project deductions.")

    st.markdown("#### County carbon watchlist")
    st.caption(
        f"Latest available ESA reference map: **{AGB_YEARS_AVAIL[-1]}**. Save counties, then compare two available years to flag meaningful change."
    )
    watch_col1, watch_col2 = st.columns([2, 1])
    with watch_col1:
        watch_defaults = [
            county for county in st.session_state.get("watched_counties", p["county_selection"])
            if county in p["county_selection"]
        ]
        watched_counties = st.multiselect(
            "Counties to watch", options=list(p["county_selection"]),
            default=watch_defaults,
            key="watchlist_counties",
        )
    with watch_col2:
        if st.button("Save watchlist", key="save_watchlist"):
            st.session_state["watched_counties"] = watched_counties
            st.success(f"Saved {len(watched_counties)} county/counties for this session.")

    watch_years = st.select_slider(
        "Reference years to compare", options=AGB_YEARS_AVAIL,
        value=(AGB_YEARS_AVAIL[0], AGB_YEARS_AVAIL[-1]), key="watchlist_years",
    )
    watch_threshold = st.slider(
        "Meaningful change threshold (t C/ha)", 1, 50, 10, step=1,
        help="A county is flagged only when its mean reference carbon estimate changes by at least this amount.",
        key="watchlist_threshold",
    )
    if st.button("Check watchlist for change", type="primary", key="check_watchlist"):
        if not watched_counties:
            st.warning("Choose at least one county to watch.")
        else:
            with st.spinner("Comparing available ESA reference maps for watched counties…"):
                try:
                    st.session_state["watchlist_alerts"] = compute_watchlist_alerts(
                        watched_counties, watch_years[0], watch_years[1], watch_threshold
                    )
                    st.session_state["watchlist_check"] = {
                        "baseline": watch_years[0], "comparison": watch_years[1], "threshold": watch_threshold,
                    }
                except Exception as _e:
                    st.error(f"Could not check the watchlist: {_e}")
    watchlist_alerts = st.session_state.get("watchlist_alerts")
    if watchlist_alerts is not None:
        flagged = watchlist_alerts[watchlist_alerts["Status"] != "No material change"]
        if flagged.empty:
            st.success("No watched county crossed the selected meaningful-change threshold.")
        else:
            st.warning(f"{len(flagged)} watched county/counties crossed the selected threshold.")
        st.dataframe(watchlist_alerts, width="stretch", hide_index=True)
        st.download_button(
            "Download watchlist alerts", watchlist_alerts.to_csv(index=False).encode("utf-8"),
            "county_carbon_watchlist.csv", "text/csv", key="download_watchlist_alerts",
        )
    st.info(
        "This is an on-demand watchlist, not an automatic notification service. To send email or scheduled alerts, "
        "the deployed app needs a database plus a scheduler or notification provider."
    )

    st.markdown("#### Field-data validation")
    field_file = st.file_uploader("Upload CSV plot data", type="csv", key="field_data_file")
    if field_file is not None:
        field_df = pd.read_csv(field_file)
        st.caption("Required columns: `longitude`, `latitude`, and `observed_carbon` (t C/ha). Up to 500 rows are sampled.")
        if {"longitude", "latitude", "observed_carbon"}.issubset(field_df.columns):
            field_model = st.selectbox("Model to validate", list(MODEL_IMAGES), key="field_model")
            if st.button("Validate uploaded plots", key="btn_field_validation"):
                with st.spinner("Sampling model predictions at plot locations…"):
                    try:
                        records = field_df[["longitude", "latitude", "observed_carbon"]].dropna().head(500)
                        features = [
                            ee.Feature(ee.Geometry.Point([row.longitude, row.latitude]), {
                                "observed_carbon": float(str(row.observed_carbon))
                            })
                            for row in records.itertuples(index=False)
                        ]
                        sampled = MODEL_IMAGES[field_model].rename("prediction").sampleRegions(
                            collection=ee.FeatureCollection(features), properties=["observed_carbon"], scale=300
                        ).getInfo()["features"]
                        validation_rows = [
                            {"Observed": item["properties"].get("observed_carbon"),
                             "Predicted": item["properties"].get("prediction")}
                            for item in sampled
                        ]
                        field_results = pd.DataFrame(validation_rows, columns=["Observed", "Predicted"]).dropna()
                        if field_results.empty:
                            st.warning(
                                "None of the uploaded points could be sampled. Ensure the points fall inside "
                                "the selected counties and tree-cover prediction mask, then try again."
                            )
                        else:
                            field_results = field_results.apply(pd.to_numeric, errors="coerce").dropna()
                            field_results["Error"] = field_results["Predicted"] - field_results["Observed"]
                            field_rmse = np.sqrt(np.mean(field_results["Error"] ** 2))
                            st.session_state["field_validation_results"] = (field_results, field_rmse)
                    except Exception as _e:
                        st.error(f"Field validation failed: {_e}")
        else:
            st.warning("The uploaded CSV does not have the required column names.")
    if st.session_state.get("field_validation_results"):
        field_results, field_rmse = st.session_state["field_validation_results"]
        st.metric("Field-plot RMSE", f"{field_rmse:.2f} t C/ha")
        st.dataframe(field_results, width="stretch")

with tab_validation:
    st.markdown('<div class="sec-hdr"><i class="fa-solid fa-chart-bar"></i>Validation on Held-Out Testing Set</div>', unsafe_allow_html=True)
    st.markdown(
        """
        <div class="validation-guide">
            <strong>How to use validation</strong><br>
            <b>1.</b> Click <b>Compute validation metrics</b> below. The app tests each model using
            sample locations it did not use for training.<br>
            <b>2.</b> Compare the three model rows: lower <b>RMSE</b> and <b>MAE</b> mean smaller
            prediction errors; higher <b>R²</b> (closer to 1) means the model explains more of the
            observed variation.<br>
            <b>3.</b> Treat the model with the lowest errors as the stronger option for this run,
            then inspect its scatter plot. Points close to the red diagonal line are better predictions.
            <br><br><b>Kwa Kiswahili:</b> RMSE na MAE za chini zinaonyesha makosa madogo; R² iliyo karibu na 1 inaonyesha
            modeli inaeleza data vizuri zaidi. Angalia nukta zilizo karibu na mstari mwekundu kwenye mchoro.
        </div>
        """,
        unsafe_allow_html=True,
    )

    if st.button("1. Compute validation metrics", type="primary", width='content'):
        results = {}
        failed = []
        for model_name, model_key in [
            ("Random Forest", "rf_model"),
            ("Gradient Tree Boosting", "gtb_model"),
            ("Support Vector Machine", "svm_model"),
        ]:
            with st.spinner(f"Computing {model_name} metrics…"):
                try:
                    results[model_name] = compute_validation_metrics(
                        sample["testing_set"], models[model_key], sample["dependent_variable"]
                    )
                except Exception as _e:
                    failed.append(f"{model_name}: {_e}")
        if failed:
            st.error("Some models failed validation: " + "; ".join(failed))
        if results:
            st.session_state["validation_results"] = results

    results = st.session_state.get("validation_results")
    if results:
        metrics_df = pd.DataFrame(
            {name: {"RMSE": r["rmse"], "MAE": r["mae"], "R²": r["r2"]}
             for name, r in results.items()}
        ).T
        # Earth Engine may return a numeric value as text (or null for a failed
        # correlation). Coerce before applying a numeric display format.
        metrics_df = metrics_df.apply(pd.to_numeric, errors="coerce")
        st.dataframe(metrics_df.style.format("{:.3f}", na_rep="—"), width='stretch')
        if metrics_df["RMSE"].notna().any():
            best_model = metrics_df["RMSE"].idxmin()
            best_rmse = metrics_df.loc[best_model, "RMSE"]
            best_r2 = metrics_df.loc[best_model, "R²"]
            st.success(
                f"Validation snapshot: **{best_model}** has the lowest RMSE "
                f"({best_rmse:.3f}); R² = {best_r2:.3f}" if pd.notna(best_r2)
                else f"Validation snapshot: **{best_model}** has the lowest RMSE ({best_rmse:.3f})."
            )
        st.markdown("---")

        scatter_colors = {
            "Random Forest":          "#2d6a4f",
            "Gradient Tree Boosting": "#40916c",
            "Support Vector Machine": "#6a2d6a",
        }
        cols = st.columns(3)
        for col, (name, r) in zip(cols, results.items()):
            with col:
                fig = make_scatter_plot(r["actual"], r["predicted"], name, scatter_colors[name])
                st.pyplot(fig, width='stretch')
    else:
        st.markdown("""
        <div class="welcome" style="padding:2rem;">
            <i class="fa-solid fa-chart-bar icon" style="font-size:2rem;"></i>
            <h3 style="font-size:1rem;">No results yet</h3>
            <p style="font-size:0.88rem;">Start with <strong>1. Compute validation metrics</strong> above.
            This calls Earth Engine and may take a minute.</p>
        </div>
        """, unsafe_allow_html=True)

# ── TAB: ZONAL STATISTICS ────────────────────────────────────────────────────
with tab_quality:
    st.markdown('<div class="sec-hdr"><i class="fa-solid fa-circle-check"></i>Run Data-Quality Score</div>', unsafe_allow_html=True)
    st.markdown(
        "This is an explainable screening score for the current run. It helps identify missing checks before you interpret "
        "or compare results; it is not a scientific accuracy certificate."
    )
    if st.button("Assess run data quality", type="primary", key="assess_data_quality"):
        st.session_state["data_quality_assessment"] = assess_data_quality(
            p,
            validation_results=st.session_state.get("validation_results"),
            mean_spread=st.session_state.get("mean_spread"),
        )
    assessment = st.session_state.get("data_quality_assessment")
    if assessment:
        q1, q2 = st.columns([1, 3])
        with q1:
            st.metric("Screening score", f"{assessment['total']} / 100")
        with q2:
            st.markdown(f"### {assessment['label']}")
            st.caption("Complete missing checks and review the notes below before treating outputs as decision support.")
        component_rows = pd.DataFrame(
            [
                {"Check": name, "Score (out of 25)": score, "What it means": note}
                for name, score, note in assessment["components"]
            ]
        )
        st.dataframe(component_rows, width="stretch", hide_index=True)
        st.info(
            "**Coverage note:** geographic coverage is estimated from samples per selected county. It does not prove that "
            "every land-cover type or remote location is represented; inspect maps and field data for that."
        )
    else:
        st.info(
            "For a complete score, first run Validation and compute Model Comparison → mean model spread. "
            "You can still assess the run now to see what is missing."
        )

with tab_zonal:
    st.markdown('<div class="sec-hdr"><i class="fa-solid fa-location-dot"></i>Per-County Zonal Statistics</div>', unsafe_allow_html=True)

    zonal_model_choice = st.selectbox("Model for zonal stats:", list(MODEL_IMAGES.keys()),
                                      key="zonal_model")
    if st.button("Compute zonal statistics"):
        with st.spinner("Reducing regions on Earth Engine…"):
            try:
                zonal = MODEL_IMAGES[zonal_model_choice].reduceRegions(
                    collection=selected_fc,
                    reducer=(
                        ee.Reducer.mean()
                        .combine(ee.Reducer.min(), sharedInputs=True)
                        .combine(ee.Reducer.max(), sharedInputs=True)
                        .combine(ee.Reducer.sum(), sharedInputs=True)
                    ),
                    scale=2000,
                    tileScale=16,
                )
                features = zonal.getInfo()["features"]
                rows = [
                    {
                        "County":        f["properties"].get("shapeName", "Unknown"),
                        "Mean (t/ha)":   f["properties"].get("mean"),
                        "Min (t/ha)":    f["properties"].get("min"),
                        "Max (t/ha)":    f["properties"].get("max"),
                        "Sum (t)":       f["properties"].get("sum"),
                    }
                    for f in features
                ]
                zonal_df = pd.DataFrame(rows).sort_values("Mean (t/ha)", ascending=False)
                st.session_state["zonal_df"] = zonal_df
            except Exception as _e:
                st.error(
                    f"Zonal statistics failed ({_e}). "
                    "Try selecting fewer counties or choosing a single-model output instead of Ensemble."
                )

    zonal_df = st.session_state.get("zonal_df")
    if zonal_df is not None:
        st.dataframe(zonal_df, width='stretch')

        ranking_basis = st.radio(
            "County ranking", ["Mean carbon stock", "Total carbon stock"], horizontal=True,
            key="county_ranking_basis",
        )
        ranking_column = "Mean (t/ha)" if ranking_basis == "Mean carbon stock" else "Sum (t)"
        ranked_counties = zonal_df.sort_values(ranking_column, ascending=False).reset_index(drop=True).copy()
        ranked_counties.insert(0, "Rank", ranked_counties.index + 1)
        st.dataframe(ranked_counties[["Rank", "County", ranking_column]], width="stretch", hide_index=True)

        fig_z, ax_z = plt.subplots(figsize=(9, 3.5))
        counties = zonal_df["County"].tolist()
        means    = zonal_df["Mean (t/ha)"].tolist()
        bars = ax_z.barh(counties[::-1], means[::-1],
                         color="#52b788", edgecolor="none", height=0.65)
        ax_z.bar_label(bars, fmt="%.1f", padding=4, fontsize=8, color="#2d6a4f")
        ax_z.set_xlabel("Mean Carbon Stock (t C/ha)")
        ax_z.set_title(f"Mean Estimated Carbon Stock by County — {zonal_model_choice}")
        ax_z.set_xlim(0, max(means) * 1.15)
        fig_z.tight_layout()
        st.pyplot(fig_z, width='stretch')

        csv = zonal_df.to_csv(index=False).encode("utf-8")
        st.download_button("Download as CSV", csv, "zonal_statistics.csv", "text/csv")
    else:
        st.markdown("""
        <div class="welcome" style="padding:2rem;">
            <i class="fa-solid fa-location-dot icon" style="font-size:2rem;"></i>
            <h3 style="font-size:1rem;">No data yet</h3>
            <p style="font-size:0.88rem;">Click <strong>Compute zonal statistics</strong> to
            aggregate estimated carbon stock per county.</p>
        </div>
        """, unsafe_allow_html=True)

# ── TAB: VARIABLE IMPORTANCE ─────────────────────────────────────────────────
with tab_importance:
    st.markdown('<div class="sec-hdr"><i class="fa-solid fa-star"></i>Variable Importance</div>', unsafe_allow_html=True)

    imp_choice = st.radio("Model:", ["Random Forest", "Gradient Tree Boosting"], horizontal=True)
    importance_model = models["rf_model"] if imp_choice == "Random Forest" else models["gtb_model"]
    if st.button("Load variable importance", key="btn_variable_importance"):
        with st.spinner(f"Fetching {imp_choice} importance from Earth Engine…"):
            try:
                st.session_state.setdefault("importance_results", {})[imp_choice] = (
                    importance_model.explain().getInfo()
                )
            except Exception as _e:
                st.error(f"Variable importance could not be computed: {_e}")
    raw_importance = st.session_state.get("importance_results", {}).get(imp_choice)

    if raw_importance is None:
        st.info(
            "Variable importance could not be computed — the Earth Engine request timed out. "
            "Try reducing the number of trees or sample pixels and re-run."
        )
    else:
        importance_dict = raw_importance.get("importance", raw_importance)

    if raw_importance is not None and isinstance(importance_dict, dict):
        imp_df = (
            pd.DataFrame(list(importance_dict.items()), columns=["Variable", "Importance"])
            .sort_values("Importance", ascending=False)
        )
        top_n  = imp_df.head(20)
        palette = ["#1a472a", "#2d6a4f", "#40916c", "#52b788", "#74c69d",
                   "#95d5b2", "#b7e4c7", "#d8f3dc"]
        n_bars = len(top_n)
        colors = [palette[min(i // max(1, n_bars // len(palette)), len(palette) - 1)]
                  for i in range(n_bars)]

        fig_i, ax_i = plt.subplots(figsize=(9, max(4, n_bars * 0.38)))
        bars_i = ax_i.barh(
            top_n["Variable"].tolist()[::-1],
            top_n["Importance"].tolist()[::-1],
            color=colors[::-1], edgecolor="none", height=0.65,
        )
        ax_i.bar_label(bars_i, fmt="%.2f", padding=4, fontsize=8, color="#1a472a")
        ax_i.set_xlabel("Importance Score")
        ax_i.set_title(f"Top {n_bars} Variables — {imp_choice}")
        ax_i.set_xlim(0, top_n["Importance"].max() * 1.15)
        fig_i.tight_layout()
        st.pyplot(fig_i, width='stretch')

        with st.expander("Full importance table"):
            st.dataframe(imp_df, width='stretch')
    elif raw_importance is not None:
        st.json(raw_importance)

with tab_report:
    st.markdown('<div class="sec-hdr"><i class="fa-solid fa-file-export"></i>Report &amp; Export</div>', unsafe_allow_html=True)
    report_col, export_col = st.columns(2)
    with report_col:
        st.markdown("#### Shareable PDF summary")
        report_pdf = build_report_pdf(
            p, st.session_state.get("validation_results"), st.session_state.get("zonal_df")
        )
        st.download_button(
            "Download PDF report", report_pdf, "carbon_agb_summary.pdf", "application/pdf",
            width="stretch",
        )
        st.caption("Includes active settings and any validation or zonal results already computed.")

    with export_col:
        st.markdown("#### GeoTIFF export task")
        export_model = st.selectbox("Map to export", list(MODEL_IMAGES), key="export_model")
        export_scale = st.select_slider("Export scale (m)", options=[100, 300, 500, 1000], value=300)
        export_folder = st.text_input("Google Drive folder", value="EarthEngineExports")
        if st.button("Create GeoTIFF export task", key="btn_geotiff_export"):
            try:
                export_slug = "".join(
                    character.lower() if character.isalnum() else "_" for character in export_model
                ).strip("_")
                batch_api = getattr(ee, "batch")
                task = batch_api.Export.image.toDrive(
                    image=MODEL_IMAGES[export_model].clip(selected_fc.geometry()),
                    description=f"carbon_agb_{export_slug}",
                    folder=export_folder,
                    fileNamePrefix=f"carbon_agb_{p['agb_year']}_{export_slug}",
                    region=selected_fc.geometry(), scale=export_scale,
                    maxPixels=10_000_000_000_000, fileFormat="GeoTIFF",
                )
                task.start()
                st.success(f"Export task started: {task.id}")
            except Exception as _e:
                st.error(f"Could not create export task: {_e}")
        st.caption("The task runs in Earth Engine and writes to your connected Google Drive.")

    st.markdown("#### Project portfolio")
    st.caption("Save the current run as a named project snapshot, compare it with others in this browser session, and export one combined report.")
    portfolio_name_col, portfolio_save_col = st.columns([3, 1])
    with portfolio_name_col:
        portfolio_name = st.text_input("Project name", placeholder="e.g. Western Kenya restoration screening")
    with portfolio_save_col:
        st.write("")
        st.write("")
        save_portfolio_project = st.button("Save project", type="primary", key="save_portfolio_project")

    if save_portfolio_project:
        if not portfolio_name.strip():
            st.warning("Give this analysis a project name before saving it.")
        else:
            validation = st.session_state.get("validation_results", {})
            best_model, best_rmse = "Not validated", "—"
            valid_models = []
            for model_name, result in validation.items():
                try:
                    valid_models.append((model_name, float(result.get("rmse"))))
                except (TypeError, ValueError):
                    continue
            if valid_models:
                best_model, numeric_rmse = min(valid_models, key=lambda item: item[1])
                best_rmse = f"{numeric_rmse:.3f} t C/ha"
            spread = st.session_state.get("mean_spread")
            try:
                spread_label = f"{float(spread):.2f} t C/ha" if spread is not None else "Not computed"
            except (TypeError, ValueError):
                spread_label = "Not computed"
            snapshot = {
                "Project": portfolio_name.strip(),
                "Counties": f"{len(p['county_selection'])} county/counties",
                "Reference year": p["agb_year"],
                "Samples": f"{p['num_pixels']:,}",
                "Best model": best_model,
                "Best RMSE": best_rmse,
                "Mean spread": spread_label,
                "Configuration": json.loads(json.dumps(p)),
            }
            portfolio = st.session_state.setdefault("project_portfolio", [])
            portfolio[:] = [entry for entry in portfolio if entry["Project"] != snapshot["Project"]]
            portfolio.append(snapshot)
            st.success(f"Saved {snapshot['Project']} to this session's portfolio.")

    portfolio = st.session_state.get("project_portfolio", [])
    if portfolio:
        portfolio_df = pd.DataFrame(portfolio).drop(columns=["Configuration"], errors="ignore")
        st.dataframe(portfolio_df, width="stretch", hide_index=True)
        portfolio_pdf = build_portfolio_pdf(portfolio)
        portfolio_col1, portfolio_col2, portfolio_col3 = st.columns(3)
        with portfolio_col1:
            st.download_button(
                "Download combined PDF", portfolio_pdf, "carbon_project_portfolio.pdf", "application/pdf",
                key="download_portfolio_pdf", width="stretch",
            )
        with portfolio_col2:
            st.download_button(
                "Download portfolio JSON", json.dumps(portfolio, indent=2).encode("utf-8"),
                "carbon_project_portfolio.json", "application/json", key="download_portfolio_json", width="stretch",
            )
        with portfolio_col3:
            if st.button("Clear portfolio", key="clear_portfolio", width="stretch"):
                st.session_state["project_portfolio"] = []
                st.rerun()
    else:
        st.info("No projects saved yet. Run an analysis, compute any results you want to preserve, then save it here.")

    st.markdown("#### Compare saved scenarios")
    scenario_files = st.file_uploader(
        "Upload one or more saved run configurations", type="json", accept_multiple_files=True,
        key="scenario_files",
    )
    if scenario_files:
        scenario_rows = []
        for scenario_file in scenario_files:
            try:
                scenario = json.loads(scenario_file.getvalue().decode("utf-8"))
                scenario_rows.append({
                    "Scenario": scenario_file.name,
                    "Counties": len(scenario.get("county_selection", [])),
                    "Year": scenario.get("agb_year", "—"),
                    "Samples": scenario.get("num_pixels", "—"),
                    "RF trees": scenario.get("rf_trees", "—"),
                    "GTB trees": scenario.get("gtb_trees", "—"),
                })
            except Exception:
                st.warning(f"Could not read {scenario_file.name}.")
        if scenario_rows:
            st.dataframe(pd.DataFrame(scenario_rows), width="stretch", hide_index=True)

# ============================================================================
# FOOTER
# ============================================================================
st.markdown("""
<div class="app-footer">
    <strong>Data sources:</strong>
    ESA CCI AGB · Sentinel-1/2 (Copernicus) · Dynamic World · SRTM · WorldClim ·
    OpenLandMap Soil Organic Carbon · Meta Canopy Height · MODIS LST · JAXA ALOS PALSAR<br>
    Carbon-to-biomass conversion factor: <strong>0.47</strong> (IPCC default, Mg C / Mg dry biomass)
</div>
""", unsafe_allow_html=True)
