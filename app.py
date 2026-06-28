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
os.environ.setdefault("USE_FOLIUM", "1")  # ensure geemap.__init__ uses folium branch

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

import ee
import geemap.foliumap as geemap  # folium-backed geemap -> works in Streamlit

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
.stTabs [data-baseweb="tab-list"] {
    gap: 4px;
    background: #dceee0;
    border-radius: 12px;
    padding: 5px;
    border: none;
}
.stTabs [data-baseweb="tab"] {
    border-radius: 9px;
    padding: 0.45rem 1.1rem;
    font-weight: 600;
    font-size: 0.88rem;
    color: var(--text-muted) !important;
    background: transparent !important;
    border: none !important;
    transition: all 0.15s ease;
}
.stTabs [aria-selected="true"] {
    background: #ffffff !important;
    color: var(--green-dark) !important;
    box-shadow: 0 1px 5px var(--shadow) !important;
}
.stTabs [data-baseweb="tab-panel"] {
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
    "Nakuru", "Baringo", "Laikipia", "Trans Nzoia", "Kakamega", "Busia",
    "Meru", "Nandi", "Siaya", "Vihiga", "Nyandarua", "Tharaka", "Kericho",
]

DATE_START_S2    = "2020-01-01"
DATE_END_S2      = "2023-12-31"
AGB_YEARS_AVAIL  = ["2017", "2018", "2019", "2020"]

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
def init_earth_engine(project_id: str) -> tuple[bool, str]:
    try:
        # Service-account path: credentials stored in st.secrets (Streamlit Cloud deployment)
        if "gee" in st.secrets and "credentials" in st.secrets["gee"]:
            import json as _json
            cred_dict = _json.loads(st.secrets["gee"]["credentials"])
            credentials = ee.ServiceAccountCredentials(
                email=cred_dict["client_email"],
                key_data=st.secrets["gee"]["credentials"],
            )
            ee.Initialize(credentials=credentials, project=project_id)
            return True, "Earth Engine initialized via service account."

        # Local path: use credentials cached by `earthengine authenticate`
        ee.Initialize(project=project_id)
        return True, "Earth Engine initialized."
    except Exception:
        try:
            ee.Authenticate()
            ee.Initialize(project=project_id)
            return True, "Earth Engine authenticated and initialized."
        except Exception as e2:
            return False, f"Could not initialize Earth Engine: {e2}"


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
        ee.ImageCollection("projects/sat-io/open-datasets/ESA/ESA_CCI_AGB")
        .filterDate(f"{agb_year}-01-01", f"{agb_year}-12-31")
        .first()
        .select("AGB")
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

    landcover = (
        ee.ImageCollection("GOOGLE/DYNAMICWORLD/V1")
        .select("label")
        .filterDate(DATE_START_S2, DATE_END_S2)
        .filterBounds(geom)
        .mode()
        .eq(1)
    )
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
        region=selected_fc.geometry(), scale=100,
        numPixels=num_pixels, geometries=True, tileScale=16,
    )
    all_sampled  = all_sampled.randomColumn(seed=seed)
    training_set = all_sampled.filter(ee.Filter.lt("random", train_split))
    testing_set  = all_sampled.filter(ee.Filter.gte("random", train_split))

    return {
        "dependent_variable":   dependent_variable,
        "predictor_band_names": predictor_band_names,
        "training_set":  training_set,
        "testing_set":   testing_set,
        "n_total": all_sampled.size().getInfo(),
        "n_train": training_set.size().getInfo(),
        "n_test":  testing_set.size().getInfo(),
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
    rf_importance = rf_model.explain().getInfo()

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
    gtb_importance = gtb_model.explain().getInfo()

    return {
        "rf_model":  rf_model,  "svm_model": svm_model, "gtb_model": gtb_model,
        "rf_importance": rf_importance, "gtb_importance": gtb_importance,
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
    _ok, _ = init_earth_engine(_pid)
    if _ok:
        st.session_state["ee_ready"] = True
        st.session_state["ee_project_id"] = _pid

with st.sidebar.expander("Earth Engine Setup", expanded="ee_ready" not in st.session_state):
    if st.session_state.get("ee_ready") and "gee" in st.secrets:
        st.success("Connected via service account.")
    else:
        st.markdown(
            "Requires a **Google Earth Engine** account with a registered Cloud project. "
            "Run `earthengine authenticate` once in a terminal, then enter your project ID below."
        )
        default_project = st.session_state.get("ee_project_id", "")
        project_id = st.text_input("GEE Cloud Project ID", value=default_project,
                                   placeholder="my-gee-project")
        if st.button("Connect to Earth Engine", width='stretch'):
            with st.spinner("Connecting…"):
                ok, msg = init_earth_engine(project_id)
            if ok:
                st.session_state["ee_ready"] = True
                st.session_state["ee_project_id"] = project_id
                st.success(msg)
            else:
                st.session_state["ee_ready"] = False
                st.error(msg)

ee_ready = st.session_state.get("ee_ready", False)

st.sidebar.markdown("---")
st.sidebar.markdown("**Study Area**")
agb_year = st.sidebar.select_slider(
    "AGB reference year",
    options=AGB_YEARS_AVAIL,
    value="2020",
    help="ESA CCI Above-Ground Biomass dataset year used as the regression target",
)
county_selection = st.sidebar.multiselect(
    "Kenyan counties (ADM1)",
    options=ALL_KENYA_COUNTIES_OPTIONS,
    default=ALL_KENYA_COUNTIES_OPTIONS,
)

st.sidebar.markdown("**Sampling**")
num_pixels  = st.sidebar.slider("Sample pixels", 1000, 20000, 10000, step=1000)
train_split = st.sidebar.slider("Training split", 0.5, 0.9, 0.7, step=0.05,
                                 help="Fraction of pixels used for training")
seed        = st.sidebar.number_input("Random seed", value=0, step=1)

with st.sidebar.expander("Model Hyperparameters"):
    st.markdown("**Random Forest**")
    rf_trees          = st.slider("Trees", 50, 500, 300, step=10, key="rf_trees")
    rf_vars_per_split = st.slider("Variables per split", 1, 15, 6, key="rf_vars")
    rf_min_leaf       = st.slider("Min leaf population", 1, 50, 10, key="rf_leaf")

    st.markdown("**Gradient Tree Boosting**")
    gtb_trees         = st.slider("Trees", 50, 500, 300, step=10, key="gtb_trees")
    gtb_shrinkage     = st.slider("Shrinkage", 0.001, 0.1, 0.005, step=0.001,
                                   format="%.3f", key="gtb_shrink")
    gtb_sampling_rate = st.slider("Sampling rate", 0.1, 1.0, 0.6, step=0.05, key="gtb_rate")
    gtb_max_nodes     = st.slider("Max nodes", 2, 32, 8, key="gtb_nodes")

    st.markdown("**Support Vector Machine**")
    svm_gamma = st.slider("Gamma", 0.01, 2.0, 0.6, step=0.01, key="svm_gamma")
    svm_cost  = st.slider("Cost",  1.0, 100.0, 10.0, step=1.0, key="svm_cost")

st.sidebar.markdown("---")
run_clicked = st.sidebar.button(
    "Run Analysis", type="primary", width='stretch', disabled=not ee_ready
)
if not ee_ready:
    st.sidebar.warning("Connect to Earth Engine above first.")


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
    st.session_state["analysis_ready"] = True
    st.session_state["params"] = dict(
        county_selection=tuple(county_selection),
        agb_year=agb_year,
        num_pixels=num_pixels, train_split=train_split, seed=seed,
        rf_trees=rf_trees, rf_vars_per_split=rf_vars_per_split, rf_min_leaf=rf_min_leaf,
        svm_gamma=svm_gamma, svm_cost=svm_cost,
        gtb_trees=gtb_trees, gtb_shrinkage=gtb_shrinkage,
        gtb_sampling_rate=gtb_sampling_rate, gtb_max_nodes=gtb_max_nodes,
    )

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

# ----------------------------------------------------------------------------
# BUILD PREDICTOR STACK
# ----------------------------------------------------------------------------
with st.spinner("Building predictor stack on Earth Engine…"):
    stack = build_predictor_stack(project_id, p["county_selection"], p["agb_year"])

with st.spinner(f"Sampling {p['num_pixels']:,} pixels and splitting train / test…"):
    sample = sample_and_split(
        project_id, p["county_selection"],
        p["num_pixels"], p["train_split"], p["seed"], p["agb_year"],
    )

col1, col2, col3, col4 = st.columns(4)
col1.metric("AGB Reference Year",   p["agb_year"])
col2.metric("Total sampled points", f"{sample['n_total']:,}")
col3.metric("Training points",      f"{sample['n_train']:,}")
col4.metric("Testing points",       f"{sample['n_test']:,}")

with st.spinner("Training Random Forest, GTB, and SVM models on Earth Engine…"):
    models = train_models(
        project_id, p["county_selection"],
        p["num_pixels"], p["train_split"], p["seed"],
        p["rf_trees"], p["rf_vars_per_split"], p["rf_min_leaf"],
        p["svm_gamma"], p["svm_cost"],
        p["gtb_trees"], p["gtb_shrinkage"], p["gtb_sampling_rate"], p["gtb_max_nodes"],
        p["agb_year"],
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

MODEL_IMAGES = {
    "Random Forest":           estimated_carbon_rf,
    "Gradient Tree Boosting":  estimated_carbon_gtb,
    "Support Vector Machine":  estimated_carbon_svm,
    "Ensemble (RF+GTB+SVM)":   estimated_carbon_ensemble,
}

# ============================================================================
# TABS
# ============================================================================
tab_map, tab_compare, tab_validation, tab_zonal, tab_importance = st.tabs([
    "Interactive Map",
    "Model Comparison",
    "Validation",
    "Zonal Statistics",
    "Variable Importance",
])

# ── TAB: INTERACTIVE MAP ─────────────────────────────────────────────────────
with tab_map:
    st.markdown('<div class="sec-hdr"><i class="fa-solid fa-map"></i>Estimated Carbon Stock Map</div>', unsafe_allow_html=True)

    ctrl_col, map_col = st.columns([1, 3], gap="medium")
    with ctrl_col:
        selected_model_name   = st.radio("Model", list(MODEL_IMAGES.keys()), index=0)
        show_agb              = st.checkbox("Show AGB instead of carbon")
        show_counties_outline = st.checkbox("Show county boundaries", value=True)
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

        m.centerObject(selected_fc.geometry(), 9)
        if show_counties_outline:
            m.addLayer(selected_fc, {"color": "FF4444"}, "Selected Counties", True)
        m.addLayer(display_image.clip(selected_fc), vis, layer_label, True)
        m.add_colorbar(vis, label="t C/ha" if not show_agb else "Mg/ha AGB")
        m.to_streamlit(height=560)

# ── TAB: MODEL COMPARISON ────────────────────────────────────────────────────
with tab_compare:
    st.markdown(
        '<div class="sec-hdr"><i class="fa-solid fa-right-left"></i>Model Comparison &amp; Agreement</div>',
        unsafe_allow_html=True,
    )

    # ── Agreement metric (on demand) ─────────────────────────────────────────
    if st.button("Compute mean model spread across study area"):
        with st.spinner("Reducing model spread on Earth Engine…"):
            spread_result = model_spread.reduceRegion(
                reducer=ee.Reducer.mean(),
                geometry=selected_fc.geometry(),
                scale=1000,
                maxPixels=1e9,
                tileScale=4,
            ).getInfo()
            st.session_state["mean_spread"] = spread_result.get("Model_Spread")

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
    carbon_difference = (
        estimated_carbon_rf.subtract(estimated_carbon_gtb).abs().rename("Carbon_Difference_RF_GTB")
    )
    m2 = geemap.Map(center=[0.3, 36.0], zoom=7)
    m2.centerObject(selected_fc.geometry(), 9)
    m2.addLayer(carbon_difference.clip(selected_fc), VIS_PARAMS_DIFF, "|RF − GTB| (t C/ha)", True)
    m2.add_colorbar(VIS_PARAMS_DIFF, label="|RF − GTB| (t C/ha)")
    m2.to_streamlit(height=420)
    st.caption("Larger values = greater disagreement between Random Forest and Gradient Tree Boosting.")

    st.markdown("---")

    # ── 3-model spread map ───────────────────────────────────────────────────
    st.markdown("**3-Model Spread — Pixel-wise Standard Deviation (RF, GTB, SVM)**")
    m3 = geemap.Map(center=[0.3, 36.0], zoom=7)
    m3.centerObject(selected_fc.geometry(), 9)
    m3.addLayer(model_spread.clip(selected_fc), VIS_PARAMS_SPREAD, "Model Spread σ (t C/ha)", True)
    m3.add_colorbar(VIS_PARAMS_SPREAD, label="σ (t C/ha)")
    m3.to_streamlit(height=420)
    st.caption(
        "Per-pixel standard deviation across all three model predictions. "
        "Dark blue = high uncertainty; light = strong model agreement."
    )

# ── TAB: VALIDATION ──────────────────────────────────────────────────────────
with tab_validation:
    st.markdown('<div class="sec-hdr"><i class="fa-solid fa-chart-bar"></i>Validation on Held-Out Testing Set</div>', unsafe_allow_html=True)

    if st.button("Compute validation metrics", width='content'):
        results = {}
        for model_name, model_key in [
            ("Random Forest", "rf_model"),
            ("Gradient Tree Boosting", "gtb_model"),
            ("Support Vector Machine", "svm_model"),
        ]:
            with st.spinner(f"Computing {model_name} metrics…"):
                results[model_name] = compute_validation_metrics(
                    sample["testing_set"], models[model_key], sample["dependent_variable"]
                )
        st.session_state["validation_results"] = results

    results = st.session_state.get("validation_results")
    if results:
        metrics_df = pd.DataFrame(
            {name: {"RMSE": r["rmse"], "MAE": r["mae"], "R²": r["r2"]}
             for name, r in results.items()}
        ).T
        st.dataframe(metrics_df.style.format("{:.3f}"), width='stretch')
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
            <p style="font-size:0.88rem;">Click <strong>Compute validation metrics</strong> above.
            This calls Earth Engine and may take a minute.</p>
        </div>
        """, unsafe_allow_html=True)

# ── TAB: ZONAL STATISTICS ────────────────────────────────────────────────────
with tab_zonal:
    st.markdown('<div class="sec-hdr"><i class="fa-solid fa-location-dot"></i>Per-County Zonal Statistics</div>', unsafe_allow_html=True)

    zonal_model_choice = st.selectbox("Model for zonal stats:", list(MODEL_IMAGES.keys()),
                                      key="zonal_model")
    if st.button("Compute zonal statistics"):
        with st.spinner("Reducing regions on Earth Engine…"):
            zonal = MODEL_IMAGES[zonal_model_choice].reduceRegions(
                collection=selected_fc,
                reducer=(
                    ee.Reducer.mean()
                    .combine(ee.Reducer.min(), sharedInputs=True)
                    .combine(ee.Reducer.max(), sharedInputs=True)
                    .combine(ee.Reducer.sum(), sharedInputs=True)
                ),
                scale=1000,   # matches ESA CCI AGB native resolution; 100 m exceeds EE memory
                tileScale=8,
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

    zonal_df = st.session_state.get("zonal_df")
    if zonal_df is not None:
        st.dataframe(zonal_df, width='stretch')

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
    raw_importance  = models["rf_importance"] if imp_choice == "Random Forest" else models["gtb_importance"]
    importance_dict = raw_importance.get("importance", raw_importance)

    if isinstance(importance_dict, dict):
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
    else:
        st.json(raw_importance)

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
