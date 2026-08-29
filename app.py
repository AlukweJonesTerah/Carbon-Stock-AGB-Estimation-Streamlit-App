"""
Carbon Stock & Above-Ground Biomass (AGB) Estimation — Streamlit App
======================================================================
A Streamlit port of the Google Earth Engine notebook that estimates
above-ground biomass carbon stock across selected Kenyan counties using
Sentinel-1/2, SRTM, ERA5-Land, PALSAR, WorldCover, Hansen, soil, and canopy-height predictors,
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

# `_busy` is True for the duration of a long-running action (e.g. the main
# "Run analysis" pipeline). While it's True, the buttons/inputs that trigger
# another run are rendered disabled so a fast double-click (or fiddling with
# settings mid-run) can't queue a second, overlapping execution.
st.session_state.setdefault("_busy", False)

# ----------------------------------------------------------------------------
# THEME / CSS
# ----------------------------------------------------------------------------

# Load Config & Theme
from src.config import *
setup_matplotlib_theme()

import importlib
import src.ee_auth
import src.ee_processing
importlib.reload(src.ee_processing)

# Load Modules
from src.ee_auth import init_earth_engine, _is_cloud_deployment, _STREAMLIT_CLOUD_SETUP
from src.ee_processing import get_study_area, build_predictor_stack, sample_and_split, train_models, compute_regional_statistics
from src.evaluation import compute_validation_metrics, compute_watchlist_alerts, assess_data_quality
from src.plotting import make_scatter_plot, build_report_pdf, build_portfolio_pdf, build_map_briefing
from src.ai_guide import build_learning_context, ask_learning_guide, offline_learning_response, render_voice_player

# Load CSS
with open('assets/style.css', 'r', encoding='utf-8') as f:
    st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)


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
        gtb_shrinkage     = st.slider("Shrinkage", 0.001, 0.2, 0.05, step=0.01,
                                       format="%.3f", key="gtb_shrink")
        gtb_sampling_rate = st.slider("Sampling rate", 0.1, 1.0, 0.6, step=0.05, key="gtb_rate")
        gtb_max_nodes     = st.slider("Max nodes", 2, 32, 8, key="gtb_nodes")

        st.markdown("**Support Vector Machine**")
        svm_gamma = st.slider("Gamma", 0.01, 2.0, 0.6, step=0.01, key="svm_gamma")
        svm_cost  = st.slider("Cost", 1.0, 100.0, 10.0, step=1.0, key="svm_cost")

    def _mark_analysis_busy():
        st.session_state["_busy"] = True

    run_clicked = st.form_submit_button(
        "Run analysis", type="primary", width="stretch",
        disabled=(not ee_ready) or st.session_state["_busy"],
        on_click=_mark_analysis_busy,
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
        disabled=saved_config_file is None or not ee_ready or st.session_state["_busy"],
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
        <span class="hero-tag">ERA5-Land & WorldClim</span>
        <span class="hero-tag">ESA WorldCover</span>
        <span class="hero-tag">Hansen GFC</span>
        <span class="hero-tag">Human Modification Index</span>
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
    st.session_state["_busy"] = False
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
    st.session_state["_busy"] = False
    st.warning("Select at least one county in the sidebar.")
    if run_clicked:
        st.rerun()
    st.stop()

import threading

@st.cache_resource
def get_global_concurrency_state():
    return {"active_runs": 0, "lock": threading.Lock(), "max_runs": 5}

global_state = get_global_concurrency_state()

if run_clicked:
    # --- Anti-DoS Protection ---
    # 1. Session Cooldown: Prevent a single user from spamming the run button
    now = time.time()
    last_run = st.session_state.get("last_run_timestamp", 0)
    if now - last_run < 5.0: # 5 second cooldown
        st.error("Anti-Spam: Please wait a few seconds before running another analysis.")
        st.session_state["_busy"] = False
        st.stop()
    st.session_state["last_run_timestamp"] = now
    
    # 2. Global Concurrency Lock: Prevent DDoS from overwhelming Earth Engine API
    with global_state["lock"]:
        if global_state["active_runs"] >= global_state["max_runs"]:
            st.error("Server is currently under heavy load (Maximum concurrent analyses reached). Please try again in a minute.")
            st.session_state["_busy"] = False
            st.stop()
        global_state["active_runs"] += 1
        
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
    st.session_state["_busy"] = False
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
st.session_state["_busy"] = True
try:
    run_status = st.status("Preparing analysis", expanded=False)
    run_status.update(label="1/3 Building predictor stack", state="running")
    with st.spinner("Building predictor stack on Earth Engine…"):
        stack = build_predictor_stack(project_id, p["county_selection"], p["agb_year"])

    with st.spinner(f"Sampling {p['num_pixels']:,} pixels and splitting train / test…"):
        run_status.update(label="2/3 Sampling training and testing pixels", state="running")
        sample = sample_and_split(
            project_id, p["county_selection"],
            p["num_pixels"], p["train_split"], p["seed"], p["agb_year"],
            "cache_buster_v2"
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
            "cache_buster_v2"
        )

    if run_clicked:
        st.session_state["last_preparation_seconds"] = (
            time.perf_counter() - st.session_state["analysis_started_at"]
        )
    run_status.update(label="Analysis ready", state="complete", expanded=False)
finally:
    st.session_state["_busy"] = False
    if run_clicked:
        with global_state["lock"]:
            global_state["active_runs"] = max(0, global_state["active_runs"] - 1)
        # The button above was rendered disabled at the start of this exact
        # run (the on_click callback set _busy before the sidebar drew it).
        # Resetting the flag doesn't retroactively update that already-sent
        # widget, so trigger one more rerun to redraw the sidebar with the
        # button enabled again. This only fires right after a fresh submit,
        # not on every later interaction, so it can't loop.
        st.rerun()
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

from src.evaluation import compute_smart_ensemble_weights
with st.spinner("Fusing Smart Weighted Ensemble..."):
    weights = compute_smart_ensemble_weights(
        sample["testing_set"], models, sample["dependent_variable"], "cache_buster_v2"
    )

estimated_carbon_ensemble = (
    estimated_carbon_rf.multiply(weights["rf_model"])
    .add(estimated_carbon_gtb.multiply(weights["gtb_model"]))
    .add(estimated_carbon_svm.multiply(weights["svm_model"]))
    .rename("Estimated Carbon Stock Ensemble")
)

st.session_state["ensemble_weights"] = weights

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
    """Render Folium maps through Streamlit-Folium without websocket lag."""
    map_obj.add_layer_control()
    
    # By passing an empty array to returned_objects when bidirectional is False,
    # st_folium will render the map normally but will NOT send websocket 
    # updates back to Python when the user pans or zooms. This completely eliminates
    # Streamlit lag while preserving the necessary Leaflet JS dependencies.
    returned_objects = ["last_clicked", "last_active_drawing"] if bidirectional else []
    
    return st_folium(
        map_obj,
        height=height,
        use_container_width=False, # Deprecated in Streamlit
        width="100%", # For st_folium to scale
        returned_objects=returned_objects,
    )


MODEL_IMAGES = {
    "Random Forest":           estimated_carbon_rf,
    "Gradient Tree Boosting":  estimated_carbon_gtb,
    "Support Vector Machine":  estimated_carbon_svm,
    "Smart Weighted Ensemble": estimated_carbon_ensemble,
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
        enable_split_map      = st.checkbox(
            "Enable split-map swipe tool", 
            help="Swipe between the Carbon model and true-color Sentinel-2 satellite imagery."
        )
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
        try:
            stats = compute_regional_statistics(
                st.session_state["ee_project_id"],
                p["county_selection"],
                MODEL_IMAGES["Smart Weighted Ensemble"],
                predictor_variables.select("hansen_loss"),
                selected_fc
            )
            met1, met2, met3, met4, met5 = st.columns(5)
            st.session_state["regional_stats"] = stats
            met1.metric("Total Carbon", f"{stats['total_carbon'] / 1e6:,.2f} M t C")
            met2.metric("Mean Density", f"{stats['mean_density']:,.1f} t C/ha")
            met3.metric("Deforested Area", f"{stats['deforested_area']:,.0f} ha")
            met4.metric("Total Area", f"{stats['total_area'] / 1000:,.1f}k ha")
            
            est_value = (stats['total_carbon'] * (44 / 12) * 20) / 1e9
            met5.metric("Indicative Value", f"${est_value:,.2f}B", help="Rough gross value assuming $20/t CO2e in voluntary markets.")
        except Exception as e:
            st.warning(f"Could not calculate regional statistics: {e}")

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
        
        if enable_split_map:
            import ipyleaflet
            # Create Earth Engine tile layers for the swipe tool
            left_layer = geemap.ee_tile_layer(display_image.clip(_display_geom), vis, layer_label)
            s2_rgb = predictor_variables.select(["B4", "B3", "B2"]).clip(_display_geom)
            right_layer = geemap.ee_tile_layer(s2_rgb, {"min": 0, "max": 3000}, "Sentinel-2 RGB")
            m.split_map(left_layer, right_layer)
        else:
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
    try:
        stored_gemini_key = st.secrets.get("ai", {}).get("gemini_api_key", "")
    except Exception:
        stored_gemini_key = ""
        
with tab_briefing:
    st.markdown('<div class="sec-hdr"><i class="fa-solid fa-clipboard-check"></i>Automated Map Briefing</div>', unsafe_allow_html=True)
    st.markdown(
        "Generate a plain-language explanation of the selected counties, strongest tested model, "
        "uncertainty, county hotspots, and important cautions."
    )
    if st.button("Generate map briefing", type="primary", key="btn_map_briefing", disabled=st.session_state["_busy"]):
        with st.spinner("Generating Map Briefing..."):
            fallback_briefing = build_map_briefing(
                p,
                validation_results=st.session_state.get("validation_results"),
                mean_spread=st.session_state.get("mean_spread"),
                zonal_df=st.session_state.get("zonal_df"),
            )
            if stored_gemini_key:
                try:
                    from src.ai_guide import build_learning_context, ask_learning_guide
                    context = build_learning_context(p)
                    z_df = st.session_state.get("zonal_df")
                    if z_df is not None:
                        context += "\nZonal Stats Data: " + z_df.to_json(orient="records")
                    
                    prompt = "Write an executive map briefing summarizing this carbon analysis. Highlight the most accurate model, uncertainty levels, regional carbon stats, and county hotspots (from the zonal stats). Do not explain what the app is, just summarize the data insights. Use bullet points and a professional yet accessible tone. Keep it under 250 words."
                    ai_briefing = ask_learning_guide("Gemini", "gemini-1.5-flash", prompt, [], context, stored_gemini_key)
                    st.session_state["map_briefing"] = "✨ **AI-Generated Executive Summary:**\n\n" + ai_briefing
                except Exception as e:
                    st.session_state["map_briefing"] = f"*(AI Generation failed: {e})*\n\n" + fallback_briefing
            else:
                st.session_state["map_briefing"] = fallback_briefing

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
            quick_label = quick_prompt if quick_prompt.endswith("?") else f"{quick_prompt.split('?')[0].strip()}?"
            if st.button(
                quick_label,
                key=f"guide_{quick_prompts.index(quick_prompt)}",
                disabled=st.session_state.get("_busy", False),
                width="stretch",
            ):
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
    if st.button("Compute mean model spread across study area", disabled=st.session_state["_busy"]):
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
    if st.button("Render RF vs GTB difference map", key="btn_diff_map", disabled=st.session_state["_busy"]):
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
    if st.button("Render 3-model spread map", key="btn_spread_map", disabled=st.session_state["_busy"]):
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

    st.markdown("---")
    st.markdown("**Statistical Agreement (RF vs GTB)**")
    st.markdown("Sample 1,000 random locations to see if models systematically diverge.")
    if st.button("Generate Scatter Plot", key="btn_scatter", disabled=st.session_state["_busy"]):
        with st.spinner("Sampling predictions across the landscape..."):
            combo = estimated_carbon_rf.addBands(estimated_carbon_gtb)
            samples = combo.sample(
                region=_display_geom,
                scale=1000,
                numPixels=1000,
                geometries=False,
                tileScale=16
            ).getInfo()
            
            features = samples.get("features", [])
            rf_vals = [f["properties"].get("Estimated Carbon Stock RF") for f in features]
            gtb_vals = [f["properties"].get("Estimated Carbon Stock GTB") for f in features]
            
            valid = [(r, g) for r, g in zip(rf_vals, gtb_vals) if r is not None and g is not None]
            if valid:
                rf_vals, gtb_vals = zip(*valid)
                df = pd.DataFrame({"Random Forest (t C/ha)": rf_vals, "Gradient Tree Boosting (t C/ha)": gtb_vals})
                import plotly.express as px
                fig = px.scatter(df, x="Random Forest (t C/ha)", y="Gradient Tree Boosting (t C/ha)", 
                                 opacity=0.5, 
                                 title="Model Prediction Agreement (1,000 Random Points)")
                fig.update_layout(shapes=[
                    dict(type='line', x0=0, y0=0, x1=df.max().max(), y1=df.max().max(),
                         line=dict(color='red', dash='dash'))
                ])
                st.plotly_chart(fig, width='stretch')
                st.caption("The red dashed line represents perfect 1:1 agreement. Significant deviation off this line indicates one model systematically predicts higher carbon than the other.")
            else:
                st.warning("Could not extract valid sample points.")

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
        calculate_scenario = st.button("Calculate restoration scenario", type="primary", key="calculate_restoration", disabled=st.session_state["_busy"])

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
            with st.spinner("Calculating restoration screening scenario (with ecological suitability masking)…"):
                try:
                    # Restoration Suitability Mask
                    wc = predictor_variables.select("worldcover_class")
                    hm = predictor_variables.select("human_modification")
                    tc = predictor_variables.select("hansen_treecover")
                    
                    # Exclude Water (80), Urban (50), Snow (70), highly modified (>0.5), and dense forest (>40%)
                    unsuitable = wc.eq(80).Or(wc.eq(50)).Or(wc.eq(70)).Or(hm.gt(0.5)).Or(tc.gt(40))
                    suitability_mask = unsuitable.Not()
                    
                    gross_area_ha = ee.Number(scenario_geometry.area()).divide(10000).getInfo()
                    
                    pixel_area = ee.Image.pixelArea().divide(10000)
                    restorable_area_ha = pixel_area.updateMask(suitability_mask).reduceRegion(
                        reducer=ee.Reducer.sum(), geometry=scenario_geometry, scale=1000,
                        maxPixels=1e10, tileScale=16, bestEffort=True,
                    ).getInfo().get("area")
                    area_ha = float(restorable_area_ha or 0.0)
                    
                    baseline_carbon = estimated_carbon_ensemble.updateMask(suitability_mask).reduceRegion(
                        reducer=ee.Reducer.mean(), geometry=scenario_geometry, scale=1000,
                        maxPixels=1e10, tileScale=16, bestEffort=True,
                    ).getInfo().get("Estimated Carbon Stock Ensemble")
                    uncertainty = model_spread.updateMask(suitability_mask).reduceRegion(
                        reducer=ee.Reducer.mean(), geometry=scenario_geometry, scale=1000,
                        maxPixels=1e10, tileScale=16, bestEffort=True,
                    ).getInfo().get("Model_Spread")
                    
                    dominant_biome = predictor_variables.select("biome").updateMask(suitability_mask).reduceRegion(
                        reducer=ee.Reducer.mode(), geometry=scenario_geometry, scale=1000,
                        maxPixels=1e10, tileScale=16, bestEffort=True,
                    ).getInfo().get("biome")
                    
                    import math
                    biome_params = {
                        1: {"max_c": 180, "name": "Tropical Moist Forest"},
                        2: {"max_c": 100, "name": "Tropical Dry Forest"},
                        7: {"max_c": 35,  "name": "Savanna / Shrubland"},
                        9: {"max_c": 45,  "name": "Flooded Savanna"},
                        10: {"max_c": 50, "name": "Montane Grasslands"},
                        14: {"max_c": 250, "name": "Mangroves"},
                    }
                    b_num = int(dominant_biome) if dominant_biome is not None else 7
                    params = biome_params.get(b_num, {"max_c": 60, "name": "Mixed/Other"})
                    
                    C_0 = baseline_carbon if baseline_carbon is not None else 1.0
                    C_0 = max(1.0, C_0)
                    K = params["max_c"]
                    
                    growth_curve = []
                    if C_0 >= K:
                        C_t = C_0
                        for y in range(scenario_years + 1): growth_curve.append(C_0)
                    else:
                        user_annual_c_growth = annual_growth * CARBON_TO_BIOMASS_FACTOR
                        r = user_annual_c_growth / (C_0 * (1 - C_0/K))
                        r = min(r, 0.3)
                        for y in range(scenario_years + 1):
                            cy = K / (1 + ((K - C_0) / C_0) * math.exp(-r * y))
                            growth_curve.append(cy)
                        C_t = growth_curve[-1]
                        
                    added_carbon_per_ha = C_t - C_0
                    added_biomass_per_ha = added_carbon_per_ha / CARBON_TO_BIOMASS_FACTOR

                    total_added_carbon = (area_ha or 0) * added_carbon_per_ha
                    total_added_co2e = total_added_carbon * (44 / 12)
                    total_biomass = (area_ha or 0) * added_biomass_per_ha
                    st.session_state["restoration_scenario"] = {
                        "gross_area_ha": gross_area_ha,
                        "area_ha": area_ha or 0, "baseline_carbon": baseline_carbon,
                        "uncertainty": uncertainty, "added_biomass_per_ha": added_biomass_per_ha,
                        "added_carbon_per_ha": added_carbon_per_ha, "total_biomass": total_biomass,
                        "total_carbon": total_added_carbon, "total_co2e": total_added_co2e,
                        "restoration_type": restoration_type, "years": scenario_years,
                        "biome_name": params["name"],
                        "max_c": K,
                        "growth_curve": growth_curve
                    }
                except Exception as _e:
                    st.error(f"Could not calculate the restoration scenario: {_e}")

    scenario = st.session_state.get("restoration_scenario")
    if scenario:
        st.markdown("#### Scenario results (Suitability & Biome Filtered)")
        r1, r2, r3, r4, r5, r6 = st.columns(6)
        r1.metric("Gross Area", f"{scenario.get('gross_area_ha', 0):,.1f} ha")
        r2.metric("Restorable Area", f"{scenario['area_ha']:,.1f} ha")
        r3.metric("Biomass gain", f"{scenario['total_biomass']:,.0f} Mg")
        r4.metric("Carbon gain", f"{scenario['total_carbon']:,.0f} t C")
        r5.metric("Indicative CO₂e", f"{scenario['total_co2e']:,.0f} t CO₂e")
        
        proj_value = scenario['total_co2e'] * 20
        if proj_value > 1e6:
            r6.metric("Projected Revenue", f"${proj_value/1e6:,.1f}M", help="Assuming $20/t CO2e in voluntary markets.")
        else:
            r6.metric("Projected Revenue", f"${proj_value:,.0f}", help="Assuming $20/t CO2e in voluntary markets.")
        baseline_text = "not available" if scenario["baseline_carbon"] is None else f"{float(scenario['baseline_carbon']):.1f} t C/ha"
        uncertainty_text = "not available" if scenario["uncertainty"] is None else f"± {float(scenario['uncertainty']):.1f} t C/ha"
        
        st.info(
            f"**Dominant Biome:** {scenario.get('biome_name', 'Unknown')} (Max Capacity: {scenario.get('max_c', 'N/A')} t C/ha)\n\n"
            f"**Current baseline:** {baseline_text} ({uncertainty_text}). "
            f"**Added carbon per ha:** {scenario['added_carbon_per_ha']:.1f} t C/ha. "
            f"Based on a logistic growth curve over {scenario['years']} years."
        )
        
        if "growth_curve" in scenario:
            st.markdown("**Carbon Accumulation Curve (t C/ha)**")
            curve_df = pd.DataFrame({"Year": range(len(scenario["growth_curve"])), "Carbon (t C/ha)": scenario["growth_curve"]})
            st.line_chart(curve_df, x="Year", y="Carbon (t C/ha)")

with tab_tools:
    st.markdown('<div class="sec-hdr"><i class="fa-solid fa-compass"></i>Decision Tools</div>', unsafe_allow_html=True)
    change_col, calculator_col = st.columns(2)
    with change_col:
        st.markdown("#### Carbon change explorer")
        change_years = st.select_slider(
            "Compare reference years", options=AGB_YEARS_AVAIL,
            value=("2010", "2022"), key="change_years",
        )
        if st.button("Render carbon change map", key="btn_change_map", disabled=st.session_state["_busy"]):
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
        st.markdown("#### Restoration project calculator")
        project_area = st.number_input("Project area (hectares)", min_value=1, value=1000)
        annual_biomass_gain = st.number_input(
            "Expected annual biomass gain (Mg/ha/year)", min_value=0.0, value=5.0, step=0.5,
        )
        project_years = st.slider("Project duration (years)", 1, 30, 10)
        carbon_price = st.slider("Carbon Price (USD per t CO₂e)", min_value=1.0, max_value=100.0, value=15.0, step=1.0)
        
        projected_co2e = project_area * annual_biomass_gain * project_years * CARBON_TO_BIOMASS_FACTOR * (44 / 12)
        gross_revenue = projected_co2e * carbon_price
        
        met1, met2 = st.columns(2)
        met1.metric("Indicative Sequestration", f"{projected_co2e:,.0f} t CO₂e")
        met2.metric("Estimated Gross Revenue", f"${gross_revenue:,.0f} USD")
        
        st.caption("Screening estimate only: excludes baseline, leakage, permanence, and project deductions.")
        
        # --- New Climate Risk Assessment powered by WorldClim ---
        st.markdown("#### Climate Risk Assessment (WorldClim)")
        if st.button("Assess local climate risk", key="btn_climate_risk", disabled=st.session_state["_busy"]):
            with st.spinner("Analyzing bioclimatic variables..."):
                worldclim = ee.Image("WORLDCLIM/V1/BIO")
                climate_stats = worldclim.select(["bio04", "bio15"]).reduceRegion(
                    reducer=ee.Reducer.mean(),
                    geometry=selected_fc.geometry(),
                    scale=5000, maxPixels=1e9,
                    bestEffort=True
                ).getInfo()
                
                temp_cv = climate_stats.get("bio04", 0) / 100.0
                precip_cv = climate_stats.get("bio15", 0)
                
                cr1, cr2 = st.columns(2)
                cr1.metric("Temp Seasonality (StdDev)", f"{temp_cv:.1f}°C")
                cr2.metric("Precip Seasonality (CV)", f"{precip_cv:.1f}%")
                
                if precip_cv > 80:
                    st.warning("⚠️ **High Precipitation Seasonality:** This region experiences severe dry seasons. Restoration projects here will require drought-resistant species and high early-stage irrigation budgets to prevent seedling mortality.")
                elif precip_cv > 50:
                    st.info("ℹ️ **Moderate Seasonality:** Standard mixed-species restoration is viable, but planting should be strictly timed with the onset of the rainy season.")
                else:
                    st.success("✅ **Low Seasonality:** The region has highly uniform precipitation year-round, ideal for rapid, continuous biomass accumulation.")


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
    if st.button("Check watchlist for change", type="primary", key="check_watchlist", disabled=st.session_state["_busy"]):
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
            if st.button("Validate uploaded plots", key="btn_field_validation", disabled=st.session_state["_busy"]):
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

    if st.button("1. Compute validation metrics", type="primary", width='content', disabled=st.session_state["_busy"]):
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
            {name: {"RMSE": r["rmse"], "MAE": r["mae"], "Bias": r["bias"], "MAPE (%)": r["mape"], "R²": r["r2"]}
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

        st.markdown("### Residual Error Distribution")
        st.caption("Analyzes where the models are over-predicting (positive error) or under-predicting (negative error).")
        import plotly.graph_objects as go
        fig_resid = go.Figure()
        for name, r in results.items():
            actual_arr = np.array(r["actual"])
            pred_arr = np.array(r["predicted"])
            residuals = pred_arr - actual_arr
            fig_resid.add_trace(go.Violin(y=residuals, name=name, box_visible=True, meanline_visible=True, line_color=scatter_colors[name]))
        fig_resid.update_layout(yaxis_title="Error (Predicted - Actual) t C/ha", title="Error Variance per Model")
        st.plotly_chart(fig_resid, width='stretch')
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
    if st.button("Assess run data quality", type="primary", key="assess_data_quality", disabled=st.session_state["_busy"]):
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
    if st.button("Compute zonal statistics", disabled=st.session_state["_busy"]):
        with st.spinner("Reducing regions on Earth Engine…"):
            try:
                # Mathematically correct sum = mean (t/ha) * area (ha). ReduceRegions sum on arbitrary pixels is incorrect.
                # Using scale 2500 is much faster and statistically valid for county-level means.
                def calc_zonal(feat):
                    stats = MODEL_IMAGES[zonal_model_choice].reduceRegion(
                        reducer=ee.Reducer.mean().combine(ee.Reducer.min(), sharedInputs=True).combine(ee.Reducer.max(), sharedInputs=True),
                        geometry=feat.geometry(),
                        scale=2500,
                        maxPixels=1e10,
                        tileScale=16,
                        bestEffort=True
                    )
                    area_ha = ee.Number(feat.geometry().area()).divide(10000)
                    bname = ee.String(MODEL_IMAGES[zonal_model_choice].bandNames().get(0))
                    # When using combine() with mean as the base, output is usually bname_mean, bname_min, etc.
                    mean_val = ee.Number(stats.get(bname.cat("_mean")))
                    min_val = ee.Number(stats.get(bname.cat("_min")))
                    max_val = ee.Number(stats.get(bname.cat("_max")))
                    return feat.set("mean", mean_val).set("min", min_val).set("max", max_val).set("sum", mean_val.multiply(area_ha))
                
                zonal = selected_fc.map(calc_zonal)
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
        
        import plotly.express as px
        fig = px.bar(
            ranked_counties, 
            x="County", y=ranking_column,
            color=ranking_column, color_continuous_scale="Viridis",
            title=f"{ranking_basis} by County",
            labels={ranking_column: ranking_basis}
        )
        st.plotly_chart(fig, width='stretch')

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
    if st.button("Load variable importance", key="btn_variable_importance", disabled=st.session_state["_busy"]):
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
            p, st.session_state.get("validation_results"), st.session_state.get("zonal_df"), st.session_state.get("map_briefing")
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
        if st.button("Create GeoTIFF export task", key="btn_geotiff_export", disabled=st.session_state["_busy"]):
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
    ESA CCI AGB · Sentinel-1/2 (Copernicus) · Dynamic World · ESA WorldCover · Hansen GFC · SRTM · ERA5-Land ·
    OpenLandMap Soil Organic Carbon · Meta Canopy Height · MODIS LST · JAXA ALOS PALSAR<br>
    Carbon-to-biomass conversion factor: <strong>0.47</strong> (IPCC default, Mg C / Mg dry biomass)
</div>
""", unsafe_allow_html=True)
