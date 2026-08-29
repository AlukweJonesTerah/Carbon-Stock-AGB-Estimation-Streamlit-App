import ee
import pandas as pd
import numpy as np
from src.config import *
from src.ee_processing import get_study_area

def compute_validation_metrics(testing_set, model, dependent_variable):
    # Vectorized client-side evaluation (1 network call instead of 5)
    predicted = testing_set.classify(model)
    
    # Pull arrays to client
    actual_vals = predicted.aggregate_array(dependent_variable).getInfo()
    pred_vals   = predicted.aggregate_array("classification").getInfo()
    
    # Calculate metrics instantly in memory using numpy
    actual = np.array(actual_vals)
    pred = np.array(pred_vals)
    
    if len(actual) == 0:
        return {"rmse": 0, "mae": 0, "r2": 0, "actual": [], "predicted": []}
        
    rmse = np.sqrt(np.mean((actual - pred)**2))
    mae = np.mean(np.abs(actual - pred))
    bias = np.mean(pred - actual)
    mape = np.mean(np.abs((actual - pred) / np.maximum(actual, 1e-6))) * 100
    
    # R2 Calculation
    ss_res = np.sum((actual - pred)**2)
    ss_tot = np.sum((actual - np.mean(actual))**2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
    
    return {"rmse": float(rmse), "mae": float(mae), "bias": float(bias), "mape": float(mape), "r2": float(r2), "actual": actual_vals, "predicted": pred_vals}

import streamlit as st
@st.cache_resource(show_spinner=False)
def compute_smart_ensemble_weights(_testing_set, _models, dependent_variable, _cache_buster):
    """Calculate inverse-variance weights for the ensemble based on testing RMSE."""
    # Classify with all models simultaneously to avoid re-evaluating the sampling graph 6 times
    multi_pred = _testing_set \
        .classify(_models["rf_model"], "rf_pred") \
        .classify(_models["gtb_model"], "gtb_pred") \
        .classify(_models["svm_model"], "svm_pred")
    
    # Fetch all predictions in exactly ONE network call
    try:
        features = multi_pred.select([dependent_variable, "rf_pred", "gtb_pred", "svm_pred"]).getInfo()["features"]
    except Exception:
        # Fallback to equal weights if GEE fails
        return {"rf_model": 0.333, "gtb_model": 0.333, "svm_model": 0.333}
        
    actual = np.array([f["properties"].get(dependent_variable, 0) for f in features])
    preds = {
        "rf_model": np.array([f["properties"].get("rf_pred", 0) for f in features]),
        "gtb_model": np.array([f["properties"].get("gtb_pred", 0) for f in features]),
        "svm_model": np.array([f["properties"].get("svm_pred", 0) for f in features]),
    }
    
    weights = {}
    for name, p_arr in preds.items():
        if len(actual) > 0:
            rmse = np.sqrt(np.mean((actual - p_arr)**2))
            weights[name] = 1.0 / (rmse ** 2 + 1e-6)
        else:
            weights[name] = 1.0
            
    total = sum(weights.values())
    return {k: v / total for k, v in weights.items()}


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

