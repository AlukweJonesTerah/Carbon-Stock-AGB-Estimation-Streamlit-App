import ee
import streamlit as st
from src.config import *

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
        .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 30))  # Loosened to 30% to prevent missing data patches in a 2-year window
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
    
    # Only mask out Water (0), Urban/Built (6), and Snow/Ice (8).
    # Previously this was strictly .eq(1) (Trees), which completely deleted the 
    # entire carbon map for Eastern Kenya's savannas and shrublands.
    invalid_landcover = dynamic_world_label.eq(0).Or(dynamic_world_label.eq(6)).Or(dynamic_world_label.eq(8))
    valid_landcover = invalid_landcover.Not()
    
    masked_biomass = biomass_data.updateMask(valid_landcover)
    # Allow ESA AGB values of 0 (deserts) to be trained on so the model 
    # learns to predict 0 in arid regions instead of ignoring them completely.
    biomass_mask   = masked_biomass.mask().gte(0)

    dem       = ee.Image("USGS/SRTMGL1_003")
    elevation = dem.select("elevation")
    slope     = ee.Terrain.slope(dem).rename("slope")
    aspect    = ee.Terrain.aspect(dem).rename("aspect")

    # Dynamic, up-to-date climate data from ERA5-Land
    era5 = (
        ee.ImageCollection("ECMWF/ERA5_LAND/MONTHLY_AGGR")
        .filterDate(DATE_START_S2, DATE_END_S2)
    )
    # Temperature in K to C
    mean_temp = era5.select("temperature_2m").mean().subtract(273.15).rename("mean_temp")
    # Precipitation in m/month to mm/year
    annual_precip = era5.select("total_precipitation_sum").mean().multiply(12000).rename("annual_precip")
    
    # Add WorldClim Bioclimatic variables (Crucial for ecological modeling!)
    worldclim = ee.Image("WORLDCLIM/V1/BIO")
    temp_seasonality = worldclim.select("bio04").rename("temp_seasonality")
    precip_seasonality = worldclim.select("bio15").rename("precip_seasonality")

    s1 = (
        ee.ImageCollection("COPERNICUS/S1_GRD")
        .filterBounds(geom)
        .filterDate(DATE_START_S2, DATE_END_S2)
        .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VH"))
        .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VV"))
        .filter(ee.Filter.eq("instrumentMode", "IW"))
        .select(["VV", "VH"])
        .median()
        .rename(["S1_VV", "S1_VH"])
    )
    vh_int   = s1.multiply(100).toInt32()
    glcm     = vh_int.glcmTexture(size=4)
    contrast = glcm.select("S1_VH_contrast").rename("S1_contrast")

    soil_carbon = (
        ee.Image("OpenLandMap/SOL/SOL_ORGANIC-CARBON_USDA-6A1C_M/v02")
        .select("b0")
        .rename("soil_carbon")
    )
    
    soil_clay = (
        ee.Image("OpenLandMap/SOL/SOL_CLAY-WFRACTION_USDA-3A1A1A_M/v02")
        .select("b0")
        .rename("soil_clay")
    )

    gpp = (
        ee.ImageCollection("MODIS/061/MOD17A2H")
        .filterDate(DATE_START_S2, DATE_END_S2)
        .select("Gpp")
        .median()
        .rename("modis_gpp")
    )
    canopy_height = (
        ee.ImageCollection("projects/meta-forest-monitoring-okw37/assets/CanopyHeight")
        .mosaic()
        .rename("canopy_height")
    )
    modis_lst = ee.ImageCollection("MODIS/061/MOD11A2").filterDate(DATE_START_S2, DATE_END_S2).median()
    lst = modis_lst.select("LST_Day_1km").multiply(0.02).subtract(273.15).rename("LST")

    # PALSAR is released yearly and may lag behind current date; use a wider window
    palsar_composite = (
        ee.ImageCollection("JAXA/ALOS/PALSAR/YEARLY/SAR_EPOCH")
        .filterDate("2015-01-01", DATE_END_S2)
        .filterBounds(geom)
        .median()
    )
    hh = palsar_composite.select("HH").rename("PALSAR_HH")
    hv = palsar_composite.select("HV").rename("PALSAR_HV")

    # Add Hansen Global Forest Change treecover & historical loss as predictors
    hansen = ee.Image("UMD/hansen/global_forest_change_2025_v1_13")
    treecover2000 = hansen.select("treecover2000").rename("hansen_treecover")
    forest_loss = hansen.select("loss").rename("hansen_loss")

    # Add ESA WorldCover 2021 as a predictor
    worldcover = ee.ImageCollection("ESA/WorldCover/v200").first().select("Map").rename("worldcover_class")

    # Add Global Human Modification Index (proxy for degradation/urbanization)
    human_modification = ee.ImageCollection("CSP/HM/GlobalHumanModification").mosaic().rename("human_modification")

    # Add spatial coordinates to help the ML models account for spatial autocorrelation
    coords = ee.Image.pixelLonLat()
    lat = coords.select("latitude").rename("lat")
    lon = coords.select("longitude").rename("lon")

    # Add Ecoregions (Biome number)
    biome = ee.FeatureCollection("RESOLVE/ECOREGIONS/2017").reduceToImage(["BIOME_NUM"], ee.Reducer.first()).rename("biome")

    predictors_all = (
        ee.Image.constant(1)
        .addBands(s2).addBands(ndvi).addBands(evi).addBands(savi)
        .addBands(elevation).addBands(slope).addBands(aspect)
        .addBands(s1).addBands(canopy_height)
        .addBands(mean_temp).addBands(annual_precip)
        .addBands(temp_seasonality).addBands(precip_seasonality)
        .addBands(soil_carbon).addBands(soil_clay)
        .addBands(hh).addBands(hv)
        .addBands(contrast).addBands(ndmi).addBands(ndre).addBands(lst)
        .addBands(treecover2000).addBands(forest_loss)
        .addBands(worldcover).addBands(human_modification)
        .addBands(gpp).addBands(biome)
        .addBands(lat).addBands(lon)
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
def sample_and_split(_project_id, county_selection, num_pixels, train_split, seed, agb_year, _cache_buster=None):
    stack = build_predictor_stack(_project_id, county_selection, agb_year)
    predictor_variables = stack["predictor_variables"]
    final_biomass       = stack["final_biomass"]
    selected_fc         = stack["selected_fc"]

    dependent_variable   = "carbon_tonnes_per_ha"
    predictor_band_names = predictor_variables.bandNames().getInfo()
    if "constant" in predictor_band_names:
        predictor_band_names.remove("constant")

    combined_dataset = predictor_variables.addBands(final_biomass)
    
    # Stratified Random Sampling guarantees the model learns rare "high carbon" classes (like dense forests)
    # rather than just memorizing the majority class (like bare ground or grass).
    # We stratify by the 'worldcover' band which is automatically included in predictor_variables.
    points_per_class = max(int(num_pixels / 7), 50)
    
    all_sampled = combined_dataset.stratifiedSample(
        numPoints=points_per_class,
        classBand='worldcover_class',
        region=selected_fc.geometry(), 
        scale=2500, # Increased from 1000m to 2500m for lightning-fast spatial extraction over massive regions like Eastern Kenya
        geometries=True, 
        tileScale=16,
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
def train_models(
    _project_id, county_selection, num_pixels, train_split, seed,
    rf_trees, rf_vars_per_split, rf_min_leaf,
    svm_gamma, svm_cost,
    gtb_trees, gtb_shrinkage, gtb_sampling_rate, gtb_max_nodes,
    agb_year,
    _cache_buster=None
):
    sample = sample_and_split(_project_id, county_selection, num_pixels, train_split, seed, agb_year, _cache_buster)
    training_set         = sample["training_set"]
    dependent_variable   = sample["dependent_variable"]
    predictor_band_names = sample["predictor_band_names"]

    rf_vars = int(rf_vars_per_split) if rf_vars_per_split and int(rf_vars_per_split) > 0 else None
    
    rf_classifier = ee.Classifier.smileRandomForest(
        numberOfTrees=rf_trees, variablesPerSplit=rf_vars,
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

@st.cache_resource(show_spinner=False)
def compute_regional_statistics(_project_id, _county_selection, _estimated_carbon, _forest_loss, _selected_fc, model_name):
    pixel_area_ha = ee.Image.pixelArea().divide(10000)
    total_carbon = _estimated_carbon.multiply(pixel_area_ha)
    deforested_area = _forest_loss.gt(0).multiply(pixel_area_ha)
    
    stats_image = (
        _estimated_carbon.rename("mean_density")
        .addBands(total_carbon.rename("total_carbon"))
        .addBands(pixel_area_ha.rename("total_area"))
        .addBands(deforested_area.rename("deforested_area"))
    )
    
    stats = stats_image.reduceRegion(
        reducer=ee.Reducer.mean().combine(ee.Reducer.sum(), sharedInputs=True),
        geometry=_selected_fc.geometry(),
        scale=1000,  # Increased from 300m to 1km to prevent memory limits over large areas
        maxPixels=1e10,
        tileScale=16,
        bestEffort=True
    ).getInfo()
    
    return {
        "mean_density": stats.get("mean_density_mean", 0),
        "total_carbon": stats.get("total_carbon_sum", 0),
        "total_area": stats.get("total_area_sum", 0),
        "deforested_area": stats.get("deforested_area_sum", 0)
    }
