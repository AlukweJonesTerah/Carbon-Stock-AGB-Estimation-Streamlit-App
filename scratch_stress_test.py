import ee
import time
import sys
import os

# Ensure the src module can be found
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.ee_processing import train_models, build_predictor_stack, sample_and_split
from src.config import ALL_KENYA_COUNTIES_OPTIONS

from src.ee_auth import _ee_credentials_from_file

def run_stress_test():
    print("Initializing Earth Engine...")
    project_id = "gen-lang-client-0365181451"
    try:
        creds = _ee_credentials_from_file()
        if creds:
            ee.Initialize(credentials=creds, project=project_id)
        else:
            ee.Initialize(project=project_id)
    except Exception as e:
        print(f"Could not initialize EE: {e}")
        return

    print(f"\n--- STRESS TEST 1: MAXIMUM EXTENT ({len(ALL_KENYA_COUNTIES_OPTIONS)} COUNTIES) ---")
    print("Training models with 15,000 stratified samples and 250 trees...")
    start_time = time.time()
    
    try:
        models = train_models(
            _project_id=project_id,
            county_selection=ALL_KENYA_COUNTIES_OPTIONS,
            num_pixels=15000,
            train_split=0.8,
            seed=42,
            rf_trees=250, rf_vars_per_split=0, rf_min_leaf=1,
            svm_gamma=1.0, svm_cost=1.0,
            gtb_trees=200, gtb_shrinkage=0.05, gtb_sampling_rate=0.7, gtb_max_nodes=10,
            agb_year="2020"
        )
        print(f"[SUCCESS] Models trained in {time.time() - start_time:.2f} seconds.")
    except Exception as e:
        print(f"[FAILED] Stress Test 1 Failed: {e}")
        return
        
    print(f"\n--- STRESS TEST 2: ZONAL REDUCTIONS AT EXTREME SCALE ---")
    print("Forcing Earth Engine to evaluate the RF model across the entire landscape...")
    start_time = time.time()
    try:
        stack = build_predictor_stack(project_id, ALL_KENYA_COUNTIES_OPTIONS, "2020")
        estimated_rf = stack["predictor_variables"].classify(models["rf_model"])
        
        # Run a reduceRegion over the entire 28 counties
        stats = estimated_rf.reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=stack["selected_fc"].geometry(),
            scale=1000,
            maxPixels=1e13,
            tileScale=16,
            bestEffort=True
        ).getInfo()
        print(f"[SUCCESS] Zonal reduction completed in {time.time() - start_time:.2f} seconds.")
        print(f"Result: {stats}")
    except Exception as e:
        print(f"[FAILED] Stress Test 2 Failed: {e}")

if __name__ == '__main__':
    run_stress_test()
