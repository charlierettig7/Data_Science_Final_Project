import pandas as pd
import lightgbm as lgb
import xgboost as xgb
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats

# LightGBM with variants
pat_data_vars = pd.read_parquet('./data/pat6_to_predict.parquet')
pat_data_vars.drop(columns=['responses'], inplace = True)
pat_data_vars = pat_data_vars.fillna(0)

model_path_vars = './results/results_vars/lightGBM_results_vars/lightgbm_vars_scaled_notune_model.txt'
bst_vars = lgb.Booster(model_file=model_path_vars)

predictions_vars = bst_vars.predict(pat_data_vars)

pd.DataFrame({'index': pat_data_vars.index, 'prediction': predictions_vars}) \
    .to_csv('results/patient_predictions/patient_predictions_lgbm_vars.csv', index=False)


# LightGBM without variants
lincs_raw_ref = pd.read_parquet('./data/signature_response_features_r2_top0.7_final.parquet')
col_diff = pat_data_vars.columns.difference(lincs_raw_ref.columns)
pat_data_raw = pat_data_vars.drop(columns=col_diff)
for col in col_diff:
    print(col, flush = True)
print("Patient data (without variants, reference) loaded.", flush=True)

model_path_raw = './results/results_raw/lightGBM_results_raw/lightgbm_raw_scaled_notune_model.txt'
bst_raw = lgb.Booster(model_file=model_path_raw)
print("Model (without variants) loaded successfully.", flush=True)

predictions_raw = bst_raw.predict(pat_data_raw)

# LGBM Without Variants
pd.DataFrame({'index': pat_data_raw.index, 'prediction': predictions_raw}) \
    .to_csv(f"results/patient_predictions/patient_predictions_lgbm_raw.csv", index=False)