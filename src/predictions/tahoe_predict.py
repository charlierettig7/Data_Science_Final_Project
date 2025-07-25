import pandas as pd
import lightgbm as lgb
import xgboost as xgb
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score
import scipy.stats as stats
from joblib import load

# --- Model 1: LGBM With Variants ---
print("--- Processing Model with Variants ---", flush=True)
tahoe_vars = pd.read_parquet('./data/tahoe_formatted_normalized_separate_scaler_for_genes_same_scaler_for_dosage_cellosaurus.parquet')
print("Tahoe data (with variants) loaded.", flush=True)

# Load the trained LightGBM model
model_path_vars = './results/results_vars/lightGBM_results_vars/lightgbm_vars_scaled_notune_model.txt'
bst_vars = lgb.Booster(model_file=model_path_vars)
print("Model (with variants) loaded successfully.", flush=True)

# Prepare data and make predictions
feature_names_vars = bst_vars.feature_name()
X_tahoe_vars = tahoe_vars[feature_names_vars]
X_tahoe_vars = X_tahoe_vars.fillna(0)
predictions_vars = bst_vars.predict(X_tahoe_vars)

# Calculate RMSE
y_true_vars = tahoe_vars['responses']
rmse_vars = np.sqrt(mean_squared_error(y_true_vars, predictions_vars))
print(f"RMSE (with variants): {rmse_vars:.4f}\n")

# Calculate R2 Score
r2_vars = r2_score(y_true_vars, predictions_vars)
print(f"R2 (with variants): {r2_vars:.4f}\n")


# --- Model 2: LGBM Without Variants (Raw) ---
print("--- Processing Model without Variants ---", flush=True)
# Assuming this is the correct dataset for the raw model
lincs_raw_ref = pd.read_parquet('./data/signature_response_features_r2_top0.7_final.parquet')
col_diff = tahoe_vars.columns.difference(lincs_raw_ref.columns)
tahoe_raw = tahoe_vars.drop(columns=col_diff)
for col in col_diff:
    print(col, flush = True)
print("Tahoe data (without variants, reference) loaded.", flush=True)


# Load the trained LightGBM model
# Correcting the path to the model trained on raw data
model_path_raw = './results/results_raw/lightGBM_results_raw/lightgbm_raw_scaled_notune_model.txt'
bst_raw = lgb.Booster(model_file=model_path_raw)
print("Model (without variants) loaded successfully.", flush=True)


# Prepare data and make predictions
# The target column name in the raw data might be 'response', changing to 'responses' to match the other dataset
if 'response' in tahoe_raw.columns and 'responses' not in tahoe_raw.columns:
    tahoe_raw.rename(columns = {'response': 'responses'}, inplace=True)
# The feature name might be 'dose', changing to 'nearest_dose' to match the model
if 'dose' in tahoe_raw.columns and 'nearest_dose' not in tahoe_raw.columns:
    tahoe_raw.rename(columns = {'dose': 'nearest_dose'}, inplace=True)

feature_names_raw = bst_raw.feature_name()
X_tahoe_raw = tahoe_raw[feature_names_raw]
X_tahoe_raw = X_tahoe_raw.fillna(0)
predictions_raw = bst_raw.predict(X_tahoe_raw)

# Calculate RMSE
y_true_raw = tahoe_raw['responses']
rmse_raw = np.sqrt(mean_squared_error(y_true_raw, predictions_raw))
print(f"RMSE (without variants): {rmse_raw:.4f}\n")

r2_raw = r2_score(y_true_raw, predictions_raw)
print(f"R2 (without variants): {r2_raw:.4f}\n")


# --- Model 3: XGBoost Without Variants (Raw) ---
print("--- Processing XGBoost Model without Variants ---", flush=True)
model_path_xgb = './results/results_raw/XGBoost_results_raw/xgboost_model_raw_scaled_notune.json'
bst_xgb = xgb.XGBRegressor()
bst_xgb.load_model(model_path_xgb)
print("Model (XGBoost) loaded successfully.", flush=True)

# Prepare data and make predictions
# We can reuse the 'tahoe_raw' DataFrame as it uses the same source data
feature_names_xgb = bst_xgb.feature_names_in_
X_tahoe_xgb = tahoe_raw[feature_names_xgb]
X_tahoe_xgb = X_tahoe_xgb.fillna(0)
predictions_xgb = bst_xgb.predict(X_tahoe_xgb)
# Capture raw XGBoost predictions before override
predictions_xgb_raw = predictions_xgb

# Calculate RMSE
# The y_true_raw is the same as for the LGBM raw model
rmse_xgb = np.sqrt(mean_squared_error(y_true_raw, predictions_xgb))
print(f"RMSE (XGBoost without variants): {rmse_xgb:.4f}\n")

r2_xgb = r2_score(y_true_raw, predictions_xgb)
print(f"R2 (XGBoost without variants): {r2_xgb:.4f}\n")


# --- Model 4: XGBoost With Variants ---
print("--- Processing XGBoost Model with Variants ---", flush=True)
model_path_xgb = './results/results_vars/XGBoost_results_vars/xgboost_model_vars_scaled_notune.json'
bst_xgb = xgb.XGBRegressor()
bst_xgb.load_model(model_path_xgb)
print("Model (XGBoost) loaded successfully.", flush=True)

if 'dose' in tahoe_raw.columns and 'nearest_dose' not in tahoe_raw.columns:
    tahoe_raw.rename(columns={'dose': 'nearest_dose'}, inplace=True)

# Prepare data and make predictions
feature_names_xgb = bst_xgb.feature_names_in_
X_tahoe_xgb = tahoe_vars[feature_names_xgb]
X_tahoe_xgb = X_tahoe_xgb.fillna(0)
predictions_xgb = bst_xgb.predict(X_tahoe_xgb)
# Capture XGBoost variant predictions
predictions_xgb_vars = predictions_xgb

# Calculate RMSE
rmse_xgb_vars = np.sqrt(mean_squared_error(y_true_vars, predictions_xgb_vars))
print(f"RMSE (XGBoost with variants): {rmse_xgb_vars:.4f}\n")

r2_xgb_vars = r2_score(y_true_vars, predictions_xgb_vars)
print(f"R2 (XGBoost with variants): {r2_xgb_vars:.4f}\n")

# --- Model 5: Ridge Regression Without Variants (Raw) ---
print("--- Processing Ridge Regression without Variants ---", flush=True)
ridge_raw = load('./results/results_raw/linear_results_raw/ridge_raw_notune_scaled_model.joblib')
predictions_ridge_raw = ridge_raw.predict(X_tahoe_raw)
rmse_ridge_raw = np.sqrt(mean_squared_error(y_true_raw, predictions_ridge_raw))
print(f"RMSE (Ridge without variants): {rmse_ridge_raw:.4f}\n")
r2_ridge_raw = r2_score(y_true_raw, predictions_ridge_raw)
print(f"R2 (Ridge without variants): {r2_ridge_raw:.4f}\n")

# Log-transform target and predictions for Ridge without variants
y_true_ridge_raw_log = np.log1p(y_true_raw)
predictions_ridge_raw_log = np.log1p(predictions_ridge_raw)
rmse_ridge_raw_log = np.sqrt(mean_squared_error(y_true_ridge_raw_log, predictions_ridge_raw_log))
r2_ridge_raw_log = r2_score(y_true_ridge_raw_log, predictions_ridge_raw_log)
print(f"Log RMSE (Ridge without variants): {rmse_ridge_raw_log:.4f}\n")
print(f"Log R2 (Ridge without variants): {r2_ridge_raw_log:.4f}\n")

# --- Model 6: Ridge Regression With Variants ---
print("--- Processing Ridge Regression with Variants ---", flush=True)
ridge_vars = load('./results/results_vars/linear_results_vars/ridge_vars_notune_scaled_model.joblib')
predictions_ridge_vars = ridge_vars.predict(X_tahoe_vars)
rmse_ridge_vars = np.sqrt(mean_squared_error(y_true_vars, predictions_ridge_vars))
print(f"RMSE (Ridge with variants): {rmse_ridge_vars:.4f}\n")
r2_ridge_vars = r2_score(y_true_vars, predictions_ridge_vars)
print(f"R2 (Ridge with variants): {r2_ridge_vars:.4f}\n")

# Log-transform target and predictions for Ridge with variants
y_true_ridge_vars_log = np.log1p(y_true_vars)
predictions_ridge_vars_log = np.log1p(predictions_ridge_vars)
rmse_ridge_vars_log = np.sqrt(mean_squared_error(y_true_ridge_vars_log, predictions_ridge_vars_log))
r2_ridge_vars_log = r2_score(y_true_ridge_vars_log, predictions_ridge_vars_log)
print(f"Log RMSE (Ridge with variants): {rmse_ridge_vars_log:.4f}\n")

print(f"Log R2 (Ridge with variants): {r2_ridge_vars_log:.4f}\n")

# --- Generating Log-Scale RMSE Comparison Plot for Ridge Models ---
print("--- Generating Log-Scale RMSE Comparison Plot for Ridge Models ---", flush=True)
model_names_log = ['Ridge Without Variants (Raw Log)', 'Ridge With Variants (Log)']
rmse_log_values = [rmse_ridge_raw_log, rmse_ridge_vars_log]
fig, ax = plt.subplots(figsize=(8, 6))
bars = ax.bar(model_names_log, rmse_log_values)
ax.set_ylabel('Log-Scale Root Mean Squared Error (RMSE)', fontsize=12)
ax.set_title('Log-Scale RMSE Comparison of Ridge Models', fontsize=14, pad=20)
ax.set_ylim(0, max(rmse_log_values) * 1.2)
for bar in bars:
    yval = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2.0, yval, f'{yval:.4f}', va='bottom', ha='center')
log_rmse_plot_path = './results/tahoe_predictions/log_rmse_comparison_ridge.png'
plt.savefig(log_rmse_plot_path, dpi=300, bbox_inches='tight')
print(f"Log RMSE bar plot saved to: {log_rmse_plot_path}")

# --- Generating Log-Scale R2 Comparison Plot for Ridge Models ---
print("--- Generating Log-Scale R2 Comparison Plot for Ridge Models ---", flush=True)
r2_log_values = [r2_ridge_raw_log, r2_ridge_vars_log]
fig, ax = plt.subplots(figsize=(8, 6))
bars = ax.bar(model_names_log, r2_log_values)
ax.set_ylabel('Log-Scale R2 Score', fontsize=12)
ax.set_title('Log-Scale R2 Comparison of Ridge Models', fontsize=14, pad=20)
ax.set_ylim(min(r2_log_values) - 0.1, max(r2_log_values) + 0.1)
for bar in bars:
    yval = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2.0, yval, f'{yval:.4f}', va='bottom', ha='center')
log_r2_plot_path = './results/tahoe_predictions/log_r2_comparison_ridge.png'
plt.savefig(log_r2_plot_path, dpi=300, bbox_inches='tight')
print(f"Log R2 bar plot saved to: {log_r2_plot_path}")

# --- Assumption Checks for Ridge Models ---

# Residuals vs Fitted for Ridge without variants
# Use log-transformed residuals for Ridge without variants
residuals_ridge_raw = y_true_ridge_raw_log - predictions_ridge_raw_log
plt.figure(figsize=(8, 6))
plt.scatter(predictions_ridge_raw_log, residuals_ridge_raw)
plt.axhline(y=0, linestyle='--', linewidth=1, color='gray')
plt.xlabel('Log of fitted values')
plt.ylabel('Log residuals')
plt.title('Residuals vs Fitted (Ridge without variants)')
plt.savefig('./results/results_raw/linear_results_raw/plots/residuals_vs_fitted_ridge_raw_log.png', dpi=300, bbox_inches='tight')
plt.close()

# Scale-Location Plot for Ridge without variants
standardized_residuals_raw = np.sqrt(np.abs((residuals_ridge_raw - np.mean(residuals_ridge_raw)) / np.std(residuals_ridge_raw)))
plt.figure(figsize=(8, 6))
plt.scatter(predictions_ridge_raw, standardized_residuals_raw)
plt.xlabel('Fitted values')
plt.ylabel('Sqrt(Standardized residuals)')
plt.title('Scale-Location (Ridge without variants)')
plt.savefig('./results/results_raw/linear_results_raw/plots/scale_location_ridge_raw_log.png', dpi=300, bbox_inches='tight')
plt.close()

# Q-Q Plot for Ridge without variants
plt.figure(figsize=(8, 6))
stats.probplot(residuals_ridge_raw, dist="norm", plot=plt)
plt.title('Q-Q Plot (Ridge without variants)')
plt.savefig('./results/results_raw/linear_results_raw/plots/qq_plot_ridge_raw_log.png', dpi=300, bbox_inches='tight')
plt.close()

# Residuals vs Fitted for Ridge with variants
# Use log-transformed residuals for Ridge with variants
residuals_ridge_vars = y_true_ridge_vars_log - predictions_ridge_vars_log
plt.figure(figsize=(8, 6))
plt.scatter(predictions_ridge_vars, residuals_ridge_vars)
plt.axhline(y=0, linestyle='--', linewidth=1, color='gray')
plt.xlabel('Fitted values')
plt.ylabel('Residuals')
plt.title('Residuals vs Fitted (Ridge with variants)')
plt.savefig('./results/results_vars/linear_results_vars/plots/residuals_vs_fitted_ridge_vars_log.png', dpi=300, bbox_inches='tight')
plt.close()

# Scale-Location Plot for Ridge with variants
standardized_residuals_vars = np.sqrt(np.abs((residuals_ridge_vars - np.mean(residuals_ridge_vars)) / np.std(residuals_ridge_vars)))
plt.figure(figsize=(8, 6))
plt.scatter(predictions_ridge_vars, standardized_residuals_vars)
plt.xlabel('Fitted values')
plt.ylabel('Sqrt(Standardized residuals)')
plt.title('Scale-Location (Ridge with variants)')
plt.savefig('./results/results_vars/linear_results_vars/plots/scale_location_ridge_vars_log.png', dpi=300, bbox_inches='tight')
plt.close()

# Q-Q Plot for Ridge with variants
plt.figure(figsize=(8, 6))
stats.probplot(residuals_ridge_vars, dist="norm", plot=plt)
plt.title('Q-Q Plot (Ridge with variants)')
plt.savefig('./results/results_vars/linear_results_vars/plots/qq_plot_ridge_vars_log.png', dpi=300, bbox_inches='tight')
plt.close()

# --- Create and save bar plot ---
print("--- Generating RMSE Comparison Plot ---", flush=True)
model_names = ['LGBM With Variants', 'LGBM Without Variants', 'XGBoost With Variants', 'XGBoost Without Variants', 'Ridge Without Variants', 'Ridge With Variants']
rmse_values = [rmse_vars, rmse_raw, rmse_xgb_vars, rmse_xgb, rmse_ridge_raw, rmse_ridge_vars]

plt.style.use('seaborn-v0_8-whitegrid')
fig, ax = plt.subplots(figsize=(10, 6))

bars = ax.bar(model_names, rmse_values, color=['#4C72B0', "#9083E5", "#EE8122", "#EDA460"])

# Add labels and title
ax.set_ylabel('Root Mean Squared Error (RMSE)', fontsize=12)
ax.set_title('RMSE Comparison of Models (Tahoe Dataset)', fontsize=14, pad=20)
ax.set_ylim(0, max(rmse_values) * 1.2)

# Add the value on top of each bar
for bar in bars:
    yval = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2.0, yval, f'{yval:.4f}', va='bottom', ha='center')

# Save plot
plot_path = './results/tahoe_predictions/rmse_comparison_barchart.png'
plt.savefig(plot_path, dpi=300, bbox_inches='tight')

print(f"Bar plot saved to: {plot_path}")

# --- Generating R2 Comparison Plot ---
model_names = ['LGBM With Variants', 'LGBM Without Variants', 'XGBoost With Variants', 'XGBoost Without Variants', 'Ridge Without Variants', 'Ridge With Variants']
r2_values = [r2_vars, r2_raw, r2_xgb_vars, r2_xgb, r2_ridge_raw, r2_ridge_vars]
fig, ax = plt.subplots(figsize=(10, 6))
bars = ax.bar(model_names, r2_values)
ax.set_ylabel('R2 Score', fontsize=12)
ax.set_title('R2 Score Comparison of Models (Tahoe Dataset)', fontsize=14, pad=20)

# Add the value on top of each bar
for bar in bars:
    yval = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2.0, yval, f'{yval:.4f}', va='bottom', ha='center')
r2_plot_path = './results/tahoe_predictions/r2_comparison_barchart.png'
plt.savefig(r2_plot_path, dpi=300, bbox_inches='tight')
print(f"Bar plot saved to: {r2_plot_path}")

# --- Save predictions along with input indices for each model ---
results_dir = './results/tahoe_predictions'
# LGBM With Variants
pd.DataFrame({'index': tahoe_vars.index, 'prediction': predictions_vars}) \
    .to_csv(f"{results_dir}/predictions_lgbm_vars.csv", index=False)
# LGBM Without Variants
pd.DataFrame({'index': tahoe_raw.index, 'prediction': predictions_raw}) \
    .to_csv(f"{results_dir}/predictions_lgbm_raw.csv", index=False)
# XGBoost Without Variants (Raw)
pd.DataFrame({'index': tahoe_raw.index, 'prediction': predictions_xgb_raw}) \
    .to_csv(f"{results_dir}/predictions_xgboost_raw.csv", index=False)
# XGBoost With Variants
pd.DataFrame({'index': tahoe_vars.index, 'prediction': predictions_xgb_vars}) \
    .to_csv(f"{results_dir}/predictions_xgboost_vars.csv", index=False)
# Ridge Without Variants
pd.DataFrame({'index': tahoe_raw.index, 'prediction': predictions_ridge_raw}) \
    .to_csv(f"{results_dir}/predictions_ridge_raw.csv", index=False)
# Ridge With Variants
pd.DataFrame({'index': tahoe_vars.index, 'prediction': predictions_ridge_vars}) \
    .to_csv(f"{results_dir}/predictions_ridge_vars.csv", index=False)
print("Saved all model predictions with indices to CSV files.")
