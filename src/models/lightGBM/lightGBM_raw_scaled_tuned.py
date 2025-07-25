print("importing libraries...", flush = True)
import pandas as pd
import lightgbm as lgb 
from sklearn.model_selection import GridSearchCV, GroupShuffleSplit
from sklearn.metrics import r2_score
import numpy as np
import matplotlib.pyplot as plt
import json
import time
import resource

# Load in data
print("--- Importing Data ---", flush = True)
file_path = './data/lincs_formatted_normalized_cellosaurus.parquet'
file_path_no_vars = './data/signature_response_features_r2_top0.7_final.parquet'
data_load = pd.read_parquet(file_path)
data_load_no_vars = pd.read_parquet(file_path_no_vars)

# Only data_load (which has variant calling data) contains normalized data,
# so drop variants columns for raw model
col_diff = data_load.columns.difference(data_load_no_vars.columns)
data_load = data_load.drop(columns = col_diff)

# Filter out NA and inf. values from target column
label = 'responses'
print(f"Num rows before NA filtering: {len(data_load_no_vars)}", flush = True)
good = data_load[label].notnull() & np.isfinite(data_load[label])
print(f"Num rows after NA Filtering {len(good)}", flush = True)
data = data_load.loc[good].reset_index(drop=True)

# Drop unnecessary columns, extract groups for stratification (leakage prevention)
y = data[label]
groups = data['cellosaurus_id']
x = data.drop(columns = [label, 'sig_id', 'cellosaurus_id'])

# --- Outer hold-out split by cellosaurus_id ---
gss_outer = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=1)
train_idx, outer_val_idx = next(gss_outer.split(x, y, groups))
x_train = x.iloc[train_idx]
y_train = y.iloc[train_idx]
x_outer_val = x.iloc[outer_val_idx]
y_outer_val = y.iloc[outer_val_idx]

groups_train = groups.iloc[train_idx]
print(f"Full training set size: {len(x_train)} samples", flush=True)

# --- Create pilot subset for fast hyperparameter tuning ---
gss_pilot = GroupShuffleSplit(n_splits=1, test_size=0.7, random_state=42)
pilot_idx, _ = next(gss_pilot.split(x_train, y_train, groups_train))
X_pilot = x_train.iloc[pilot_idx]
y_pilot = y_train.iloc[pilot_idx]
groups_pilot = groups_train.iloc[pilot_idx]
print(f"Pilot set size: {len(X_pilot)} samples", flush=True)

# --- Inner group-aware CV splitter for GridSearch ---
cv = GroupShuffleSplit(n_splits=5, test_size=0.2, random_state=2)

# --- Hyperparameter Grid Search Setup ---
print("--- Setting up GridSearchCV ---", flush = True)

# Define the model using the scikit-learn wrapper
lgbm = lgb.LGBMRegressor(objective='regression_l2', 
                        metric='rmse',
                        random_state=1, 
                        n_jobs=1, 
                        num_iterations = 5, 
                        max_depth = 5, 
                        bagging_fraction = 0.7,
                        bagging_freq = 5,
                        learning_rate = 0.5
                        )


# Define the grid of hyperparameters to search
param_grid = {
    'num_leaves': list(np.arange(10, 21, 5)),
    'lambda_l2': list(np.arange(0, 3, 1)),
    'max_bin': list(np.arange(100, 351, 50))
}

# Configure the grid search. refit=False means we will manually retrain the best model.
grid_search = GridSearchCV(
    estimator=lgbm,
    param_grid=param_grid,
    scoring='neg_root_mean_squared_error',
    cv=cv,  # stratified 5-fold cross-validation
    n_jobs=-1,  # Use all available CPU cores
    verbose=2,  # Show progress
    refit='neg_root_mean_squared_error' # We will retrain manually to get evaluation results
)

print("--- Starting Grid Search on pilot set ---", flush=True)
# Benchmark grid search time and memory usage
mem_before_gs = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
time_start_gs = time.time()
grid_search.fit(X_pilot, y_pilot, groups=groups_pilot)
time_end_gs = time.time()
mem_after_gs = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
grid_search_time = time_end_gs - time_start_gs
grid_search_mem_usage = mem_after_gs - mem_before_gs
print(f"Grid search time: {grid_search_time:.2f} seconds", flush=True)
print(f"Grid search memory increased: {grid_search_mem_usage} KB (peak: {mem_after_gs} KB)", flush=True)
print("--- Finished Grid Search ---", flush = True)
print(f"Best parameters found: {grid_search.best_params_}", flush = True)
# Convert numpy scalar parameters to native Python types for JSON serialization
best_params_safe = {k: (v.item() if hasattr(v, "item") else v) for k, v in grid_search.best_params_.items()}
print(f"Best parameters (safe): {best_params_safe}", flush=True)

# --- Retrain final model with best parameters for full evaluation ---
print("\n--- Retraining final model with best parameters ---", flush=True)

# Get best params and add other necessary ones for retraining with early stopping
final_params = grid_search.best_params_
final_params.update({
    'objective': 'regression_l2',
    'metric': 'rmse',
    'random_state': 1,
    'n_jobs': -1,
    'num_iterations': 500  # High value, early stopping will find the optimal number
})

final_model = lgb.LGBMRegressor(**final_params)

# --- Fresh train/validation split for early stopping ---
gss_final = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
tf_idx, fv_idx = next(gss_final.split(x_train, y_train, groups_train))
X_tf, y_tf = x_train.iloc[tf_idx], y_train.iloc[tf_idx]
X_fv, y_fv = x_train.iloc[fv_idx], y_train.iloc[fv_idx]

# Benchmark final model training time and memory usage
mem_before_train = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
time_start_train = time.time()
final_model.fit(
    X_tf, y_tf,
    eval_set=[(X_tf, y_tf), (X_fv, y_fv)],
    eval_names=['train', 'early_val'],
    eval_metric='rmse'
)
time_end_train = time.time()
mem_after_train = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
training_time = time_end_train - time_start_train
mem_usage = mem_after_train - mem_before_train
print(f"Final model training time: {training_time:.2f} seconds", flush=True)
print(f"Final model memory increased: {mem_usage} KB (peak: {mem_after_train} KB)", flush=True)
print("--- Finished retraining ---", flush = True)

final_model.booster_.save_model('results/results_raw/lightGBM_results_raw/lightgbm_raw_tuned_model.txt') # CHANGE

# --- Performance Analysis ---
print("\n--- Performance Analysis ---", flush = True)

# 1. Plot training and validation loss from the retrained model
print("Plotting training and validation loss...", flush = True)
evals_result = final_model.evals_result_
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(evals_result['train']['rmse'], label='Train RMSE')
ax.plot(evals_result['early_val']['rmse'], label='Early Validation RMSE')
ax.axvline(x=final_model.best_iteration_, color='r', linestyle='--', label=f'Best Iteration ({final_model.best_iteration_})')
ax.legend()
plt.ylabel('RMSE')
plt.xlabel('Training Iteration')
plt.title('LightGBM RMSE vs. Training Iteration (No Variants, Normalized)') # CHANGE
plt.grid(True)
plot_filename = 'results/results_raw/lightGBM_results_raw/lightgbm_raw_tunedloss_plot.png' # CHANGE
plt.savefig(plot_filename)
print(f"Plot saved to {plot_filename}", flush=True)



# --- Saving evaluation metrics ---
print("\nSaving evaluation metrics...", flush=True)
metrics = {
    'best_params_from_grid_search': best_params_safe,
    'best_cv_rmse_from_grid_search': -grid_search.best_score_,
    'retrained_model_best_iteration': final_model.best_iteration_,
    'early_stop_best_validation_rmse': final_model.best_score_['early_val']['rmse'],
    'training_rmse_history': evals_result['train']['rmse'],
    'early_stop_validation_rmse_history': evals_result['early_val']['rmse'],
    'grid_search_time_seconds': grid_search_time,
    'grid_search_memory_usage_kb': grid_search_mem_usage,
    'training_time_seconds': training_time,
    'training_memory_usage_kb': mem_usage,
    'peak_memory_kb': mem_after_train
}

# Final evaluation on outer hold-out set
print("\n--- Final evaluation on outer hold-out set ---", flush=True)
from sklearn.metrics import mean_squared_error, r2_score
y_outer_pred = final_model.predict(x_outer_val)
mse_outer = mean_squared_error(y_outer_val, y_outer_pred)
rmse_outer = np.sqrt(mse_outer)
r2_outer = r2_score(y_outer_val, y_outer_pred)
print(f"Final Test RMSE: {rmse_outer:.4f}", flush=True)
print(f"Final Test R2:   {r2_outer:.4f}", flush=True)
metrics['final_test_rmse'] = rmse_outer
metrics['final_test_r2'] = r2_outer

metrics_filename = 'results/results_raw/lightGBM_results_raw/lightgbm_raw_tuned_evaluation_metrics.txt' # CHANGE
with open(metrics_filename, 'w') as f:
    json.dump(metrics, f, indent=4)
print(f"Metrics saved to {metrics_filename}", flush=True)

# 3. Save feature importance
print("\nSaving feature importance...", flush = True)
importance = final_model.booster_.feature_importance(importance_type='gain')
feature_names = x_train.columns
sorted_importance = sorted(zip(feature_names, importance), key=lambda x: x[1], reverse=True)
importance_filename = 'results/results_raw/lightGBM_results_raw/lightgbm_raw_tuned_feature_importance.txt' # CHANGE
with open(importance_filename, 'w') as f:
    f.write("Feature Importance (Gain)\n")
    f.write("===========================\n")
    for feature, score in sorted_importance:
        f.write(f"{feature}: {score}\n")
print(f"Feature importance saved to {importance_filename}", flush = True)

# Plot top 20 feature importances
print("\nPlotting top 20 feature importances...", flush = True)
top20 = sorted_importance[:20]
features = [f for f, s in top20]
scores = [s for f, s in top20]
fig, ax = plt.subplots(figsize=(10, 8))
ax.barh(features[::-1], scores[::-1])
ax.set_xlabel('Importance (Gain)')
ax.set_title('LightGBM Top 20 Feature Importances (No Variants, Normalized)')
plt.tight_layout()
fi_plot_filename = 'results/results_raw/lightGBM_results_raw/lightgbm_raw_tuned_top20_feature_importance.png' # CHANGE
plt.savefig(fi_plot_filename)
print(f"Feature importance plot saved to {fi_plot_filename}", flush=True)
