import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import matplotlib.pyplot as plt
import json

# Load in data
print("--- Importing Data ---", flush = True)
file_path = './data/signature_response_features_r2_top0.7_final.parquet'
data_load_no_vars = pd.read_parquet(file_path)

# Filter out NA and inf. values from target column
label = 'responses'
good = data_load_no_vars[label].notnull() & np.isfinite(data_load_no_vars[label])
data = data_load_no_vars.loc[good].reset_index(drop=True)
y = data[label]
x = data.drop(label, axis=1)

# Extract groups to ensure no overlapping cell lines and drop metadata columns
groups = x['cellosaurus_id']
x.drop(columns=['sig_id', 'pert_id', 'cellosaurus_id', 'inchi_key', 'cell_drug', 
                'cmap_name', 'smiles'], inplace=True)
y = data[label]

# Automatically identify object columns as categorical for robustness
categorical_cols = x.select_dtypes(include=['object']).columns.tolist()
print(f"Identified categorical columns: {categorical_cols}", flush = True)
for col in categorical_cols:
    x[col] = x[col].astype('category')

# Split data into train and temp (val+test) ensuring unique cell lines
gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=1)
train_idx, temp_idx = next(gss.split(x, y, groups))
x_train = x.iloc[train_idx]
y_train = y.iloc[train_idx]
temp_x = x.iloc[temp_idx]
temp_y = y.iloc[temp_idx]
temp_groups = groups.iloc[temp_idx]

# Split temp into validation and test sets equally, with no overlapping cell lines
gss2 = GroupShuffleSplit(n_splits=1, test_size=0.5, random_state=1)
val_idx_rel, test_idx_rel = next(gss2.split(temp_x, temp_y, temp_groups))
x_val = temp_x.iloc[val_idx_rel]
y_val = temp_y.iloc[val_idx_rel]
x_test = temp_x.iloc[test_idx_rel]
y_test = temp_y.iloc[test_idx_rel]
    
train_data = lgb.Dataset(x_train, label=y_train)
val_data = lgb.Dataset(x_val, label=y_val, reference=train_data)

# Model parameters
param = {
    'objective': 'regression_l2',
    'metric': 'rmse',
    'boosting': 'gbdt', 
    'learning_rate': 0.01, # default
    'num_leaves': 31, # default
    'num_threads': 0, # default, docs say to not change this during training
    'deterministic': False, # default, this is for reproducibility in some contexts, but slows down training
    'max_depth': -1, # default, no limit to max depth of generated trees
    'verbosity': -1, # Suppress verbose output
    'max_bin': 255 # default, but affects bias-variance tradeoff
}

num_rounds = 500 

# Train model
print("--- Starting LightGBM Training ---", flush = True)
evals_result = {}
bst = lgb.train(
    param,
    train_data,
    num_boost_round=num_rounds,
    valid_sets=[train_data, val_data],
    valid_names=['train', 'eval'],
    callbacks=[
        lgb.early_stopping(20, verbose=True),
        lgb.record_evaluation(evals_result),
        lgb.log_evaluation(period=1)
    ]
)
print("--- Finished LightGBM Training ---", flush = True)

bst.save_model('./results/results_raw/lightGBM_results_raw/lightgbm_raw_notune_model.txt')

# --- Performance Analysis ---
print("\n--- Performance Analysis ---", flush = True)

# Plot training and validation loss
print("Plotting training and validation loss...", flush = True)
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(evals_result['train']['rmse'], label='Train RMSE')
ax.plot(evals_result['eval']['rmse'], label='Validation RMSE')
ax.axvline(x=bst.best_iteration, color='r', linestyle='--', label=f'Best Iteration ({bst.best_iteration})')
ax.legend()
plt.ylabel('RMSE')
plt.xlabel('Training Iteration')
plt.title('LightGBM RMSE vs. Training Iteration (No Variants, Non-Normalized)')
plt.grid(True)
plot_filename = './results/results_raw/lightGBM_results_raw/lightgbm_raw_notune_loss_plot.png'
plt.savefig(plot_filename)
print(f"Plot saved to {plot_filename}", flush = True)

# Make predictions on test set and calculate metrics
print("\nEvaluating on test set...", flush=True)
y_pred_test = bst.predict(x_test, num_iteration=bst.best_iteration)
test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
test_r2 = r2_score(y_test, y_pred_test)
print(f"Test RMSE: {test_rmse:.4f}", flush=True)
print(f"Test R^2: {test_r2:.4f}", flush=True)

# Save evaluation metrics to a file
print("\nSaving evaluation metrics...", flush = True)

metrics = {
    'best_iteration': bst.best_iteration,
    'best_validation_rmse': bst.best_score['eval']['rmse'],
    'test_rmse': test_rmse,
    'test_r2': test_r2,
    'training_rmse_history': evals_result['train']['rmse'],
    'validation_rmse_history': evals_result['eval']['rmse']
}

metrics_filename = './results/results_raw/lightGBM_results_raw/lightgbm_raw_notune_evaluation_metrics.txt'
with open(metrics_filename, 'w') as f:
    json.dump(metrics, f, indent=4)
print(f"Metrics saved to {metrics_filename}")

# Save feature importance
print("\nSaving feature importance...")
importance = bst.feature_importance(importance_type='gain')
feature_names = x_train.columns
sorted_importance = sorted(zip(feature_names, importance), key=lambda x: x[1], reverse=True)

importance_filename = './results/results_raw/lightGBM_results_raw/lightgbm_raw_notune_feature_importance.txt'
with open(importance_filename, 'w') as f:
    f.write("Feature Importance (Gain)\n")
    f.write("===========================\n")
    for feature, score in sorted_importance:
        f.write(f"{feature}: {score}\n")
print(f"Feature importance saved to {importance_filename}")

# Plot top 20 feature importances
print("\nPlotting top 20 feature importances...", flush=True)
top_n = 20
top_features = sorted_importance[:top_n]
feature_names_top = [f[0] for f in reversed(top_features)]
importance_scores_top = [f[1] for f in reversed(top_features)]

fig, ax = plt.subplots(figsize=(12, 10))
ax.barh(feature_names_top, importance_scores_top)
ax.set_xlabel("Feature Importance (Gain)")
ax.set_ylabel("Feature")
ax.set_title(f"Top {top_n} LightGBM Feature Importances (No Variants, Non-Normalized)")
plt.tight_layout()

feature_importance_plot_filename = './results/results_raw/lightGBM_results_raw/lightgbm_raw_notune_feature_importance_plot.png'
plt.savefig(feature_importance_plot_filename)
plt.close(fig)
print(f"Feature importance plot saved to {feature_importance_plot_filename}", flush=True)