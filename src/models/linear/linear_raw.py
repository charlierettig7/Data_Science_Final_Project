import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GroupShuffleSplit, GridSearchCV, GroupKFold
import matplotlib.pyplot as plt
from sklearn.linear_model import Lasso, Ridge
import json
import joblib
from joblib import dump

file_path = './data/lincs_formatted_normalized_cellosaurus.parquet'
file_path_no_vars = './data/signature_response_features_r2_top0.7_final.parquet'

data_load = pd.read_parquet(file_path)
data_load_no_vars = pd.read_parquet(file_path_no_vars)

col_diff = data_load.columns.difference(data_load_no_vars.columns)
data_load = data_load.drop(columns = col_diff)

label = 'responses'
print(f"Num rows before NA filtering: {len(data_load_no_vars)}", flush = True)
good = data_load[label].notnull() & np.isfinite(data_load[label])
print(f"Num rows after NA Filtering {len(good)}", flush = True)

data = data_load.loc[good].reset_index(drop=True)
y = data[label]
groups = data['cellosaurus_id']
x = data.drop(columns = [label, 'sig_id', 'cellosaurus_id'])

# Single group-based split: train vs. test (hold-out)
gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=1)
train_idx, test_idx = next(gss.split(x, y, groups))
x_train = x.iloc[train_idx]
y_train = y.iloc[train_idx]
groups_train = groups.iloc[train_idx]
x_test  = x.iloc[test_idx]
y_test  = y.iloc[test_idx]

print("--- Starting group-aware cross-validation training ---")


alphas = [0.01, 0.1, 1, 10, 100]
cv = GroupKFold(n_splits=5)

# Ridge hyperparameter tuning with group-wise CV
ridge_gs = GridSearchCV(Ridge(), param_grid={'alpha': alphas}, cv=cv, scoring='neg_mean_squared_error', return_train_score=True)
ridge_gs.fit(x_train, y_train, groups=groups_train)
best_ridge = ridge_gs.best_estimator_
cv_rmse_ridge = np.sqrt(-ridge_gs.best_score_)

# Lasso hyperparameter tuning with group-wise CV
lasso_gs = GridSearchCV(Lasso(max_iter=15000), param_grid={'alpha': alphas}, cv=cv, scoring='neg_mean_squared_error', return_train_score=True)
lasso_gs.fit(x_train, y_train, groups=groups_train)
best_lasso = lasso_gs.best_estimator_
cv_rmse_lasso = np.sqrt(-lasso_gs.best_score_)

# Plot CV performance vs. alpha
alphas = ridge_gs.param_grid['alpha']

# Compute train and validation RMSE for ridge
ridge_train_rmse = np.sqrt(-ridge_gs.cv_results_['mean_train_score'])
ridge_val_rmse   = np.sqrt(-ridge_gs.cv_results_['mean_test_score'])

# Compute train and validation RMSE for lasso
lasso_train_rmse = np.sqrt(-lasso_gs.cv_results_['mean_train_score'])
lasso_val_rmse   = np.sqrt(-lasso_gs.cv_results_['mean_test_score'])

# Ridge plot
fig, ax = plt.subplots(figsize=(8,5))
ax.plot(alphas, ridge_train_rmse, marker='o', label='Train RMSE')
ax.plot(alphas, ridge_val_rmse,   marker='o', label='CV RMSE')
ax.set_xscale('log')
ax.set_xlabel('Alpha')
ax.set_ylabel('RMSE')
ax.set_title('Ridge: RMSE vs. Alpha')
ax.legend()
plt.savefig(f'results/results_raw/linear_results_raw/ridge_scaled_cv_history.png')
plt.close()

# Lasso plot
fig, ax = plt.subplots(figsize=(8,5))
ax.plot(alphas, lasso_train_rmse, marker='o', label='Train RMSE')
ax.plot(alphas, lasso_val_rmse,   marker='o', label='CV RMSE')
ax.set_xscale('log')
ax.set_xlabel('Alpha')
ax.set_ylabel('RMSE')
ax.set_title('Lasso: RMSE vs. Alpha')
ax.legend()
plt.savefig(f'results/results_raw/linear_results_raw/lasso_scaled_cv_history.png')
plt.close()

# Standard Linear Regression (no hyperparameters)
lr = LinearRegression()
from sklearn.model_selection import cross_val_score
neg_mse_scores = cross_val_score(
    lr, x_train, y_train,
    cv=cv,
    groups=groups_train,
    scoring='neg_mean_squared_error'
)
cv_rmse_lr = np.sqrt(-np.mean(neg_mse_scores))

# Always fit linear model on full train for later evaluation
lr.fit(x_train, y_train)

# Select best model based on validation RMSE
cv_rmses = {'ridge': cv_rmse_ridge, 'lasso': cv_rmse_lasso, 'linear': cv_rmse_lr}
best_model_name = min(cv_rmses, key=cv_rmses.get)
best_models = {'ridge': best_ridge, 'lasso': best_lasso, 'linear': lr}
best_model = best_models[best_model_name]

print("--- Finished Training! ---")
dump(
    best_model,
    f'results/results_raw/linear_results_raw/{best_model_name}_raw_notune_scaled_model.joblib',
    compress=3
)


# RMSE on train and test sets
train_rmse = np.sqrt(mean_squared_error(y_train, best_model.predict(x_train)))
test_rmse  = np.sqrt(mean_squared_error(y_test,  best_model.predict(x_test)))

# Save metrics
metrics = {
    'model': best_model_name,
    'train_rmse': train_rmse,
    'test_rmse': test_rmse
}
with open(f'results/results_raw/linear_results_raw/{best_model_name}_raw_notune_scaled_evaluation_metrics.txt', 'w') as f:
    json.dump(metrics, f, indent=4)

# === Compare test performance across all models ===
model_names = ['ridge', 'lasso', 'linear']
test_rmses = []
test_r2s = []
for name in model_names:
    model = best_models[name]
    preds = model.predict(x_test)
    test_rmses.append(np.sqrt(mean_squared_error(y_test, preds)))
    test_r2s.append(r2_score(y_test, preds))

# Plot training RMSE history for the best model
if best_model_name in ['ridge', 'lasso']:
    alphas = ridge_gs.param_grid['alpha']
    if best_model_name == 'ridge':
        train_hist = ridge_train_rmse
        cv_hist    = ridge_val_rmse
    else:
        train_hist = lasso_train_rmse
        cv_hist    = lasso_val_rmse
    fig, ax = plt.subplots(figsize=(8,5))
    ax.plot(alphas, train_hist, marker='o', label='Train RMSE')
    ax.plot(alphas, cv_hist,    marker='o', label='CV RMSE')
    ax.set_xscale('log')
    ax.set_xlabel('Alpha')
    ax.set_ylabel('RMSE')
    ax.set_title(f'{best_model_name.capitalize()} Training RMSE History')
    ax.legend()
    plt.savefig(f'results/results_raw/linear_results_raw/{best_model_name}_scaled_training_history.png')
    plt.close()
else:
    # Single-point training RMSE for LinearRegression
    fig, ax = plt.subplots(figsize=(8,5))
    ax.bar(['linear'], [train_rmse])
    ax.set_ylabel('Train RMSE')
    ax.set_title('Linear Regression Training RMSE')
    plt.savefig(f'results/results_raw/linear_results_raw/linear_scaled_training_history.png')
    plt.close()

# Plot barplot for test RMSE comparison
fig, ax = plt.subplots(figsize=(8,5))
ax.bar(model_names, test_rmses)
ax.set_ylabel('Test RMSE')
ax.set_title('Test RMSE by Model')
plt.savefig(f'results/results_raw/linear_results_raw/scaled_model_comparison_rmse.png')
plt.close()

# Plot barplot for test R2 comparison
fig, ax = plt.subplots(figsize=(8,5))
ax.bar(model_names, test_r2s)
ax.set_ylabel('Test R2')
ax.set_title('Test R2 by Model')
plt.savefig(f'results/results_raw/linear_results_raw/scaled_model_comparison_r2.png')
plt.close()

importance = dict(zip(x.columns, best_model.coef_))
sorted_importance = sorted(importance.items(), key=lambda item: abs(item[1]), reverse=True)
with open(f'results/results_raw/linear_results_raw/{best_model_name}_raw_notune_scaled_feature_importance.txt', 'w') as f:
    f.write("Feature Importance (Coefficient Magnitude)\n")
    f.write("===========================\n")
    for feature, score in sorted_importance:
        f.write(f"{feature}: {score}\n")
