import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GroupShuffleSplit, GroupKFold, GridSearchCV, cross_val_score
from sklearn.linear_model import Lasso, Ridge, LinearRegression
import matplotlib.pyplot as plt
import json
import joblib
from joblib import dump

print("--- Importing Data ---", flush = True)
file_path = './data/signature_response_features_r2_top0.7_final.parquet'
file_path_vars = './data/lincs_formatted_normalized_cellosaurus.parquet'

data_load_vars = pd.read_parquet(file_path_vars)
data_load_no_vars = pd.read_parquet(file_path)

label = 'responses'
gene_cols = data_load_vars.columns.difference(data_load_no_vars.columns)
for col in gene_cols:
    data_load_vars[col].fillna(0, inplace=True)

label = 'responses'
good = data_load_vars[label].notnull() & np.isfinite(data_load_vars[label])
data = data_load_vars.loc[good].reset_index(drop=True)
x = data.drop(columns = [label, 'sig_id', 'cellosaurus_id'])
y = data[label]

# Single group-based split: train vs. test (hold-out)
groups = data['cellosaurus_id']
gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=1)
train_idx, test_idx = next(gss.split(x, y, groups))
x_train = x.iloc[train_idx]
y_train = y.iloc[train_idx]
groups_train = groups.iloc[train_idx]
x_test  = x.iloc[test_idx]
y_test  = y.iloc[test_idx]

print("--- Starting group-aware cross-validation training ---")

# Hyperparameter grid and CV splitter
alphas = [0.01, 0.1, 1, 10, 100]
cv = GroupKFold(n_splits=5)

# Ridge tuning
ridge_gs = GridSearchCV(
    Ridge(), param_grid={'alpha': alphas},
    cv=cv, scoring='neg_mean_squared_error',
    return_train_score=True
)
ridge_gs.fit(x_train, y_train, groups=groups_train)
best_ridge = ridge_gs.best_estimator_
cv_rmse_ridge = np.sqrt(-ridge_gs.best_score_)
ridge_train_rmse = np.sqrt(-ridge_gs.cv_results_['mean_train_score'])
ridge_val_rmse   = np.sqrt(-ridge_gs.cv_results_['mean_test_score'])

# Lasso tuning
lasso_gs = GridSearchCV(
    Lasso(max_iter=15000),
    param_grid={'alpha': alphas},
    cv=cv, scoring='neg_mean_squared_error',
    return_train_score=True
)
lasso_gs.fit(x_train, y_train, groups=groups_train)
best_lasso = lasso_gs.best_estimator_
cv_rmse_lasso = np.sqrt(-lasso_gs.best_score_)
lasso_train_rmse = np.sqrt(-lasso_gs.cv_results_['mean_train_score'])
lasso_val_rmse   = np.sqrt(-lasso_gs.cv_results_['mean_test_score'])

# Plot CV history for Ridge
fig, ax = plt.subplots(figsize=(8,5))
ax.plot(alphas, ridge_train_rmse, marker='o', label='Train RMSE')
ax.plot(alphas, ridge_val_rmse,   marker='o', label='CV RMSE')
ax.set_xscale('log'); ax.set_xlabel('Alpha'); ax.set_ylabel('RMSE')
ax.set_title('Ridge Vars: RMSE vs. Alpha'); ax.legend()
plt.savefig('results/results_vars/linear_results_vars/ridge_vars_notune_scaled_cv_history.png')
plt.close()

# Plot CV history for Lasso
fig, ax = plt.subplots(figsize=(8,5))
ax.plot(alphas, lasso_train_rmse, marker='o', label='Train RMSE')
ax.plot(alphas, lasso_val_rmse,   marker='o', label='CV RMSE')
ax.set_xscale('log'); ax.set_xlabel('Alpha'); ax.set_ylabel('RMSE')
ax.set_title('Lasso Vars: RMSE vs. Alpha'); ax.legend()
plt.savefig('results/results_vars/linear_results_vars/lasso_vars_notune_scaled_cv_history.png')
plt.close()

# Standard Linear Regression CV and fit
lr = LinearRegression()
neg_mse_scores = cross_val_score(
    lr, x_train, y_train,
    cv=cv, groups=groups_train,
    scoring='neg_mean_squared_error'
)
cv_rmse_lr = np.sqrt(-np.mean(neg_mse_scores))
lr.fit(x_train, y_train)

# Compare models and select best
cv_rmses = {'ridge': cv_rmse_ridge, 'lasso': cv_rmse_lasso, 'linear': cv_rmse_lr}
best_model_name = min(cv_rmses, key=cv_rmses.get)
best_models = {'ridge': best_ridge, 'lasso': best_lasso, 'linear': lr}
best_model = best_models[best_model_name]

print("--- Finished Training! ---")
dump(
    best_model,
    f'results/results_vars/linear_results_vars/{best_model_name}_vars_notune_scaled_model.joblib',
    compress=3
)

# Evaluate on train and test
train_rmse = np.sqrt(mean_squared_error(y_train, best_model.predict(x_train)))
test_rmse  = np.sqrt(mean_squared_error(y_test,  best_model.predict(x_test)))

# Save metrics
metrics = {'model': best_model_name, 'train_rmse': train_rmse, 'test_rmse': test_rmse}
with open(
    f'results/results_vars/linear_results_vars/{best_model_name}_vars_notune_scaled_evaluation_metrics.txt',
    'w'
) as f:
    json.dump(metrics, f, indent=4)

# Compare test performance across models
model_names = ['ridge','lasso','linear']
test_rmses = []; test_r2s = []
for name in model_names:
    m = best_models[name]
    preds = m.predict(x_test)
    test_rmses.append(np.sqrt(mean_squared_error(y_test, preds)))
    test_r2s.append(r2_score(y_test, preds))

# Plot training history for best model
if best_model_name in ['ridge','lasso']:
    hist_alphas = alphas
    train_hist = ridge_train_rmse if best_model_name=='ridge' else lasso_train_rmse
    cv_hist    = ridge_val_rmse   if best_model_name=='ridge' else lasso_val_rmse
    fig, ax = plt.subplots(figsize=(8,5))
    ax.plot(hist_alphas, train_hist, marker='o', label='Train RMSE')
    ax.plot(hist_alphas, cv_hist,    marker='o', label='CV RMSE')
    ax.set_xscale('log'); ax.set_xlabel('Alpha'); ax.set_ylabel('RMSE')
    ax.set_title(f'{best_model_name.capitalize()} Vars Training RMSE History')
    ax.legend()
    plt.savefig(
        f'results/results_vars/linear_results_vars/{best_model_name}_vars_notune_scaled_training_history.png'
    )
    plt.close()
else:
    fig, ax = plt.subplots(figsize=(8,5))
    ax.bar(['linear'], [train_rmse])
    ax.set_ylabel('Train RMSE'); ax.set_title('Linear Vars Training RMSE')
    plt.savefig('results/results_vars/linear_results_vars/linear_vars_notune_scaled_training_history.png')
    plt.close()

# Barplots for test RMSE and R2
fig, ax = plt.subplots(figsize=(8,5))
ax.bar(model_names, test_rmses); ax.set_ylabel('Test RMSE'); ax.set_title('Test RMSE by Model (Vars)')
plt.savefig('results/results_vars/linear_results_vars/scaled_model_comparison_rmse_vars.png'); plt.close()

fig, ax = plt.subplots(figsize=(8,5))
ax.bar(model_names, test_r2s); ax.set_ylabel('Test R2'); ax.set_title('Test R2 by Model (Vars)')
plt.savefig('results/results_vars/linear_results_vars/scaled_model_comparison_r2_vars.png'); plt.close()

importance = dict(zip(x.columns, best_model.coef_))
sorted_importance = sorted(importance.items(), key=lambda item: abs(item[1]), reverse=True)
with open(f'results/results_vars/linear_results_vars/{best_model_name}_vars_notune_scaled_feature_importance.txt', 'w') as f:
    f.write("Feature Importance (Coefficient Magnitude)\n")
    f.write("===========================\n")
    for feature, score in sorted_importance:
        f.write(f"{feature}: {score}\n")
