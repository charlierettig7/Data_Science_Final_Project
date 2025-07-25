import xgboost as xgb
from xgboost.callback import EarlyStopping
from sklearn.model_selection import GridSearchCV, GroupShuffleSplit
import pandas as pd
import numpy as np
from sklearn.metrics import r2_score, root_mean_squared_error
import matplotlib.pyplot as plt
import json
import time
import resource

def load_and_preprocess(file_path_no_vars):
    """
    Load and preprocess the data for modeling.

    Parameters:
        file_path_no_vars (str): Path to parquet file containing signature response features.

    Returns:
        x_train (pandas.DataFrame): Training feature matrix.
        x_test (pandas.DataFrame): Test feature matrix.
        y_train (pandas.Series): Training response values.
        y_test (pandas.Series): Test response values.
        feature_names (pandas.Index): Names of the feature columns.
        groups_train (pandas.Series): Group labels for training set splits.
    """
    print("--- Loading and Preprocessing Data ---", flush=True)
    data = pd.read_parquet(file_path_no_vars)

    label = 'responses'
    good = data[label].notnull() & np.isfinite(data[label])
    data = data.loc[good].reset_index(drop=True)
    x = data.drop(label, axis=1)
    groups = x['cellosaurus_id']
    x.drop(columns=['sig_id', 'pert_id', 'cellosaurus_id', 'inchi_key', 'cell_drug', 'cmap_name', 'smiles'], inplace=True)
    y = data[label]

    # --- Outer hold-out split by cellosaurus_id ---
    gss_outer = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=1)
    train_idx, test_idx = next(gss_outer.split(x, y, groups))
    x_train, y_train = x.iloc[train_idx], y.iloc[train_idx]
    x_test,  y_test  = x.iloc[test_idx],  y.iloc[test_idx]
    groups_train = groups.iloc[train_idx]
    feature_names = x_train.columns

    print(f"Train set size: {len(x_train)}", flush=True)
    print(f"Test set size:  {len(x_test)}", flush=True)

    return x_train, x_test, y_train, y_test, feature_names, groups_train

def train_model(x_train, x_test, y_train, y_test, feature_names, groups_train, num_rounds=30, early_stopping_rounds=10):
    """
    Train the XGBoost model using GridSearchCV and early stopping.

    Parameters:
        x_train (pandas.DataFrame): Training feature matrix.
        x_test (pandas.DataFrame): Test feature matrix for early stopping.
        y_train (pandas.Series): Training response values.
        y_test (pandas.Series): Test response values.
        feature_names (pandas.Index): Names of feature columns.
        groups_train (pandas.Series): Group labels for cross-validation.
        num_rounds (int): Number of boosting rounds for final training.
        early_stopping_rounds (int): Rounds without improvement before stopping.

    Returns:
        final_model (xgb.XGBRegressor): Trained XGBoost model.
        metrics (dict): Dictionary containing CV results and final test performance metrics.
    """
    # --- Inner group-aware CV splitter for GridSearch ---
    cv = GroupShuffleSplit(n_splits=5, test_size=0.2, random_state=2)

    # Base parameters (moved from config)
    base_params = {
        "enable_categorical": True,
        "verbosity": 1,
        "use_rmm": False,
        "booster": "gbtree",
        "objective": "reg:squarederror",
        "eval_metric": "rmse",
        "validate_parameters": True,
        "learning_rate": 0.3,
        "min_split_loss": 0.02,
        "subsample": 0.8,
        "seed": 42,
        "nthread": 8
    }
    # Hyperparameter grid
    param_grid = {
        "max_depth": [1, 2, 3, 4],
        "min_child_weight": [1, 21, 41, 61, 81],
        "gamma": [0, 0.025, 0.05, 0.075, 0.1],
        "lambda": [0, 1, 2, 3]
    }

    # Initialize model and grid search
    model = xgb.XGBRegressor(**base_params, n_estimators=num_rounds)
    grid_search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        scoring="neg_root_mean_squared_error",
        cv=cv,
        n_jobs=-1,
        verbose=1,
        refit=False
    )
    # Benchmark grid search time and memory usage
    mem_before_gs = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    time_start_gs = time.time()
    # Fit with early stopping on test set
    grid_search.fit(
        x_train, y_train,
        groups=groups_train,
        eval_set=[(x_train, y_train), (x_test, y_test)],
        verbose=False
    )
    time_end_gs = time.time()
    mem_after_gs = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    grid_search_time = time_end_gs - time_start_gs
    grid_search_mem_usage = mem_after_gs - mem_before_gs
    print(f"Grid search time: {grid_search_time:.2f} seconds", flush=True)
    print(f"Grid search memory increased: {grid_search_mem_usage} KB (peak: {mem_after_gs} KB)", flush=True)
    print(f"Best parameters found: {grid_search.best_params_}")
    print(f"Best CV RMSE: {abs(grid_search.best_score_)}")

    # --- Retrain best model with fresh early-stop split ---
    best_params = grid_search.best_params_
    final_model = xgb.XGBRegressor(**base_params, **best_params, n_estimators=500)

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
        verbose=False
    )
    time_end_train = time.time()
    mem_after_train = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    training_time = time_end_train - time_start_train
    mem_usage = mem_after_train - mem_before_train
    print(f"Final model training time: {training_time:.2f} seconds", flush=True)
    print(f"Final model memory increased: {mem_usage} KB (peak: {mem_after_train} KB)", flush=True)

    # Save the final model
    final_model.save_model('results/results_raw/XGBoost_results_raw/xgboost_model_raw_tuned_final.json')

    # --- Final evaluation on hold-out test set ---
    y_pred_test = final_model.predict(x_test)
    rmse_test = root_mean_squared_error(y_test, y_pred_test, squared=False)
    r2_test = r2_score(y_test, y_pred_test)
    print(f"Final Test RMSE: {rmse_test:.4f}", flush=True)
    print(f"Final Test R2:   {r2_test:.4f}", flush=True)

    metrics = {
        'cv_results': grid_search.cv_results_,
        'final_test_rmse': rmse_test,
        'final_test_r2': r2_test,
        'grid_search_time_seconds': grid_search_time,
        'grid_search_memory_usage_kb': grid_search_mem_usage,
        'training_time_seconds': training_time,
        'training_memory_usage_kb': mem_usage,
        'peak_memory_kb': mem_after_train
    }
    return final_model, metrics

def analyze_performance(bst, feature_names):
    """
    Generate performance plots, save evaluation metrics, and feature importances.

    Parameters:
        bst (xgb.XGBRegressor): Trained XGBoost model with evaluation history.
        feature_names (pandas.Index): Names of feature columns.

    Returns:
        None
    """
    print("\n--- Performance Analysis ---", flush=True)
    evals_result = bst.evals_result()

    # Plot training and validation loss
    print("Plotting training and validation loss...", flush=True)
    epochs = len(evals_result['validation_0']['rmse'])
    x_axis = range(epochs)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(x_axis, evals_result['validation_0']['rmse'], label='Train RMSE')
    ax.plot(x_axis, evals_result['validation_1']['rmse'], label='Validation RMSE')
    ax.axvline(x=bst.best_iteration, color='r', linestyle='--', label=f'Best Iteration ({bst.best_iteration})')
    ax.legend()
    plt.ylabel('RMSE')
    plt.xlabel('Training Iteration')
    plt.title('XGBoost RMSE vs. Training Iteration (Without Variants, Non-Normalized)')
    plt.grid(True)
    plot_filename = 'results/results_raws/XGBoost_results_raw/xgboost_loss_plot_raw_tuned.png'
    plt.savefig(plot_filename)
    print(f"Plot saved to {plot_filename}", flush=True)

    # Save evaluation metrics used for early stopping
    metrics_es = {
        'best_iteration': bst.best_iteration,
        'training_rmse_history': evals_result['validation_0']['rmse'],
        'validation_rmse_history': evals_result['validation_1']['rmse']
    }
    metrics_filename_es = 'results/results_raw/XGBoost_results_raw/xgboost_evaluation_metrics_raw_tuned.json'
    with open(metrics_filename_es, 'w') as f:
        json.dump(metrics_es, f, indent=4)
    print(f"Metrics saved to {metrics_filename_es}", flush=True)

    # Save feature importance
    importance = bst.get_booster().get_score(importance_type='gain')
    sorted_importance = sorted(importance.items(), key=lambda item: item[1], reverse=True)
    importance_filename = 'results/results_raw/XGBoost_results_raw/xgboost_feature_importance_raw_tuned.txt'
    with open(importance_filename, 'w') as f:
        f.write("Feature Importance (Gain)\n===========================\n")
        for feature, score in sorted_importance:
            f.write(f"{feature}: {score}\n")
    print(f"Feature importance saved to {importance_filename}", flush=True)

    # Plot top 20 feature importances
    top20 = sorted_importance[:20]
    features, scores = zip(*top20)
    # Reverse for horizontal bar chart
    features = features[::-1]
    scores = scores[::-1]
    fig2, ax2 = plt.subplots(figsize=(10, 8))
    ax2.barh(features, scores)
    ax2.set_xlabel("Importance")
    ax2.set_title("XGBoost Top 20 Feature Importances (Without Variants, Non-Normalized)")
    fig2.tight_layout()
    barplot_filename = 'results/results_raw/XGBoost_results_raw/xgboost_top20_importance_raw_tuned.png'
    plt.savefig(barplot_filename)
    print(f"Top 20 feature importance plot saved to {barplot_filename}", flush=True)

def main():
    """
    Main entry point for XGBoost training and analysis pipeline.

    Steps:
        1. Load and preprocess data.
        2. Train and tune model.
        3. Analyze performance and save artifacts.

    Returns:
        None
    """
    print("\n creating file paths...")
    file_path = './data/signature_response_features_r2_top0.7_final.parquet'
    print("\n finished creating file paths!")

    print("\n loading and preprocessing...")
    x_train, x_test, y_train, y_test, feature_names, groups_train = load_and_preprocess(file_path)
    print("\n finshed loading and preprocessing!")
    print(f'Number of training columns (raw): {len(x_train.columns)}')
    
    bst, metrics = train_model(x_train, x_test, y_train, y_test, feature_names, groups_train)
    # print("\n finished tuning and training!")
    
    print(f"Metrics: {metrics}")
    print("\n analyzing performance...", flush=True)
    analyze_performance(bst, feature_names)
    print("\n finished analysis!", flush=True)

if __name__ == "__main__":
    main()
