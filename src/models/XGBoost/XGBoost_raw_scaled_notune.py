import xgboost as xgb
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GroupShuffleSplit
import matplotlib.pyplot as plt
import json

def load_and_preprocess(file_path_no_vars, file_path_vars):
    """
    Load and preprocess the data for modeling.

    Parameters:
        file_path_no_vars (str): Path to parquet file containing features without variants.
        file_path_vars (str): Path to parquet file containing normalized features with variants.

    Returns:
        dtrain (xgb.DMatrix): Training data matrix.
        dval (xgb.DMatrix): Validation data matrix.
        dtest (xgb.DMatrix): Test data matrix.
        y_test (pandas.Series): True response values for the test set.
        feature_names (Index): Names of the feature columns.
    """
    print("--- Loading and Preprocessing Data ---")
    data_load_no_vars = pd.read_parquet(file_path_no_vars)
    data_load_vars = pd.read_parquet(file_path_vars)

    # Drop variant-calling columns for raw model (data_load_vars contains normalized data)
    col_diff = data_load_vars.columns.difference(data_load_no_vars.columns)
    data = data_load_vars.drop(columns = col_diff)

    # Extract label, and groups for train/val/test split stratification (to prevent data leakage)
    label = 'responses'
    good = data[label].notnull() & np.isfinite(data[label])
    data = data.loc[good].reset_index(drop = True)
    y = data[label]
    groups = data['cellosaurus_id']
    x = data.drop(columns = [label, 'sig_id', 'cellosaurus_id'])

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

    print(f"Train set size: {len(x_train)}")
    print(f"Validation set size: {len(x_val)}")
    print(f"Test set size: {len(x_test)}")

    dtrain = xgb.DMatrix(x_train, label=y_train, enable_categorical=True)
    dval = xgb.DMatrix(x_val, label=y_val, enable_categorical=True)
    dtest = xgb.DMatrix(x_test, label=y_test, enable_categorical=True)

    return dtrain, dval, dtest, y_test, x_train.columns

def train_model(dtrain, dval, params_path, num_rounds=500, early_stopping_rounds=30):
    """
    Train an XGBoost model using specified parameters and early stopping.

    Parameters:
        dtrain (xgb.DMatrix): Training data.
        dval (xgb.DMatrix): Validation data.
        params_path (str): Path to JSON file with XGBoost configuration parameters.
        num_rounds (int): Maximum number of boosting rounds.
        early_stopping_rounds (int): Number of rounds without improvement before stopping.

    Returns:
        bst (xgb.Booster): Trained XGBoost model.
        evals_result (dict): History of evaluation metrics during training.
    """
    print("\n--- Starting XGBoost Training ---")
    with open(params_path, 'r') as f:
        params = json.load(f)

    evallist = [(dtrain, 'train'), (dval, 'eval')]
    evals_result = {}
    
    bst = xgb.train(
        params,
        dtrain,
        num_boost_round=num_rounds,
        evals=evallist,
        evals_result=evals_result,
        early_stopping_rounds=early_stopping_rounds,
        verbose_eval=10
    )
    print("--- Finished XGBoost Training ---")
    bst.save_model('results/results_raw/XGBoost_results_raw/xgboost_model_raw_scaled_notune.json')
    return bst, evals_result

def analyze_performance(bst, dtest, y_test, evals_result, feature_names):
    """
    Evaluate model performance, generate plots, and save metrics and feature importances.

    Parameters:
        bst (xgb.Booster): Trained XGBoost model.
        dtest (xgb.DMatrix): Test data.
        y_test (pandas.Series): True response values for the test set.
        evals_result (dict): Dictionary containing train and validation RMSE history.
        feature_names (Index): Names of the feature columns.

    Returns:
        None
    """
    print("\n--- Performance Analysis ---")

    # Plot training and validation loss
    print("Plotting training and validation loss...")
    epochs = len(evals_result['train']['rmse'])
    x_axis = range(0, epochs)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(x_axis, evals_result['train']['rmse'], label='Train RMSE')
    ax.plot(x_axis, evals_result['eval']['rmse'], label='Validation RMSE')
    ax.axvline(x=bst.best_iteration, color='r', linestyle='--', label=f'Best Iteration ({bst.best_iteration})')
    ax.legend()
    plt.ylabel('RMSE')
    plt.xlabel('Training Iteration')
    plt.title('XGBoost RMSE vs. Training Iteration (Without Variants, Normalized)')
    plt.grid(True)
    plot_filename = 'results/results_raw/XGBoost_results_raw/xgboost_loss_plot_raw_scaled_notune.png'
    plt.savefig(plot_filename)
    print(f"Plot saved to {plot_filename}")

    # Make predictions on test set and calculate metrics
    print("\nEvaluating on test set...", flush=True)
    y_pred_test = bst.predict(dtest, iteration_range=(0, bst.best_iteration + 1))
    test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
    test_r2 = r2_score(y_test, y_pred_test)
    print(f"Test RMSE: {test_rmse:.4f}", flush=True)
    print(f"Test R^2: {test_r2:.4f}", flush=True)

    metrics = {
        'best_iteration': bst.best_iteration,
        'best_validation_rmse': bst.best_score,
        'test_rmse': test_rmse,
        'test_r2': test_r2,
        'training_rmse_history': evals_result['train']['rmse'],
        'validation_rmse_history': evals_result['eval']['rmse']
    }

    metrics_filename = 'results/results_raw/XGBoost_results_raw/xgboost_evaluation_metrics_raw_scaled_notune.json'
    with open(metrics_filename, 'w') as f:
        json.dump(metrics, f, indent=4)
    print(f"Metrics saved to {metrics_filename}")

    # Save feature importance
    print("\nSaving feature importance...")
    importance = bst.get_score(importance_type='gain')
    sorted_importance = sorted(importance.items(), key=lambda item: item[1], reverse=True)

    importance_filename = 'results/results_raw/XGBoost_results_raw/xgboost_feature_importance_raw_scaled_notune.txt'
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
    # For horizontal bar plot, reverse the order for plotting
    feature_names_top = [f[0] for f in reversed(top_features)]
    importance_scores_top = [f[1] for f in reversed(top_features)]

    fig, ax = plt.subplots(figsize=(12, 10))
    ax.barh(feature_names_top, importance_scores_top)
    ax.set_xlabel("Feature Importance (Gain)")
    ax.set_ylabel("Feature")
    ax.set_title(f"XGBoost Top {top_n} Feature Importances (Without Variants, Normalized)")
    plt.tight_layout()
    feature_importance_plot_filename = 'results/results_raw/XGBoost_results_raw/xgboost_raw_scaled_notune_feature_importance_plot.png'
    plt.savefig(feature_importance_plot_filename)
    plt.close(fig)  # Close the figure to free memory
    print(f"Feature importance plot saved to {feature_importance_plot_filename}", flush=True)

def main():
    """
    Main function to run the XGBoost training pipeline end-to-end.

    Steps:
        1. Load and preprocess data.
        2. Train the model.
        3. Analyze performance and save artifacts.

    Returns:
        None
    """
    file_path_no_vars = './data/signature_response_features_r2_top0.7_final.parquet'
    file_path_vars = './data/lincs_formatted_normalized_cellosaurus.parquet'
    params_path = './src/models/XGBoost/xgboost_config_raw.json'

    dtrain, dval, dtest, y_test, feature_names = load_and_preprocess(file_path_no_vars, file_path_vars)
    print(f'number of training cols (raw): {dtrain.num_col()}')
    
    bst, evals_result = train_model(dtrain, dval, params_path)
    
    analyze_performance(bst, dtest, y_test, evals_result, feature_names)

if __name__ == "__main__":
    main()
