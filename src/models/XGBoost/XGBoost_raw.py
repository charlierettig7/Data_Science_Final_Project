import xgboost as xgb
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import json

def load_and_preprocess(file_path_no_vars, file_path_vars, vars = True):
    """Loads and preprocesses the data."""
    print("--- Loading and Preprocessing Data ---")
    data_load_no_vars = pd.read_parquet(file_path_no_vars)
    data_load_vars = pd.read_parquet(file_path_vars)

    gene_cols = data_load_vars.columns.difference(data_load_no_vars.columns)

    if vars:
        data = data_load_vars
        for col in gene_cols:
            data[col].fillna(0, inplace=True)
    else:
        data = data_load_no_vars
    
    label = 'responses'
    good = data[label].notnull() & np.isfinite(data[label])
    data = data.loc[good].reset_index(drop=True)
    x = data.drop(label, axis=1)
    x = x.drop('cellosaurus_ids.accession', axis=1)
    y = data[label]

    categorical_cols = ['sig_id', 'pert_id', 'cellosaurus_id', 'inchi_key', 'cell_drug', 'cmap_name', 'smiles']
    for col in categorical_cols:
        x[col] = x[col].astype('category')

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1)
    x_val, x_test, y_val, y_test = train_test_split(x_test, y_test, test_size=0.5, random_state=1)

    print(f"Train set size: {len(x_train)}")
    print(f"Validation set size: {len(x_val)}")
    print(f"Test set size: {len(x_test)}")

    dtrain = xgb.DMatrix(x_train, label=y_train, enable_categorical=True)
    dval = xgb.DMatrix(x_val, label=y_val, enable_categorical=True)
    dtest = xgb.DMatrix(x_test, label=y_test, enable_categorical=True)

    return dtrain, dval, dtest, y_test, x_train.columns

def train_model(dtrain, dval, params_path, num_rounds=500, early_stopping_rounds=10):
    """Trains the XGBoost model."""
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
    bst.save_model('xgboost_model_raw.json')
    return bst, evals_result

def analyze_performance(bst, dtest, y_test, evals_result, feature_names):
    """Analyzes model performance and saves artifacts."""
    print("\n--- Performance Analysis ---")

    # 1. Plot training and validation loss
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
    plt.title('XGBoost RMSE vs. Training Iteration')
    plt.grid(True)
    plot_filename = 'xgboost_loss_plot.png'
    plt.savefig(plot_filename)
    print(f"Plot saved to {plot_filename}")

    # 2. Save evaluation metrics
    print("\nSaving evaluation metrics...")
    preds = bst.predict(dtest, iteration_range=(0, bst.best_iteration + 1))
    test_rmse = np.sqrt(mean_squared_error(y_test, preds))
    test_r2 = r2_score(y_test, preds)
    print(f"Final RMSE on test set: {test_rmse:.4f}")
    print(f"Final R^2 on test set: {test_r2:.4f}")

    metrics = {
        'best_iteration': bst.best_iteration,
        'best_validation_rmse': bst.best_score,
        'final_test_rmse': test_rmse,
        'final_test_r2': test_r2,
        'training_rmse_history': evals_result['train']['rmse'],
        'validation_rmse_history': evals_result['eval']['rmse']
    }

    metrics_filename = 'xgboost_evaluation_metrics.json'
    with open(metrics_filename, 'w') as f:
        json.dump(metrics, f, indent=4)
    print(f"Metrics saved to {metrics_filename}")

    # 3. Save feature importance
    print("\nSaving feature importance...")
    importance = bst.get_score(importance_type='gain')
    sorted_importance = sorted(importance.items(), key=lambda item: item[1], reverse=True)

    importance_filename = 'xgboost_feature_importance.txt'
    with open(importance_filename, 'w') as f:
        f.write("Feature Importance (Gain)\n")
        f.write("===========================\n")
        for feature, score in sorted_importance:
            f.write(f"{feature}: {score}\n")
    print(f"Feature importance saved to {importance_filename}")

def main():
    """Main function to run the training pipeline."""
    file_path = './data/signature_response_features_r2_top0.7_final.parquet'
    file_path_vars = './data/lincs_top0.7_gene_level_matrix.parquet'
    params_path = './src/models/XGBoost/xgboost_config.json'

    dtrain, dval, dtest, y_test, feature_names = load_and_preprocess(file_path, file_path_vars)
    
    bst, evals_result = train_model(dtrain, dval, params_path)
    
    analyze_performance(bst, dtest, y_test, evals_result, feature_names)

if __name__ == "__main__":
    main()
