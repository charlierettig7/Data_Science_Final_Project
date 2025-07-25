# Modeling
**Overview**
All data used in training and validation, as well as results from the training and prediction processes have already been obtained and saved in this repository. Thus, any of the scripts can be run with the current contents of this repository without depending on running other scripts first. This is unfortunately not the case for scripts in patient processing and variant processing since some files are too large.

*NOTE: All scripts were run on FU's HPC cluster (curta.zedat.fu-berlin.de), and bash scripts contained in this repository are configured for this environment. 

**File structure and contents:** 
- Main directory: ds_final_project
  - bash: Bash scripts for running all Python scripts. The names of the bash scripts match the names of their corresponding Python scripts (e.g., tahoe_predict.sh runs tahoe_predict.py). All bash scripts assume access to curta.zedat.fu-berlin.de. 
  - data: preprocessed data used for training and external validation
    - lincs_formatted_normalized_cellosaurus.parquet: preprocessed LINCS dataset _with_ variant-calling data for training
    - signature_response_features_r2_top0.7_final.parquet: preprocessed LINCS dataset _without_ variant-calling data for training
    - tahoe_formatted_normalized_separate_scaler_for_genes_same_scaler_for_dosage_cellosaurus.parquet: preprocessed Tahoe dataset for external validation 
  - results: All results from training and external validation. Each sub-directory in this directory (listed below) corresponds to a specific model. Each sub-directory contains the output files for trained models themselves (to be loaded in later in prediction files) as well as a _plots_ and _eval_ directories, containing visualizations of the results of the training process and evaluation metrics/feature importances, respectively.
    - results_raw: results from raw models (trained without variants data)
    - results_vars: results from models trained with variants data
    - tahoe_predictions: results from prediction on external Tahoe dataset (all models)
- patient_predictions: contains CSV files with predictions on real patient data using LightGBM
  - src: All Python scripts for training, validating, and saving models, generating and saving evaluation metrics and visualizations, and making predictions.
    - models: contains three sub-directories, each corresponding to a single model framework (LightGBM, linear regression, and XGBoost). Python script names denote whether the model was trained with/without variants data (vars/raw), whether the training data was normalized or not (scaled/""), and whether or not the model was tuned with a hyperparameter grid search. For example, lightGBM/lightGM_vars_scaled_notune.py is the script for the LightGBM model trained on normalized columns with variants data and manual tuning (no grid search, just hand-tuning).
      - lightGBM: all scripts for training different versions of LightGBM
      - linear: all scripts for training
      - XGBoost: all scripts for training different versions of XGBoost
    - predictions: contains the script for making predictions on the Tahoe dataset as well as saving results and visualizations.

*NOTE: For LightGBM and the linear models, the parameters are set in their respective Python scripts for training. XGBoost, however, is configured with JSON files in src/XGBoost. 

**Instructions for Running Code**

The src/predictions/tahoe_predict.py script can be run directly to generate predictions on the Tahoe dataset, as all files and results that this script depends on are already in the repository. However, if you wish to go through the entire workflow, you may follow the steps below. These steps take you through an example workflow for training lightGBM on data containing normalized numerical columns _with_ variants data and manual tuning (no grid search). 

The models that are used to make predictions on the Tahoe dataset are:
- LightGBM with variants and normalized numeric columns: src/models/lightGBM/lightGBM_vars_scaled_notune.py
- LightGBM without variants and normalized numeric columns: src/models/lightGBM/lightGBM_raw_scaled_notune.py
- XGBoost with variants and normalized numeric columns: src/models/XGBoost/XGBoost_vars_scaled_notune.py
- XGBoost without variants and normalized numeric columns: src/models/XGBoost/XGBoost_raw_scaled_notune.py
- Ridge regression with variants and normalized numeric columns: src/models/linear/linear_vars_tuned.py
- Ridge regression without variants and normalized numeric columns: src/models/linear/linear_raw_tuned.py

_Starting with preprocessed data in the _data_ directory and with _ds_final_project_ as your working directory:_
1. Model training: In your terminal (CLI), run the bash script corresponding to the model you wish to train (bash names match Python script names): sbatch bash/lightGBM_vars_scaled_notune.sh
2. View the results (saved model files, evaluation metrics, feature importances, and plots) in results/results_vars/lightGBM_results_vars
3. Prediction on external (Tahoe) dataset: In your terminal (CLI), run tahoe_predict.sh: sbatch bash/tahoe_predict.sh
4. View results and CSVs containing predictions in results/tahoe_predictions

**Congratulations! You have successfully trained a LightGBM model and made predictions on an external dataset.**
