# Data_Science_Final_Project
Data Science in the Life Sciences - Final Project

Welcome to the final project Group 3 in Data Science in the Life Sciences! This repository contains the entire code base for our project, which we have _unofficially_ named scTherapy+. 

**Inspiration**
Tumor heterogeneity represents a major challenge in the development of both general and targeted treatment of malignant cancers. Features such as the tumoral microenvironment and intertumoral differential gene expression (including evolutionary differential expression) result in innumerable effective variations in tumoral profiles, blurring the identification of viable treatment targets, especially those that aim to be patient-specific. While the original manuscript by Ianevski et al. does lay out methods for explicitly dealing with tumoral heterogeneity, we believe that they left out potentially informative dimensions of integratable data (e.g., pathway analysis and variant calling) that could not only improve confidence in tumor classification and ultimate model predictions, but also provide a deeper level of biological interpretability – something that the original manuscript lacks. To the latter point, we believe that for this type of prediction task, biological interpretation must be clear to translate any conclusions into follow-up development of clinical treatments. 

Our method seeks to improve on the scTherapy (hence the name scTherapy+), a method developed by Ianevski et al. (2024), which uses scRNA-seq data from patient tumor samples to predict optimal patient-specific drug/dose combinations for cancer treatment. In this project, we attempt to both replicate their original method and improve its performance by integrating variant-calling and comparing different modeling frameworks. We also take the method a step further by annotating the results with pathway enrichment analysis in hopes of providing biologically sound interpretability of our models. 

Concretely, our method leverages scRNA-seq data, from which we obtain differential gene expression data and identification of transcriptome-wide SNPs, along with the chemical structures of cancer drugs, to predict post-treatment cell viability (as a percentage). 

The original manuscript from Ianevski et al. can be found at this link: https://www.nature.com/articles/s41467-024-52980-5#Sec8 

The read me is split up by how we divided tasks. We apologize for any inconvenience this causes.
## Modelling: 

**Overview**

All data used in training and validation, as well as results from the training and prediction processes have already been obtained and saved in this repository. Thus, any of the scripts can be run with the current contents of this repository without depending on running other scripts first.

*NOTE: All scripts were run on FU's HPC cluster (curta.zedat.fu-berlin.de), and bash scripts contained in this repository are configured for this environment. 

**File structure and contents:** 
- Main directory:
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

## Patient Data and Variants
## Folder Structure

├── bash_scripts
├── GSEA
│   ├── GSEA.R
│   ├── lgbm_raw.zip
│   ├── lgbm_vars.zip
│   ├── xgboost_raw.zip
│   ├── xgboost_vars.zip
├── Process_patient_data
│   ├── data
│   │   ├── 17_HMM_predHMMi6.rand_trees.hmm_mode-subclusters.patient5.cell_groupings
│   │   ├── 17_HMM_predHMMi6.rand_trees.hmm_mode-subclusters.patient6.cell_groupings
│   │   ├── aggregated_muts_12.csv
│   │   ├── aggregated_muts_5.csv
│   │   ├── aggregated_muts_6.csv
│   │   ├── counts_patient5.csv
│   │   ├── counts_patient6.csv         # (too large - derived through process_patient_expression.R)
│   │   ├── gene_expression_processed_pat5.csv
│   │   ├── gene_expression_processed_pat6.csv
│   │   ├── mutation_pat12.csv
│   │   ├── mutation_pat5.csv
│   │   ├── mutation_pat6.csv
│   │   ├── patient12.RDS
│   │   ├── patient5.RDS
│   │   ├── patient6.RDS
│   │   ├── pat5_to_predict.parquet
│   │   ├── pat6_to_predict.parquet
│   │   ├── row_to_subclone_pat5.parquet
│   │   ├── row_to_subclone_pat6.parquet
│   │   ├── SRR30720406_.vcf
│   │   ├── SRR30720407_.vcf            # (too large - scAllele)
│   │   ├── SRR30720408_.vcf
│   │   └── .DS_Store
│   ├── filter_vcf.R
│   ├── pat_12_pseudobulk.ipynb
│   ├── pat_5_pseudobulk.ipynb
│   ├── pat_6_pseudobulk.ipynb
│   ├── process_patient_expression.R
│   ├── plots
│   │   ├── Rplots_6.pdf
│   │   ├── Rplots_12.pdf
│   │   ├── patient12_UMAP.png
│   │   ├── patient12_elbowplot.png
│   │   ├── patient5_UMAP.png
│   │   ├── patient5_elbowplot.png
│   │   ├── patient6_UMAP.png
│   │   ├── patient6_elbowplot.png
│   │   └── patient_5_Rplots.pdf
│   └── scTherapy_plus.R
├── Preprocess_Variants
│   ├── cell_annotate.R
│   ├── create_mutation_presence_cols_np.py
│   ├── data
│   │   ├── Model.csv
│   │   ├── OmicsSomaticMutations.csv     # (too large - source depmap)
│   │   └── mutations_cellosaurus_full.csv # (too large - derived using bash_scripts/annotate_cell_ids.sh)
│   ├── merge_variants_into_lincs.ipynb
│   ├── mutation_data.R
│   ├── query_tahoe_nonoverlapping.ipynb
│   └── query_tahoe_overlapping.ipynb


---

## Usage

### Bash Scripts

Bash Scripts were used to run various tasks on the FU servers. These scripts are saved in this folder

### GSEA

- Scripts related to Gene Set Enrichment Analysis are located in the `GSEA/` folder.
- `GSEA.R` performs enrichment analysis and visualization using preprocessed data.
- The zip files contain both plots and data on the 10 best and 10 worst predictions for each of the respective model. The R file can be run as is, if the paths are correct.

### Patient Data Processing

- The `Process_patient_data/` directory contains notebooks (`*.ipynb`) and scripts to preprocess and analyze patient-specific data.
- `filter_vcf.R` filters the outputs of scAllele
- `process_patient_expression.R` transforms the expression data in a way to allow for pseudobulking using the respective .ipynb notebooks
- Data files and intermediate files are in `Process_patient_data/data`.
    - `.vcf` contain scAllele output
    - `patX_to_predict.parquet` contain the processed patient data, including expression and gene-level variants to predict on
    - `.cell_groupings` are scTherapy output that indicate which subclone each cell_id belongs to
- Plot outputs can be found under `Process_patient_data/plots`. These are created through scTherapy
- Note: we weren't able to fully process patient 12 since there was an ongoing issue with I/O Locks on the necessary files on the server. While we did solve this problem in time, there did not remain enough time to complete the subsequent analyses.

### Variant Preprocessing

- The `Preprocess_Variants/` folder contains scripts and notebooks for variant annotation and mutation data integration.
- `cell_annotate.R` matches IDs used in DepMap to Cellosaurus IDs used in lincs and Tahoe
- `data/` contains mutation files for our cell lines of interest
- `merge_variants_into_lincs.ipynb` appends gene-level mutation data to lincs
- `query_tahoe_...` does the same for tahoe
- `mutation_data.R` contains some EDA and how we came up with cell_annotate.R
- We used T2T from UCSC as a reference, in addition to the RefSeq Annotation file - we did not upload these files since they are too large for github.