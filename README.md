# DS_Final_Project
Data Science in the Life Sciences - Final Project

## How to run the code
In order to run notebooks you need to create python environment with the packages mentioned in requirements.txt.
After that to run the code ```sbatch ./utils/sbatch_jupyter.slurm``` to create jupyter notebook. 

You also can change the parameters in ./utils/sbatch_jupyter.slurm (cpu and memory for large datasets).

In addition to this, python scripts could be run via the same slurm commands (ex. ```sbatch download_matrices.slurm```)

## The structure of the folder:
```
'tree ./'
```

```
./
|-- cell_line_processing
|   |-- 01_Lincs_PharmacoDB
|   |   |-- notebooks
|   |   |   |-- 00a_getting_data_lincs_2020.ipynb
|   |   |   |-- 00b_get_inferred_genes.ipynb
|   |   |   |-- 01_filter_lincs_2020.ipynb
|   |   |   |-- 03a_processing_compounds_pharmacodb.ipynb
|   |   |   |-- 03b_processing_compounds_pharmdb.ipynb
|   |   |   |-- 04_processing_celllines_pharmacodb.ipynb
|   |   |   |-- 05_processing_response_pharmacodb.ipynb
|   |   |   |-- 06_look_at_overlapping.ipynb
|   |   |   |-- 07_fingerprints.ipynb
|   |   |   |-- 08_drug_response_curve_filtering.ipynb
|   |   |   |-- 09_drug_response_curve_compute_response_r2_top0.7.ipynb
|   |   |   |-- 10_combine_all_the_data_inferred.ipynb
|   |   |   |-- comp2pubchem_dictionary.pkl
|   |   |   `-- getting_data_pubchem.ipynb
|   |   `-- scripts
|   |       |-- getting_data_pharmacodb
|   |       |   |-- cell_lines_by_id.py
|   |       |   |-- cell_lines.py
|   |       |   |-- compounds_by_id.py
|   |       |   |-- compounds.py
|   |       |   `-- experiments_by_cell_line.py
|   |       `-- getting_expression_data_lincs
|   |           |-- 00c_download_matrices.py
|   |           `-- download_matrices.slurm
|   |-- 02_Tahoe
|   |   |-- notebooks
|   |   |   |-- 00_GCSFileSystem.ipynb
|   |   |   |-- 01_tahoe_save_metadata.ipynb
|   |   |   |-- 02_tahoe_lincs_common_genes.ipynb
|   |   |   |-- 03_tahoe_pharmacodb_common_entities.ipynb
|   |   |   |-- 04_compute_pseudobulks_plate1.ipynb
|   |   |   |-- 05_process_pseudobulks_plate1.ipynb
|   |   |   |-- 06_process_output_expressions.ipynb
|   |   |   |-- 07_tahoe_fingerprints.ipynb
|   |   |   |-- 08_tahoe_pharmacodb_overlapping.ipynb
|   |   |   |-- 09_drug_response_curve_filtering.ipynb
|   |   |   |-- 10_drug_response_curve_compute_response_r2_top0.7.ipynb
|   |   |   |-- 11_combine_all_the_data_inferred.ipynb
|   |   |   |-- 12_clustering_tahoe.ipynb
|   |   |   |-- 12_clustering_tahoe_overlapping.ipynb
|   |   |   `-- get_to_know_lincs_lvl3.ipynb
|   |   `-- scripts
|   |       |-- 03a_plates_filter.py
|   |       |-- load_tahoe.sh
|   |       `-- srun_python.slurm
|   |-- 03_formatting_normalizing
|   |   |-- lincs_tahoe.ipynb
|   |   |-- lincs_tahoe_norm_add_cellline.ipynb
|   |   `-- lincs_tahoe_norm.ipynb
|   `-- utils
|       `-- sbatch_jupyter.slurm
`-- README.md
```
