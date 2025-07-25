#!/bin/bash
#SBATCH -J cell_ids 
#SBATCH -o find_cell_id.%j.out
#SBATCH -e find_cell_id.%j.err
#SBATCH --time=2-02:00:00
#SBATCH --nodes=1
#SBATCH --mem=20G
#SBATCH --partition=small
#SBATCH --account=dsls

# only necessary on the first run
#eval "$(/home/vonkl01/miniconda3/bin/conda shell.bash hook)"
#conda create -n annotate_muts r-base r-essentials r-devtools r-remotes r-tidyverse

conda init bash
conda activate annotate_muts
#conda install -c conda-forge r-tidyverse
#conda install -c conda-forge r-curl
conda install r-data.table

Rscript cell_annotate.R