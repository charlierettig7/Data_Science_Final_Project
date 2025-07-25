#!/bin/bash
#SBATCH -J np_mut
#SBATCH -o np_mut.%j.out
#SBATCH -e np_mut.%j.err
#SBATCH --time=2-00:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=65G
#SBATCH --partition=big
#SBATCH --account=dsls

# only necessary on the first run
#eval "$(/home/vonkl01/miniconda3/bin/conda shell.bash hook)"
#conda create -n annotate_muts r-base r-essentials r-devtools r-remotes r-data.table

conda init bash
conda activate annotate_muts
conda install numpy
#conda install python pandas pyarrow

python3 create_mutation_presence_cols_np.py