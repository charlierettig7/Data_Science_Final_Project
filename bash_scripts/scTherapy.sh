#!/bin/bash
#SBATCH -J scTherapy
#SBATCH -o patient6_scTherapy.%j.out
#SBATCH -e patient6_scTherapy.%j.err
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --mem=50G
#SBATCH --cpus-per-task=5
#SBATCH --partition=big
#SBATCH --account=dsls

singularity exec --bind /data/scratch/vonkl01/:/data/scratch/vonkl01/ docker://kmnader/sctherapy_v5 bash -c "Rscript scTherapy_plus.R"