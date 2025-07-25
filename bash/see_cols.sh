#!/bin/bash
#SBATCH -J LightGBM_raw_train 
#SBATCH -o find_cell.%j.out
#SBATCH -e find_cell.%j.err
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=30G
#SBATCH --partition=scavenger
#SBATCH --account=agjahn
#SBATCH --qos=standard

hostname
date

python3 src/predictions/see_cols.py
