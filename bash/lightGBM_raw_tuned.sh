#!/bin/bash
#SBATCH -J LGBM_rt 
#SBATCH -o find_cell.%j.out
#SBATCH -e find_cell.%j.err
#SBATCH --time=72:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=250G
#SBATCH --partition=scavenger
#SBATCH --account=agjahn
#SBATCH --qos=standard

hostname
date

python3 src/models/lightGBM/lightGBM_raw_tuned.py
