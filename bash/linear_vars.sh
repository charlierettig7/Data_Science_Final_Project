#!/bin/bash
#SBATCH -J lin_vnt 
#SBATCH -o find_cell.%j.out
#SBATCH -e find_cell.%j.err
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=150G
#SBATCH --partition=scavenger
#SBATCH --account=agjahn
#SBATCH --qos=standard

hostname
date

python3 src/models/linear/linear_vars.py