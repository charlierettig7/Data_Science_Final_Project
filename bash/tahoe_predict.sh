#!/bin/bash
#SBATCH -J pred
#SBATCH -o find_cell.%j.out
#SBATCH -e find_cell.%j.err
#SBATCH --time=72:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=30G
#SBATCH --partition=main
#SBATCH --account=agjahn
#SBATCH --qos=standard

hostname
date

python3 src/predictions/tahoe_predict.py
