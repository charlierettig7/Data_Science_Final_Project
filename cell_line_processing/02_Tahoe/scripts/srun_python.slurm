#!/bin/bash
#SBATCH -J python
#SBATCH -o python.%j.out
#SBATCH -e python.%j.err
#SBATCH -t 08:00:00
#SBATCH -n 1
#SBATCH --mem=250G
#SBATCH --partition=big
#SBATCH --cpus-per-task=3

source /home/olgan96/.env/bin/activate
python3 03a_plates_filter.py
