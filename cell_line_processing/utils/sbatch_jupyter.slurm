#!/bin/bash
#SBATCH -J JupyterServer
#SBATCH -o jupyter.%j.out
#SBATCH -e jupyter.%j.err
#SBATCH -t 10:00:00
#SBATCH -n 1
#SBATCH --mem=20G
#SBATCH --partition=big
#SBATCH --cpus-per-task=3

source /home/olgan96/.env/bin/activate

jupyter notebook --no-browser --port=4222 --ip "0.0.0.0"
