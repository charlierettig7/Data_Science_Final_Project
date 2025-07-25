#!/bin/bash
#SBATCH -J JupyterServer
#SBATCH -o jupyter.%j.out
#SBATCH -e jupyter.%j.err
#SBATCH -t 03:00:00
#SBATCH -n 1
#SBATCH --mem=20G
#SBATCH --partition=big
#SBATCH --cpus-per-task=1

source /home/olgan96/.env/bin/activate
pip3 install matplotlib
pip3 install cmapBQ -q
pip3 install pandas -q
python3 00c_download_matrices.py
