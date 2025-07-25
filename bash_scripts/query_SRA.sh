#!/bin/bash
#SBATCH -J get_SRA 
#SBATCH -o get_SRA.%j.out
#SBATCH -e get_SRA.%j.err
#SBATCH --time=05:00:00
#SBATCH --nodes=1
#SBATCH --mem=15G
#SBATCH --cpus-per-task=8
#SBATCH --partition=small
#SBATCH --account=dsls

cd /home/vonkl01/NO_BACKUP
# retrieve fastq files from SRA in a quick way
fasterq-dump SRR30720408

fasterq-dump SRR30720407

fasterq-dump SRR30720406