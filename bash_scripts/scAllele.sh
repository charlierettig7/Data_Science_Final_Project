#!/bin/bash
#SBATCH -J scAllele 
#SBATCH -o pipe_scAllele.%j.out
#SBATCH -e pipe_scAllele.%j.err
#SBATCH --time=2-00:00:00
#SBATCH --nodes=1
#SBATCH --mem=40G
#SBATCH --cpus-per-task=12
#SBATCH --partition=big
#SBATCH --account=dsls

export SINGULARITY_HOME="/home/vonkl01/:/home/vonkl01/"
# This script runs the reccomended preprocessing steps for 
# single cell variant calling using scAllele
conda init
conda create -n variant_calling python=3.10 scipy=1.11.4
conda activate variant_calling

fastqc -t 8 -o NO_BACKUP/fastqc_output NO_BACKUP/*.fastq
samtools faidx NO_BACKUP/hs1.fa

# this was not needed in the end
#conda create -n agat_env python=3.10
#conda activate agat_env
#conda install -c bioconda -c conda-forge agat

#agat_convert_sp_gff2gtf.pl \
#    --gff NO_BACKUP/Homo_sapiens-GCA_009914755.4-2022_07-genes.gff3 \
#    -o NO_BACKUP/Homo_sapiens-GCA_009914755.4-2022_07-genes.gtf

STAR-2.7.11b/source/STAR \
     --runThreadN 8 \
     --runMode genomeGenerate \
     --genomeDir NO_BACKUP/T2T \
     --genomeFastaFiles NO_BACKUP/hs1.fa \
     --sjdbGTFfile NO_BACKUP/hs1.ncbiRefSeq.gtf \
     --sjdbOverhang 89


SAMPLES=("SRR30720406" "SRR30720407" "SRR30720408")
for ID in "${SAMPLES[@]}"
do
    echo "Processing $ID..."

    # Define input files
    R1="NO_BACKUP/${ID}_S1_R1.fastq"
    R2="NO_BACKUP/${ID}_S1_R2.fastq"

    # Define output prefix
    OUT_PREFIX="NO_BACKUP/res_STAR/${ID}_"
    OUT_PREFIX_PICARD="NO_BACKUP/res_picard/${ID}_"
    OUT_PREFIX_SAMTOOLS="NO_BACKUP/res_sam/${ID}_"
    OUT_PREFIX_SAMTOOLS="res_sam/${ID}_"
    OUT_PREFIX_SCALLELE="res_scAllele/${ID}_"
    # Run STAR
    STAR-2.7.11b/source/STAR \
         --genomeDir NO_BACKUP/T2T \
         --readFilesIn "$R1" "$R2" \
         --runThreadN 10 \
         --quantMode GeneCounts \
         --twopassMode Basic \
         --outFileNamePrefix "$OUT_PREFIX" \
         --outSAMtype BAM SortedByCoordinate
    
    samtools addreplacerg \
        -r "@RG\tID:RG1\tSM:SampleName\tPL:Illumina\tLB:Library.fa" \
        -o ${OUT_PREFIX}RG.bam \
        ${OUT_PREFIX}Aligned.sortedByCoord.out.bam

    # mark PCR duplicates
    java -jar picard.jar MarkDuplicates \
        -I ${OUT_PREFIX}RG.bam \
        -O ${OUT_PREFIX_PICARD}marked_dup.bam \
        -M ${OUT_PREFIX_PICARD}dup_metrics.txt

    samtools sort \
        ${OUT_PREFIX_PICARD}marked_dup.bam \
        -o ${OUT_PREFIX_SAMTOOLS}sorted.bam
    
    samtools index ${OUT_PREFIX_SAMTOOLS}sorted.bam

    # this runs scAllele
    singularity exec --bind /data/scratch/vonkl01/:/data/scratch/vonkl01/ s5_latest.sif bash -c "ls NO_BACKUP && scAllele -b /data/scratch/vonkl01/${OUT_PREFIX_SAMTOOLS}sorted.bam -g /home/vonkl01/hs1.fa -o /data/scratch/vonkl01/${OUT_PREFIX_SCALLELE}"
    
    echo "Finished $ID"

done