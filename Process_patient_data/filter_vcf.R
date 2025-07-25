# This code processes the scAllele varient calls
# and summarizes to gene level as our model expects the input
# Patient 5  = SRR30720408
# Patient 6  = SRR30720407
# Patient 12 = SRR30720406

library(vcfR)
library(tidyverse)
library(data.table)
library(arrow)

genes_in_lincs <- arrow::read_parquet("data/genes_info.parquet")
# read reference file
gtf_genes <- data.table::fread("data/hs1.ncbiRefSeq.gtf", sep = "\t", header = FALSE) |>
  magrittr::set_colnames(
    c(
      "seqname",
      "source",
      "feature",
      "start",
      "end",
      "score",
      "strand",
      "frame",
      "attribute"
    )
  ) |>
  # the last column attribute contains a variable amount of extra information on a sequence, including gene name
  # because the length varies and therefore also the position of each information, we need this somewhat complicated solution
  dplyr::mutate(row = row_number()) %>%
  dplyr::mutate(split = str_split(attribute, pattern = ";\\s*")) %>%
  tidyr::unnest(split) %>%
  dplyr::filter(split != "") %>%
  # match key-value pairs
  tidyr::extract(split, into = c("key", "value"), regex = "^(\\S+)\\s+(.*)$") %>%
  dplyr::mutate(value = str_replace_all(value, "\"", "")) %>%
  # pivot wide to create new columns based on derived key - value pairs
  tidyr::pivot_wider(
    names_from = key,
    values_from = value,
    values_fn = list(value = first)
  ) %>%
  dplyr::select(-row) |>
  # now we end up with gene start and end coordinates to which we can map our mutations
  dplyr::group_by(gene_id, seqname) |>
  dplyr::summarize(start = min(start), end = max(end))

vcf_files <- c("SRR30720408_.vcf", "SRR30720407_.vcf", "SRR30720406_.vcf")
patients <- c(5, 6, 12)
# create a loop to avoid unneccesary duplication of code
# could have also created a function

for (i in 1:3){
  
  vcf <- read.vcfR(paste0("data/", vcf_files[i]))
  
  muts <- vcf |> 
    vcfR2tidy() |> 
    (\(x) x$fix)() |> 
    # only use muts that pass the QC of scAllele
    dplyr::filter(FILTER == "PASS") 
  
  # Prepare data tables for efficient overlap finding
  setDT(muts)[, `:=`(start = POS, end = POS)]
  setDT(gtf_genes)
  
  # Set interval keys (also needed for overlaps)
  setkey(gtf_genes, seqname, start, end)
  
  # Perform overlap join
  result <- muts |> 
    data.table::foverlaps(gtf_genes, by.x = c("CHROM", "start", "end"), by.y = key(gtf_genes), type = "within") |> 
    dplyr::filter(!is.na(gene_id))
  # this file contains all mutations for a patient, not summarized by gene in case we want to do additional analyses or processing later on
  readr::write_csv(result, paste0("data/mutation_pat", patients[i],".csv"))
  
  summarized <- result |> 
    dplyr::filter(gene_id %in% genes_in_lincs$gene_symbol) |> 
    dplyr::group_by(gene_id) |> 
    dplyr::summarize(n_muts = n()) |> 
    tidyr::pivot_wider(names_from = gene_id,
                       values_from = n_muts)
  # this file contains a column for each gene and one row containing the count of mutations
  readr::write_csv(summarized, paste0("data/aggregated_muts_", patients[i],".csv"))
  print(paste0("Patient ", patients[i], " fully processed."))
}
