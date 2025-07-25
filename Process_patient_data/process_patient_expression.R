library(tidyverse)
library(Seurat)

gene_name_to_lincs_id <- arrow::read_parquet("data/genes_info.parquet") |> 
  dplyr::select("gene_id", "gene_symbol")

#patient_files <- list.files("data/*.RDS", full.names = TRUE)
patient_files <- c("data/patient5.RDS", "data/patient6.RDS")
for (file in patient_files){
  patient_name <- sub(".RDS", "", file)
  patient_name <- sub("data/", "", patient_name)
  patient_sample <- base::readRDS(file)
  print(paste0("N-Cells prefilter: ", length(SeuratObject::Cells(patient_sample))))
  
  # Get the count matrix
  counts <- SeuratObject::GetAssayData(patient_sample, assay = "RNA", layer = "counts")
  # Calculate number of nonzero genes per cell
  gene_count <- Matrix::colSums(counts > 0)
  patient_sample[["gene_count"]] <- gene_count
  
  read_count <- Matrix::colSums(counts)
  patient_sample[["mread_count"]] <- read_count
  
  # use violinplots to investigate sample ranges
  Seurat::VlnPlot(patient_sample, features = c("gene_count", "nCount_RNA", "mread_count"), pt.size = 0.1, log = FALSE)
  
  # enable filtering by cell type
  meta <- patient_sample@meta.data %>%
    dplyr::mutate(cell_id = rownames(patient_sample@meta.data),
                  identity = SeuratObject::Idents(patient_sample)) %>% 
    dplyr::select("cell_id", "identity", "gene_count", "nCount_RNA", "mread_count")
  
  # Define cutoff per identity (adjust cutoffs as desired)
  quantiles <- meta %>%
    dplyr::group_by(identity) %>%
    dplyr::summarise(
      p10_gene_count = quantile(gene_count, 0.10, na.rm = TRUE),
      p90_gene_count = quantile(gene_count, 0.90, na.rm = TRUE),
      p10_nCount_RNA = quantile(nCount_RNA, 0.10, na.rm = TRUE),
      p90_nCount_RNA = quantile(nCount_RNA, 0.90, na.rm = TRUE),
      p10_mread_count = quantile(mread_count, 0.10, na.rm = TRUE),
      p90_mread_count = quantile(mread_count, 0.90, na.rm = TRUE)
    )
  
  cells_kept <- meta %>%
    dplyr::left_join(quantiles, by = "identity") %>%
    dplyr::filter(dplyr::between(gene_count, p10_gene_count, p90_gene_count) &
                    dplyr::between(nCount_RNA, p10_nCount_RNA, p90_nCount_RNA) &
                    dplyr::between(mread_count, p10_mread_count, p90_mread_count)) %>%
    dplyr::pull(cell_id)
  
  patient_filtered <- subset(patient_sample, cells = cells_kept)
  print(paste0("N-Cells postfilter: ", length(SeuratObject::Cells(patient_filtered))))
  
  subclones <- readr::read_tsv(
    paste0("data/17_HMM_predHMMi6.rand_trees.hmm_mode-subclusters.", patient_name, ".cell_groupings")
    ) |> 
    dplyr::mutate(cell_group_name = gsub("all_observations.all_observations.", "", cell_group_name))
  patient_sample[["subclone"]] <- dplyr::pull(subclones, cell_group_name)
  
  
  # transform seurat object to data frame to reuse Olgas exact pseudobulking
  meta <- patient_sample@meta.data |> 
    tibble::rownames_to_column("cell_id") |> 
    dplyr::select(c("cell_id", "subclone"))
  
  counts <- SeuratObject::GetAssayData(patient_sample, assay = "RNA", layer = "counts") |>
    base::as.data.frame() |> 
    tibble::rownames_to_column("gene_symbol") |> 
    dplyr::left_join(gene_name_to_lincs_id, by = "gene_symbol") |> 
    dplyr::filter(!is.na(gene_id)) |> 
    dplyr::select(-"gene_symbol") |> 
    tibble::column_to_rownames("gene_id") |> 
    base::t() |> 
    base::as.data.frame() |> 
    tibble::rownames_to_column("cell_id") |> 
    dplyr::left_join(meta, by = "cell_id")
  
  readr::write_csv(counts, paste0("data/counts_", patient_name, ".csv"))
}
