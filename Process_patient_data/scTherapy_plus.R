# run this code to install required libraries
packages <- c(
  "dplyr",
  "Seurat",
  "HGNChelper",
  "openxlsx",
  "copykat",
  "copykatRcpp",
  "ggplot2",
  "SCEVAN",
  "yaGST",
  "cowplot",
  "Rcpp",
  "Rclusterpp",
  "parallel",
  "logger",
  "httr",
  "jsonlite",
  "readr",
  "future"
)

install_load_packages <- function(packages) {
  if (!requireNamespace("BiocManager", quietly = TRUE)) {
    install.packages("BiocManager")
  }
  library("BiocManager")

  if (!requireNamespace("devtools", quietly = TRUE)) {
    install.packages("devtools")
  }
  library("devtools")

  sapply(packages, function(pkg) {
    if (!require(pkg, character.only = TRUE)) {
      if (
        pkg %in%
          c("copykat", "yaGST", "SCEVAN", "Rclusterpp", "copykatRcpp")
      ) {
        tryCatch(
          {
            if (pkg == "copykat") {
              devtools::install_github("navinlabcode/copykat")
            } else if (pkg == "yaGST") {
              devtools::install_github("miccec/yaGST")
            } else if (pkg == "SCEVAN") {
              devtools::install_github("AntonioDeFalco/SCEVAN")
            } else if (pkg == "Rclusterpp") {
              devtools::install_github("nolanlab/Rclusterpp")
            } else if (pkg == "copykatRcpp") {
              devtools::install_github(
                "IanevskiAleksandr/copykatRcpp"
              )
            }
            library(pkg, character.only = TRUE)
          },
          error = function(e) {
            install_from_CRAN_or_Bioconductor(pkg)
          }
        )
      } else {
        install_from_CRAN_or_Bioconductor(pkg)
      }
    }
  })
}

install_from_CRAN_or_Bioconductor <- function(pkg) {
  tryCatch(
    {
      install.packages(pkg)
      library(pkg, character.only = TRUE)
    },
    error = function(e) {
      BiocManager::install(pkg)
      library(pkg, character.only = TRUE)
    }
  )
}

install_load_packages(packages)

# Load required libraries and source functions
invisible(lapply(
  c(
    "dplyr",
    "Seurat",
    "HGNChelper",
    "openxlsx",
    "copykat",
    "copykatRcpp",
    "ggplot2",
    "SCEVAN",
    "yaGST",
    "cowplot",
    "Rcpp",
    "Rclusterpp",
    "parallel",
    "biomaRt",
    "logger",
    "httr",
    "jsonlite",
    "readr",
    "future"
  ),
  library,
  character.only = !0
))

invisible(lapply(
  c(
    "https://raw.githubusercontent.com/kris-nader/scTherapy/main/R/identify_healthy_mal_v5.R",
    "https://raw.githubusercontent.com/kris-nader/scTherapy/main/R/identify_subclones_v5.R",
    "https://raw.githubusercontent.com/kris-nader/scTherapy/main/R/predict_compounds.R"
  ),
  source
))

# Load example dataset
library(data.table)
library(future)
library(ggplot2) # for saving plots

# List of all patient datasets (paths to raw count matrices)
patient_files <- list.files("/home/vonkl01/data/patient_data", full.names = TRUE)

# loop over all patients
for (file in patient_files) {
  patient_name <- sub(".RDS", "", file)
  cat("\nProcessing:", patient_name, "\n")
  
  # Load data and create Seurat object
  data <- GetAssayData(readRDS(file), layer= "counts")
  seu <- CreateSeuratObject(counts = data)
  
  # Filtering
  seu[["percent.mt"]] <- PercentageFeatureSet(seu, pattern = "^MT-")
  seu <- subset(seu, subset = nFeature_RNA > 200 & nFeature_RNA < 2500 & percent.mt < 5)
  
  # Processing and visualization
  seu <- NormalizeData(seu)
  seu <- FindVariableFeatures(seu, selection.method = "vst", nfeatures = 2000)
  seu <- ScaleData(seu)
  seu <- RunPCA(seu)
  ElbowPlot(seu)
  ggsave(filename = paste0("/home/vonkl01/data/scTherapy/", patient_name, "_elbowplot.png"))
  
  # Clustering and UMAP
  seu <- FindNeighbors(seu, dims = 1:10)
  seu <- FindClusters(seu, resolution = 0.8)
  seu <- RunUMAP(seu, dims = 1:10)
  p_umap <- DimPlot(seu, reduction = "umap", group.by = "seurat_clusters")
  ggsave(plot = p_umap, filename = paste0("/home/vonkl01/data/scTherapy/", patient_name, "_UMAP.png"))
  
  # Cell type annotation
  seu <- run_sctype(seu, known_tissue_type = "Immune system", plot = TRUE)
  # Save annotation plot if generated by run_sctype
  
  # Identify normal cells
  norm_cells <- get_normal_cells(seu, c("Memory CD4+ T cells"))
  
  # Ensemble (malignancy) analysis
  seu <- run_ensemble(seu, disease = "AML", known_normal_cells = norm_cells, plot = TRUE)
  visualize_ensemble_step(seu) # Save plot if generated

  # InferCNV (plots saved internally by run_infercnv, or save as needed)
  seu <- run_infercnv(seu)
  
  saveRDS(seu, file = paste0("/home/vonkl01/data/scTherapy/", patient_name, ".RDS"))
  cat("\nDone with:", patient_name, "\n")
}
