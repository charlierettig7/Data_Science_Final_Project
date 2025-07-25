library(tidyverse)
library(data.table)

# read in metadata to get some exploratory info on how many of our
# lincs cell lines are in the mutation calling file
metadata <- utils::read.delim("data/cellinfo_beta.txt", header = TRUE, stringsAsFactors = FALSE)

cell_lines_lincs <- metadata |>
  dplyr::select(cell_iname, cell_type, ccle_name, cell_alias)

cell_line_info <- read_csv("data/Model.csv") |>
  dplyr::select(ModelID, CellLineName, StrippedCellLineName, CatalogNumber, CCLEName)

# there are 240 cell lines in lincs
base::nrow(cell_lines_lincs)

# there is the following composition
base::table(cell_lines_lincs$cell_type)
# normal   pool  tumor
# 23     29    188

base::intersect(cell_lines_lincs$ccle_name, cell_line_info$CCLEName) |>
  base::length()
# we have an intersect of 137

not_included <- cell_lines_lincs |>
  dplyr::filter(!ccle_name %in% cell_line_info$CCLEName) |>
  dplyr::filter(cell_type == "tumor")

mutations <- readr::read_csv("data/OmicsSomaticMutations.csv") |>
  dplyr::left_join(cell_line_info, by = dplyr::join_by(ModelID))

# since this takes a long time, we ran it on the server instead to get the annotations for all cell lines
AnnotationGx::mapCell2Accession(
  c("MA-MEL-46"),
  numResults = 10000,
  from = "idsy",
  # sort = "ac",
  keep_duplicates = FALSE,
  fuzzy = FALSE,
  query_only = FALSE,
  raw = FALSE,
  parsed = TRUE
)


# merge datasets
library(arrow)
mutation_annotated <- data.table::fread("data/mutations_cellosaurus_full.csv")
lincs <- arrow::read_parquet("data/signature_response_features_r2_top0.7_final.parquet")

data_mut <- mutation_annotated |> dplyr::select(
  cellosaurus_ids.accession,
  HugoSymbol,
  Chrom,
  Pos,
  Ref,
  Alt,
  ProteinChange,
  AF,
  VariantType,
  VariantInfo,
  VepImpact,
  VepBiotype,
  RevelScore,
  VepLofTool,
  LikelyLoF,
  OncogeneHighImpact,
  TumorSuppressorHighImpact,
  AMClass,
  AMPathogenicity,
  Hotspot
)
