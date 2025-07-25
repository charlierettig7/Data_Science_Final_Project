dir.create(
  path = Sys.getenv("R_LIBS_USER"),
  showWarnings = FALSE,
  recursive = TRUE
)
.libPaths(Sys.getenv("R_LIBS_USER"))
# remotes::install_github("bhklab/AnnotationGx")
library(tidyverse)
library(data.table)
library(AnnotationGx)

metadata <- utils::read.delim(
  "data/cellinfo_beta.txt",
  header = TRUE,
  stringsAsFactors = FALSE
)

cell_line_info <- readr::read_csv("data/Model.csv") |>
  dplyr::select(
    "ModelID",
    "CellLineName",
    "StrippedCellLineName",
    "CatalogNumber",
    "CCLEName"
  )

mutations <- data.table::fread("data/OmicsSomaticMutations.csv") |>
  dplyr::left_join(cell_line_info, by = dplyr::join_by(ModelID))

unique_names <- unique(mutations$CellLineName)
name_mapping <- data.table::data.table(
  CellLineName = unique_names,
  # this is the call that gets us the necessary information
  cellosaurus_ids = AnnotationGx::mapCell2Accession(unique_names)
)

mutations_cellosaurus_ids <- merge(mutations, name_mapping, by = "CellLineName")

data.table::fwrite(
  mutations_cellosaurus_ids,
  "data/mutations_cellosaurus_full.csv"
)
