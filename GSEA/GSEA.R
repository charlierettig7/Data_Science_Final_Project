# this file contains code to run gsea on some of our made predictions
# as an attempt to better understand what the model is able to learn

# Setup
library(fgsea)
library(tidyverse)

comparison_results <- list()
gene_set <- "data/c2.cp.kegg_medicus.v2024.1.Hs.symbols.gmt"

gene_name_to_lincs_id <- arrow::read_parquet("data/genes_info.parquet") |> 
  dplyr::select("gene_id", "gene_symbol") |> 
  dplyr::mutate(gene_id = as.character(gene_id))

# map ids used in lincs to gene name and change to suitable format for GSEA
lincs <- arrow::read_parquet("data/signature_response_features_r2_top0.7_final.parquet") |> 
  dplyr::select(-c("sig_id", "pert_id", "cellosaurus_id", "inchi_key", "cell_drug", "cmap_name", "smiles")) |> 
  dplyr::mutate(index = row_number())

files <- c("data/predictions_xgboost_raw.csv", "data/predictions_xgboost_vars.csv",
           "data/predictions_lgbm_raw.csv", "data/predictions_lgbm_vars.csv")
name <- c("xgboost_raw", "xgboost_vars",
          "lgbm_raw", "lgbm_vars")

# loop through the prediction files and create GSEA
# for 10 best and 10 worst predictions respectively
for(x in 1:4){
  
  prediction <- readr::read_csv(files[x]) |> 
    dplyr::right_join(lincs, by = "index") |> 
    dplyr::mutate(difference = abs(responses - prediction))
  
  top10_indices <- prediction %>%
    mutate(row_id = row_number()) %>%
    arrange(desc(difference)) %>%
    slice(1:10) %>%
    pull(row_id)
  
  bottom10_indices <- prediction %>%
    mutate(row_id = row_number()) %>%
    arrange(difference) %>%
    slice(1:10) %>%
    pull(row_id)
  
  combined <- c(top10_indices, bottom10_indices)
  
  
  for(row in 1:length(combined)){
    # for descriptive output files so it stays easy to trace back
    if (row <= 10) {
      file_name_add <- paste0("_top_", row)
    } else {
      file_name_add <- paste0("_bottom_", row)
    }
    # the row of our prediction of interest
    lincs_row <- lincs |> 
      dplyr::filter(row_number() == combined[row]) |> 
      tidyr::pivot_longer(cols = where(is.numeric), names_to = "ids", values_to = "value") |>
      dplyr::left_join(gene_name_to_lincs_id, by = dplyr::join_by(ids == gene_id)) |> 
      dplyr::filter(!is.na(gene_symbol))
    
    # create GSEA input file by mapping ids to gene names
    ranks <- lincs_row |> 
      dplyr::select(gene_symbol, value) |> 
      magrittr::set_colnames(c("Gene.name", "stat")) |> 
      tidyr::drop_na() |> 
      dplyr::distinct(`Gene.name`, .keep_all = T) %>% 
      {stats::setNames(.$stat, .$`Gene.name`)}
    
    # run GSEA
    pathways <- fgsea::gmtPathways(gene_set)
    fgsea.res_full <- fgsea::fgsea(pathways, ranks)
    fgsea.res <- fgsea.res_full |> 
      base::as.data.frame() |>  
      dplyr::filter(padj < 0.05) |>  #fgsea uses BH adjustment for P values
      dplyr::mutate(leadingEdge = sapply(leadingEdge, function(x) paste(x, collapse = ", ")))
    
    # so we can save the actual values and look at the genes responsible for the pathways being called
    comparison_results[[paste0("GSEA_row_", combined[row], file_name_add)]] <- fgsea.res_full %>%
      as.data.frame() %>%
      dplyr::mutate(leadingEdge = sapply(leadingEdge, function(x) paste(x, collapse = ", "))) %>%
      dplyr::mutate(source = paste0(combined[row], file_name_add, row))
    
    data.table::setDT(fgsea.res)
    
    # only plot pathways that have the highest adjusted p values with negative or positive NES
    pathways.top_up <- fgsea.res[NES > 0 & padj < 0.05][head(order(padj), n = 10), pathway]
    pathways.top_down <- fgsea.res[NES < 0 & padj < 0.05][head(order(padj), n = 10), pathway]
    pathways.top <- c(pathways.top_up, rev(pathways.top_down))
    
    fgsea.top_plot <- fgsea::plotGseaTable(pathways[pathways.top], ranks, fgsea.res, gseaParam = 0.5)
    ggplot2::ggsave(paste0(name[x], "/gsea_row_", combined[row], file_name_add, "_padj.png"), 
                    fgsea.top_plot, width = 22, height = 8, units = "in")
    print(paste0("Row ", row, " done!"))
  }
  print(paste0("--------------- Saving data for ", name[x], " ---------------"))
  readr::write_csv(do.call(rbind, comparison_results), paste0(name[x], "/GSEA_data.csv"))
}
