# this file is a trimmed version from merge_variants_into_lincs.ipynb that we used to run it on the server
import pandas as pd
import numpy as np

# read in data
lincs_path = 'data/signature_response_features_r2_top0.7_final.parquet'
lincs = pd.read_parquet(lincs_path, engine='fastparquet')

variant_path = 'data/mutations_cellosaurus_full.csv'
variants = pd.read_csv(variant_path)
# subsample variants to speed up and use less memory
variants = variants[variants['cellosaurus_ids.accession'].isin(lincs['cellosaurus_id'])]

variants_subset = variants[[
    'cellosaurus_ids.accession',
    'HugoSymbol',
 #   'Chrom',
    'DNAChange',
    'Pos',
 #   'Ref',
 #   'Alt',
 #   'ProteinChange',
 #   'AF',
    'VariantType',
 #   'VariantInfo',
 #   'VepImpact',
 #   'VepBiotype',
 #   'RevelScore',
 #   'VepLofTool',
 #   'LikelyLoF',
 #   'OncogeneHighImpact',
 #   'TumorSuppressorHighImpact',
 #   'AMClass',
 #   'AMPathogenicity',
 #   'Hotspot'
]]

# get info on mutations and aggregate them by cell line
all_mutations = set(variants_subset['DNAChange'])
all_mutations_list = list(all_mutations)
variant_by_cell_line = {}
for cell_line, group in variants_subset[['cellosaurus_ids.accession', 'DNAChange']].groupby('cellosaurus_ids.accession'):
    # Convert group to list of dictionaries, excluding the grouping column
    variant_by_cell_line[cell_line] = group['DNAChange'].tolist()


lincs['mutations'] = lincs['cellosaurus_id'].map(variant_by_cell_line)
lincs['mutations'] = lincs['mutations'].apply(lambda x: x if isinstance(x, list) else [])

print("Setup Done and starting creation of matrix", flush = True)
mutation_presence = pd.DataFrame(0, index=range(lincs.shape[0]), columns=all_mutations_list, dtype=bool)
# use pandas to efficiently create the columns that we then add to the existing lincs data
mutation_array = mutation_presence.values
mutation_cols = {col: i for i, col in enumerate(mutation_presence.columns)}

for idx, muts in enumerate(lincs['mutations']):
    if (idx % 5000 == 0):
        print(f"Iteration {idx} ....", flush = True)
    if isinstance(muts, list) and len(muts) > 0:
        col_indices = [mutation_cols[mut] for mut in muts if mut in mutation_cols]
        mutation_array[idx, col_indices] = True

# Convert back to DataFrame
mutation_presence = pd.DataFrame(mutation_array, 
                                index=mutation_presence.index, 
                                columns=mutation_presence.columns,
                                dtype=bool)
                                
print("Concatenating ...", flush = True)
lincs_with_mutations = pd.concat([lincs, mutation_presence], axis=1)

print("Write to file ...", flush = True)
lincs_with_mutations.to_parquet(path='data/lincs_top0.7_variant_matrix.parquet', engine='pyarrow')