print('import functions')
import sys
from tqdm import tqdm as tqdm
sys.path.insert(0,'/home/olgan96/.env/lib/python3.11/site-packages/')
import pandas as pd
import anndata
import numpy as np
import h5py
print('load data')
df = anndata.read_h5ad('../../NO_BACKUP/tahoe/h5ad/plate1_filt_Vevo_Tahoe100M_WServicesFrom_ParseGigalab.h5ad', backed=True)
df_meta = pd.read_parquet('../../NO_BACKUP/tahoe/obs_metadata_filtered.parquet')
df_cell = pd.read_parquet('../../NO_BACKUP/tahoe/cell_lines_id.parquet')
df_genes = pd.read_parquet('../../NO_BACKUP/lincs_2020/common_genes_lincs_tahoe.parquet')
print('construct filters')
rows = list(set(df.obs.index).intersection(set(df_meta['BARCODE_SUB_LIB_ID'])))
rows_dmso = list(df.obs[df.obs['drug'] == 'DMSO_TF'].index)
cols = list(df_genes['gene_symbol'].unique())
#plate_cell_lines = list(set(df.obs['cell_line'].unique()).intersection(set(df_cell['cell_line'])))
#for i in tqdm(range(len(plate_cell_lines))):
#rows_cell = df.obs[df.obs['cell_line'] == plate_cell_lines[i]].index
#rows_final = list(set(rows + rows_dmso).intersection(rows_cell))
print('filter')
df[rows + rows_dmso, cols].write_h5ad('../../NO_BACKUP/tahoe/plates_filtered/plate1_filtered.h5ad')
print('close file')
df.file.close()
print('done')
