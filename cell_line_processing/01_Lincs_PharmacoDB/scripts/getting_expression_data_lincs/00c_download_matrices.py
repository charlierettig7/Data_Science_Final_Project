import os
import numpy as np
import requests
#!pip3 install matplotlib
import matplotlib.pyplot as plt
import sys
sys.path.insert(0,'/home/olgan96/.env/lib/python3.11/site-packages/')
#!pip3 install cmapBQ -q
#!pip3 install pandas -q

import pandas as pd
import cmapBQ.query as cmap_query
import cmapBQ.config as cmap_config
# URL with credentials
url = ('https://s3.amazonaws.com/data.clue.io/api/bq_creds/BQ-demo-credentials.json')

response = requests.get(url)
credentials_filepath='BQ-demo-credentials.json'

with open(credentials_filepath, 'w') as f:
    f.write(response.text)

# Set up credentials
cmap_config.setup_credentials(credentials_filepath)
bq_client = cmap_config.get_bq_client()

sig_ids = pd.read_parquet('NO_BACKUP/sig_info.parquet').sig_id.values
n = len(sig_ids)//4000

for i in range(0, 1):
    print(i)
    batch = sig_ids[i * 4000 :(i + 1) * 4000]
    #cmap_query.cmap_matrix(bq_client, cid=list(batch), verbose=False,).data_df.T.to_parquet('NO_BACKUP/matrices/matrix_' + str(i) + '.parquet')
