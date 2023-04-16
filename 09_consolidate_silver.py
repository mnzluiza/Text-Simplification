import pandas as pd
import os
from datetime import datetime

import warnings
warnings.simplefilter("ignore")

print(f'PROCESSO DE CONSOLIDAÇÃO DA CAMADA SILVER INICIALIZADO! - {datetime.now()}')

full_bronze = pd.DataFrame()
for root, dirs, files in os.walk(os.getcwd() + "\\outputs\\silver\\"):
    for file in files:
        if "nilc_metrix" in file:
            nilc_metrix = pd.read_parquet(root + file)
        elif ".parquet" in file and not "full_silver" in file:
            p_silver = pd.read_parquet(root + file)
            embed_name = p_silver.columns[-1]
            p_silver = pd.DataFrame(p_silver.pop(embed_name).values.tolist(), index=p_silver['key']).add_prefix(f"{embed_name}_")
            
            if full_bronze.empty:
                full_bronze = pd.read_parquet(f"{os.getcwd()}\\outputs\\bronze\\full_bronze.parquet")
                full_silver = full_bronze.merge(p_silver, left_on='key', right_index=True, how='left').reset_index(drop=True)
            else:
                full_silver = full_silver.merge(p_silver, left_on='key', right_index=True, how='left').reset_index(drop=True)

full_silver = full_silver.merge(nilc_metrix, on='key', how='left').reset_index(drop=True)
full_silver.to_parquet(f"{os.getcwd()}\\outputs\\silver\\full_silver.parquet")
full_silver.to_excel(f"{os.getcwd()}\\outputs\\full_silver.xlsx")

print(f'PROCESSO DE CONSOLIDAÇÃO DA CAMADA SILVER CONCLUÍDO! - {datetime.now()}')