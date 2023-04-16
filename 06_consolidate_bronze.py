import pandas as pd
import os
from datetime import datetime

import warnings
warnings.simplefilter("ignore")

print(f'PROCESSO DE CONSOLIDAÇÃO DO CORPUS COMPLETO INICIALIZADO! - {datetime.now()}')

full_bronze = pd.DataFrame()
for root, dirs, files in os.walk(os.getcwd() + "\\outputs\\bronze\\"):
    for file in files:
        if ".parquet" in file and not "full_bronze" in file:
            p_bronze = pd.read_parquet(root + file)
            full_bronze = full_bronze.append(p_bronze).reset_index(drop=True)

# limita textos a até 2000 palavras
full_bronze['Content'] = (full_bronze['Content'].str.split().str[:2000]).str.join(' ')

full_bronze.to_parquet(f"{os.getcwd()}\\outputs\\bronze\\full_bronze_with_grade_level.parquet")
full_bronze.drop(columns={'Grade Level'}, inplace=True)
full_bronze.to_parquet(f"{os.getcwd()}\\outputs\\bronze\\full_bronze.parquet")
full_bronze.to_excel(f"{os.getcwd()}\\outputs\\full_bronze.xlsx")

print(f'PROCESSO DE CONSOLIDAÇÃO DO CORPUS COMPLETO CONCLUÍDO! - {datetime.now()}')