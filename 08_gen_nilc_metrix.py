# http://fw.nilc.icmc.usp.br:23380/nilcmetrix
# https://github.com/nilc-nlp/nilcmetrix

import os
import pandas as pd
from datetime import datetime

print(f'CARREGAMENTO DAS MÉTRICAS DO NILC INICIALIZADO - {datetime.now()}')

corpus = pd.read_parquet(f"{os.getcwd()}\\outputs\\bronze\\full_bronze.parquet")
nilc_metrix = pd.DataFrame()
root = f"{os.getcwd()}\\outputs\\others\\nilc_metrix"
for file in os.listdir(root):
    metrics = pd.read_parquet(f"{root}\\{file}")
    metrics['key'] = file.replace('.parquet','')
    nilc_metrix = nilc_metrix.append(metrics)

nilc_metrix.to_parquet(os.getcwd()+ f"\\outputs\\silver\\nilc_metrix.parquet")

print(f'CARREGAMENTO DAS MÉTRICAS DO NILC CONCLUÍDO - {datetime.now()}')