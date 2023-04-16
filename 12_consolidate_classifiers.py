import os

import pandas as pd
import os
from datetime import datetime

import warnings
warnings.simplefilter("ignore")

print(f'PROCESSO DE CONSOLIDAÇÃO DE CLASSIFICADORES INICIALIZADO! - {datetime.now()}')

gold = pd.read_parquet(f"{os.getcwd()}\\outputs\\gold\\full_gold.parquet")
attr_columns = ['key', 'Title', 'Corpus', 'Label', 'Content']
consolidated_output = gold[attr_columns]
classifier_detail_output = pd.DataFrame()

tamanho_df = len(consolidated_output)
k = 0
for folderpath, _, filenames in os.walk(os.getcwd() + "\\outputs\\others\\classifiers\\parquet\\"):
    for filename in filenames:
        print(f"Processando {k} de {len(filenames)-1} - {filename[:-9]}")
        k=k+1
        classifier = pd.read_parquet(f"{folderpath}{filename}")

        classifier_output = classifier[['key', 'Prediction']]
        classifier_output.rename(columns={'Prediction':f'Prediction_{filename[:-8]}'}, inplace=True)
        classifier_output = classifier_output.drop_duplicates().reset_index(drop=True)
        consolidated_output = consolidated_output.merge(classifier_output, on='key', how='left')

        if len(consolidated_output)>tamanho_df:
            raise(f"Verifique {filename}, processo está sendo encerrado")

        detail = classifier[['Explanation', 'Classifier', 'Metrics_cols', 'Embeds_cols', 'Validation_type', 'Texts_selection']]
        detail = detail.head(1)
        detail['Name'] = filename[:-8]
        classifier_detail_output = classifier_detail_output.append(detail)

consolidated_output.dropna(subset=consolidated_output.iloc[:, len(attr_columns):].columns, how = 'all', inplace=True)
consolidated_output.reset_index(drop=True, inplace=True)
consolidated_output.to_parquet(f"{os.getcwd()}\\outputs\\others\\classifiers\\consolidated.parquet")
classifier_detail_output.to_parquet(f"{os.getcwd()}\\outputs\\others\\classifiers\\consolidation_details.parquet")

consolidated_output.to_excel(f"{os.getcwd()}\\outputs\\classificacoes.xlsx")


print(f'PROCESSO DE CONSOLIDAÇÃO DE CLASSIFICADORES ENCERRADO! - {datetime.now()}')

