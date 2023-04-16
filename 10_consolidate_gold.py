import pandas as pd
import os 
from sklearn.preprocessing import StandardScaler
from datetime import datetime

def _normalize(value_columns, df):
    metrics_only = df[value_columns]
    metrics_only = metrics_only.fillna(0)
    scaler = StandardScaler()
    norm_metrics = scaler.fit_transform(metrics_only)
    norm_metrics = pd.DataFrame(norm_metrics, index=df['key'], columns=value_columns)
    # norm_metrics.dropna(axis=1, inplace=True)
    return norm_metrics

def normalize_individually(df, attr_columns):
    # nilc_metrix
    p1 = _normalize([x for x in df.columns if ('BERTimbau' not in x) and ('FB_CBOW' not in x) and ('NILC_SG' not in x) and (x not in attr_columns)], df)
    # bertimbau
    p2 = _normalize([x for x in df.columns if 'BERTimbau' in x], df)
    # fb_cbow
    p3 = _normalize([x for x in df.columns if 'FB_CBOW' in x], df)
    # nilc_sg
    p4 = _normalize([x for x in df.columns if 'NILC_SG' in x], df)
    return df[attr_columns].merge(p1, left_on=['key'], right_index=True, how='left').merge(p2, left_on=['key'], right_index=True, how='left').merge(p3, left_on=['key'], right_index=True, how='left').merge(p4, left_on=['key'], right_index=True, how='left').reset_index(drop=True)
    
def normalize_all(df, attr_columns):
    value_columns = [x for x in df.columns if x not in attr_columns]
    normalize_all = _normalize(value_columns, df)
    return df[attr_columns].merge(normalize_all, left_on=['key'], right_index=True, how='left')

def generate_gold(attr_columns = ['key', 'Title', 'Corpus', 'Label', 'Content']):
    # Read silver
    silver = pd.read_parquet(f"{os.getcwd()}\\outputs\\silver\\full_silver.parquet")

    # Normalize results
    normalized_gold = normalize_all(silver, attr_columns)
    # normalized_by_parts_gold = normalize_individually(silver, attr_columns)

    # Export results
    normalized_gold.to_parquet(f"{os.getcwd()}\\outputs\\gold\\full_gold.parquet")
    # normalized_by_parts_gold.to_parquet(f"{os.getcwd()}\\outputs\\gold\\full_gold_normalized_by_parts.parquet")

    normalized_gold.to_excel(f"{os.getcwd()}\\outputs\\full_gold.xlsx")
    # normalized_by_parts_gold.to_excel(f"{os.getcwd()}\\outputs\\full_gold_normalized_by_parts.xlsx")

print(f'PROCESSO DE NORMALIZAÇÃO E ESTRUTURAÇÃO DAS MÉTRICAS DO CORPUS COMPLETO INICIALIZADO - {datetime.now()}')
generate_gold()
print(f'PROCESSO DE NORMALIZAÇÃO E ESTRUTURAÇÃO DAS MÉTRICAS DO CORPUS COMPLETO CONCLUÍDO - {datetime.now()}')