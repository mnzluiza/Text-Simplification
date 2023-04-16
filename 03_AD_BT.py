import pandas as pd
import os
from datetime import datetime

from deep_translator import GoogleTranslator
from tools.utils import write

# PARÂMETROS DO USUÁRIO
content_verbose = True # habilitar caso queira salvar em txt as versões geradas

print(f'BACKTRANSLATION INICIALIZADO! - {datetime.now()}')

# Carregar dados
museum_texts = pd.read_parquet(f"{os.getcwd()}\\outputs\\bronze\\museu.parquet")
museum_texts['Modified_Content'] = ""

# Itera textos para executar BT
for i, row in museum_texts.iterrows():
    if row['Label'] == "Simples":
        bt_text = GoogleTranslator(source='en', target='pt').translate(GoogleTranslator(source='pt', target='en').translate(row['Content']))
    else:
        bt_text = GoogleTranslator(source='it', target='pt').translate(GoogleTranslator(source='pt', target='it').translate(row['Content']))
    museum_texts.loc[i, 'Modified_Content'] = bt_text
    if content_verbose:
        write.str2txtFile(os.getcwd()+f"\\outputs\\others\\bt_txts\\{row['Title'][:-4]}_BT_google.txt", bt_text)

# Ajustes para salvar
museum_texts['key'] = museum_texts['key'].astype(str) + "_BT_google"
museum_texts['Corpus'] = "BT_Museu"
museum_texts = museum_texts[['key', 'Title', 'Corpus', 'Label', 'Modified_Content']]
museum_texts.rename(columns={'Modified_Content':'Content'}, inplace=True)
museum_texts = museum_texts.reset_index(drop=True)

# Salva consolidado
museum_texts.to_parquet(f"{os.getcwd()}\\outputs\\bronze\\museu_bt.parquet")

print(f'BACKTRANSLATION CONCLUÍDO! - {datetime.now()}')