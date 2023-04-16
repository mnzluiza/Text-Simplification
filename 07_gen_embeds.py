import os
import pandas as pd
from datetime import datetime

from tools.embedding.embeddings import embeddings

import fasttext
from gensim.models import KeyedVectors
from transformers import AutoTokenizer 
from transformers import AutoModel 

import warnings
warnings.simplefilter("ignore")

# CARREGA TEXTOS
print(f'CARREGAMENTO DO CORPUS INICIALIZADO - {datetime.now()}')

bronze = pd.read_parquet(f"{os.getcwd()}\\outputs\\bronze\\full_bronze.parquet")
keep_prev_results = False # quando habilitado, não gera embeddings para os textos que já tenham tido embeddings gerados

if keep_prev_results:
    backup_fb_cbow = pd.read_parquet(f"{os.getcwd()}\\outputs\\silver\\fb_cbow.parquet")
    backup_nilc_sg = pd.read_parquet(f"{os.getcwd()}\\outputs\\silver\\nilc_sg.parquet")
    backup_bertimbau = pd.read_parquet(f"{os.getcwd()}\\outputs\\silver\\bertimbau.parquet")
    backup_titles = list(backup_bertimbau['key'])
    alldocs = bronze[~(bronze['key'].isin(backup_titles))]
else:
    alldocs = bronze.copy()

print(f'CARREGAMENTO DO CORPUS CONCLUÍDO - {datetime.now()}')

# GERA VETORES PARA TEXTOS
# ESTÁTICOS
fb_model = fasttext.load_model(f"{os.getcwd()}\\tools\\embedding\\fasttext\\cc.pt.300.bin")
print(f'*** FASTTEXT FB: Leitura do modelo concluída - {datetime.now()} ***')
fb_output = embeddings.fasttext_fb(alldocs, fb_model, bronze)
if keep_prev_results:
    fb_output = backup_fb_cbow.append(fb_output).reset_index(drop=True)
fb_output.to_parquet(f"{os.getcwd()}\\outputs\\silver\\fb_cbow.parquet")

nilc_model = KeyedVectors.load_word2vec_format(f"{os.getcwd()}\\tools\\embedding\\word2vec\\skip_s300.txt")
print(f'*** WORD2VEC NILC: Leitura do modelo concluída - {datetime.now()} ***')
nilc_output = embeddings.word2vec_nilc(alldocs, nilc_model, bronze)
if keep_prev_results:
    nilc_output = backup_nilc_sg.append(nilc_output).reset_index(drop=True)
nilc_output.to_parquet(f"{os.getcwd()}\\outputs\\silver\\nilc_sg.parquet")

# CONTEXTUALIZADOS
bert_tokenizer = AutoTokenizer.from_pretrained('neuralmind/bert-base-portuguese-cased', do_lower_case=False)
bert_model = AutoModel.from_pretrained('neuralmind/bert-base-portuguese-cased')
print(f'*** BERTIMBAU: Leitura do modelo concluída - {datetime.now()} ***')
bert_output = embeddings.bertimbau(alldocs, bert_model, bert_tokenizer, bronze)
if keep_prev_results:
    bert_output = backup_bertimbau.append(bert_output).reset_index(drop=True)
bert_output.to_parquet(f"{os.getcwd()}\\outputs\\silver\\bertimbau.parquet")