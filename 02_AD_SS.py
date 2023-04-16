from tools.ss.ss import ss
from tools.utils import write
import pandas as pd
import os
from datetime import datetime
import nltk
import spacy

print(f'SUBSTITUIÇÃO POR SINÔNIMO INICIALIZADO! - {datetime.now()}')

# PARÂMETROS DO USUÁRIO
nr_texts = 5  # número máximo de textos a serem gerados a partir de um original 
content_verbose = True # habilitar caso queira salvar em txt as versões geradas
context_type = "any"  # indica qual dicionário de substituições será aplicado: number_gender ou any
keep_prev_results = True # quando habilitado, não gera substituições para os textos que já tenham tido substituições geradas

# Carregar dados
museum_texts = pd.read_parquet(f"{os.getcwd()}\\outputs\\bronze\\museu.parquet")

if keep_prev_results:
    backup = pd.read_parquet(f"{os.getcwd()}\\outputs\\bronze\\museu_ss_{context_type}.parquet")
    backup_titles = list(backup['Title'])
else:
    backup = pd.DataFrame()
    backup_titles = []

# Adicionando variáveis de apoio
modified_content = [None]*nr_texts
modified_keys = list(range(1,nr_texts+1))
museum_texts['Modified_Key'] = [modified_keys.copy() for _ in museum_texts.index]
museum_texts['Modified_Content'] = [modified_content.copy() for _ in museum_texts.index]

# Carregando contexto de palavras
if context_type=="number_gender":
    try:
        df_replacements = pd.read_parquet(os.getcwd()+f"\\tools\\ss\\context_number_gender.parquet")
    except FileNotFoundError:
        df_replacements = ss.generate_replacements_dict('number_gender')
elif context_type=="any":
    try:
        df_replacements = pd.read_parquet(os.getcwd()+f"\\tools\\ss\\context_any.parquet")
    except FileNotFoundError:
        df_replacements = ss.generate_replacements_dict('any')

# Carregando MLs
nltk.download('punkt')
sent_tokenizer = nltk.data.load('tokenizers/punkt/portuguese.pickle')
sp = spacy.load('pt_core_news_sm')

# Iterando pelos textos selecionados para gerar versões a partir da substituição léxica
for i, row in museum_texts.iterrows():
    print(f"********* Processando {row['Title']} - {datetime.now()} *********")
    if row['Title'] in backup_titles:
        print("Este texto já foi processado. Encerrando processamento...")
    else:
        if row['Label'] == "Simples":
            [output, df_replacements] = ss.replace_words(row['Content'], sp, df_replacements, nr_texts=nr_texts)
        else:
            [output, df_replacements] = ss.replace_complex_words(row['Content'], sp, df_replacements, nr_texts)

        for j in range(nr_texts):
            museum_texts.loc[i].iloc[-1][j] = output[j].replace(" , ", ", ").replace(" : ", ": ").replace(" . ", ". ").replace(" ; ", "; ").replace(" ) ", ") ").replace(" ( ", " (").replace(" ? ", "? ").replace("  ", " ")
            if content_verbose:
                write.str2txtFile(os.getcwd()+f"\\outputs\\others\\ss_txts\\{context_type}\\{row['Title'][:-4]}.txt", row['Content'])
                write.str2txtFile(os.getcwd()+f"\\outputs\\others\\ss_txts\\{context_type}\\{row['Title'][:-4]}_mod{j}_{context_type}.txt", museum_texts.loc[i].iloc[-1][j])

museum_texts = museum_texts.explode(['Modified_Content', 'Modified_Key']).drop_duplicates().reset_index(drop=True)
museum_texts['key'] = f"{museum_texts['key'].astype(str)}_v{museum_texts['Modified_Key'].astype(str)}_{context_type}"
museum_texts['Corpus'] = f"SS_Museu_{context_type}"
museum_texts = museum_texts[['key', 'Title', 'Corpus', 'Label', 'Modified_Content']]
museum_texts.rename(columns={'Modified_Content':'Content'}, inplace=True)

if keep_prev_results:
    museum_texts = museum_texts[~(museum_texts['key'].isin(backup['key'].tolist()))]
    museum_texts = backup.append(museum_texts).reset_index(drop=True)

museum_texts.to_parquet(f"{os.getcwd()}\\outputs\\bronze\\museu_ss_{context_type}.parquet")
