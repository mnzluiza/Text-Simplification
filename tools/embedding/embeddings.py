import os
from datetime import datetime

import numpy as np
import math
import torch

from nltk.tokenize import word_tokenize

from tools.utils import utils
from tools.utils import write

class embeddings:
    def df_2_list(df, stop_words=[], df_content_column_name='Content'):
        _list = []
        i = 0
        while i < df.shape[0]:
            line = df.iloc[i][df_content_column_name]
            if stop_words!=[]:
                tokens = word_tokenize(utils.clean_text(line, stop_words))
            else:
                tokens = word_tokenize(line)
            if tokens:
                _list.append(tokens)
            i += 1
        return _list
    
    def fasttext_fb(alldocs, model, _df, ndim=300):
        # FASTTEXT - FACEBOOK
        # https://fasttext.cc/docs/en/crawl-vectors.html
        print(f'PROCESSO DE GERAÇÃO DE EMBEDDING ESTÁTICO (FACEBOOK) POR FASTTEXT INICIALIZADO - {datetime.now()}')

        fb_cbow_dim = list(range(1,ndim+1))
        df = _df.copy()

        df['FB_CBOW'] = [fb_cbow_dim.copy() for _ in df.index]

        for k, row in alldocs.iterrows():
            print(f"***** processando {df.loc[k, 'Title']} - {datetime.now()}")
            text_vec = []
            for sentence in row['Content'].splitlines():
                text_vec.append(model.get_sentence_vector(sentence))
            text_mean_vec = np.mean(text_vec, axis=0)
            for i in fb_cbow_dim:
                df.loc[k, 'FB_CBOW'][i-1] = text_mean_vec[i-1]

        print(f'PROCESSO DE GERAÇÃO DE EMBEDDING ESTÁTICO (FACEBOOK) POR FASTTEXT CONCLUÍDO - {datetime.now()}')
        return df

    def word2vec_nilc(alldocs, model, _df, ndim=300):
        # WORD2VEC - NILC - SKIP GRAM 300 DIMENSÕES
        # n-gram de caractere
        # http://nilc.icmc.usp.br/embeddings
        print(f'PROCESSO DE GERAÇÃO DE EMBEDDING ESTÁTICO POR WORD2VEC (NILC) INICIALIZADO - {datetime.now()}')

        nilc_sg_dim = list(range(1,ndim+1))
        df = _df.copy()
        df['NILC_SG'] = [nilc_sg_dim.copy() for _ in df.index]
        oov = []
        for k, row in alldocs.iterrows():
            print(f"***** processando {df.loc[k, 'Title']} - {datetime.now()}")
            text_vec = []
            for token in word_tokenize(row['Content']):
                try:
                    text_vec.append(model.word_vec(token))
                except:
                    oov.append(token)
            text_mean_vec = np.mean(text_vec, axis=0)
            if len(text_mean_vec)>0:
                for i in nilc_sg_dim:
                    df.loc[k, 'NILC_SG'][i-1] = text_mean_vec[i-1]
            else:
                df.loc[k, 'NILC_SG'][i-1] = None

        write.str2txtFile(f"{os.getcwd()}\\outputs\\others\\NILC_SG_oov.txt", "\n".join(oov))
        print(f'PROCESSO DE GERAÇÃO DE EMBEDDING ESTÁTICO POR FASTTEXT CONCLUÍDO - {datetime.now()}')
        return df
    
    def bertimbau(alldocs, model, tokenizer, _df, ndim=768):
        print(f'PROCESSO DE GERAÇÃO DE EMBEDDING CONTEXTUALIZADO INICIALIZADO - {datetime.now()}')
        # https://stackoverflow.com/questions/65023526/runtimeerror-the-size-of-tensor-a-4000-must-match-the-size-of-tensor-b-512

        bertimbau_dim = list(range(1,ndim+1))
        df = _df.copy()
        df['BERTimbau'] = [bertimbau_dim.copy() for _ in df.index]

        for k, row in alldocs.iterrows():
            print(f"***** processando {df.loc[k, 'Title']} - {datetime.now()}")
            max_token = 450
            encoded_mean = []
            encoded = torch.empty((0,768))
            while not encoded.numel() and max_token>0:
                try:
                    if max_token>0 and len(row['Content'].split()) < max_token:
                        input_ids = tokenizer.encode(row['Content'], return_tensors='pt')
                        with torch.no_grad():
                            encoded = model(input_ids)[0][0, 1:-1] # Ignore [CLS] and [SEP] special tokens
                        encoded_mean = torch.mean(encoded, axis=0)
                    else:
                        for i in range(math.ceil(len(row['Content'].split())/max_token)):
                            chunked_text = " ".join(row['Content'].split()[max_token*i:(max_token-1)*(i+1)])
                            input_ids = tokenizer.encode(chunked_text, return_tensors='pt')
                            with torch.no_grad():
                                encoded = model(input_ids)[0][0, 1:-1] # Ignore [CLS] and [SEP] special tokens
                            encoded_mean.append(torch.mean(encoded, axis=0))  
                        encoded_mean = torch.mean(torch.stack(encoded_mean), axis=0)
                except Exception as e:
                    encoded = torch.empty((0,768))
                    encoded_mean = []
                    max_token = max_token - 50
                    print(f"{e}: max_token alterado para {max_token}")
            if max_token <= 0:
                print('!!!!!!!!!erro no processamento!!!!!!!!!')
                df.loc[k, 'BERTimbau'] = np.nan
            else:
                for i in bertimbau_dim:
                    df.loc[k, 'BERTimbau'][i-1] = encoded_mean[i-1].detach().numpy()+0
            
        print(f'PROCESSO DE GERAÇÃO DE EMBEDDING CONTEXTUALIZADO CONCLUÍDO - {datetime.now()}')
        return df