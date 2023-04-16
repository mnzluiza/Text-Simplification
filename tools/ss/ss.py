import os
import pandas as pd

from random import random
from nltk.tokenize import word_tokenize
import re

from tools.utils import read

class ss:
    def random_selection(df, ascending=True):
        df.fillna(0, inplace=True)
        df = df.sample(frac=1)
        while True:
            for key, r in df.iterrows():
                random_nr = random()
                if ascending and r['probability'] < random_nr:
                    return key
                elif not(ascending) and r['probability'] > random_nr:
                    return key

    def load_tep(path=os.getcwd()+"\\tools\\base_tep2\\base_tep2.txt"):
            # carrega dicionario de sinonimos (tep2.0)
            sinonimos = pd.DataFrame(read.readFile(path))
            df_sin = pd.DataFrame(columns=['type', 'infinitive'])
            df_sin['type'] = sinonimos[0].str.split(']', 1, expand=True)[0].str.split('[', 1, expand=True)[1]
            df_sin['infinitive'] = sinonimos[0].str.split('{', 1, expand=True)[1].str.split('}', 1, expand=True)[0].apply(lambda x: x.split(','))
            df_sin = df_sin.explode('infinitive').reset_index()
            df_sin.rename(columns={'index': 'key'}, inplace=True)
            return df_sin

    def load_corpop(path=os.getcwd()+"\\tools\\corpop\\Wordlist_CorPop_bruta.txt"):
        # carrega dicionário de palavras simples (corpop)
        corpop = pd.DataFrame(read.readFile(path))
        df_corpop = pd.DataFrame(columns=['hits', 'words'])
        df_corpop['hits'] = corpop.iloc[15:, 0].str.split('\t', expand=True)[1].astype(int)
        df_corpop['words'] = corpop.iloc[15:, 0].str.split('\t', expand=True)[2]

        return df_corpop

    def load_lexicon(path=f"{os.getcwd()}\\tools\portilexicon\\portilexicon-ud.tsv"):
        # carrega https://portilexicon.icmc.usp.br/
        lexicon = read.readXSVFile(path)

        # seleciona apenas as classes gramaticais de interesse
        lexicon = lexicon[lexicon['ADP'].isin(['NOUN', 'ADJ', 'ADV'])]

        # constrói dataframe
        df_lexicon = pd.DataFrame(columns=['words', 'infinitive', 'gender', 'number'])
        df_lexicon['words'] = lexicon['a']
        df_lexicon['infinitive'] = lexicon['a.1']
        df_lexicon['gender'] = lexicon['_'].str.split('Gender=', expand=True)[1].str.split('|', expand=True)[0]
        df_lexicon['number'] = lexicon['_'].str.split('Number=', expand=True)[1].str.split('|', expand=True)[0]

        # preenche gêneros e números vazios
        df_lexicon['gender'].fillna('NA', inplace=True)
        df_lexicon['number'].fillna('NA', inplace=True)

        return df_lexicon

    def get_replacement_word(df_selection, gender=None, number=None):
        if gender and number:
            replacement_word = df_selection[(abs(df_selection['hits']) > 0) & (df_selection['gender'] == gender) & (df_selection['number'] == number)]
        else:
            replacement_word = df_selection[(abs(df_selection['hits']) > 0)]
        return replacement_word[(replacement_word['hits'] == replacement_word['hits'].max())].max()['words']

    def generate_replacements_dict(type='number_gender'):
        # carrega dicionarios
        df_tep = ss.load_tep()
        df_corpop = ss.load_corpop()
        df_lexicon = ss.load_lexicon()

        # consolida tep e lexicon para captar informação de gênero e número
        df_tep['infinitive'] = df_tep['infinitive'].str.strip()
        df_lexicon['infinitive'] = df_lexicon['infinitive'].str.strip()
        df_tep_lexicon = df_tep.merge(df_lexicon, on=['infinitive'], how='left')
        df_tep_lexicon.dropna(subset=['words'], inplace=True)

        # consolida corpop e lexicon para captar informação de gênero e número
        df_corpop['words'] = df_corpop['words'].str.strip()
        df_lexicon['words'] = df_lexicon['words'].str.strip()
        df_corpop_lexicon = df_corpop.merge(df_lexicon[['words', 'infinitive']], on=['words'], how='left')
        df_corpop_lexicon.drop(columns=['words'], inplace=True)
        df_corpop_lexicon.dropna(inplace=True)
        df_corpop_lexicon.drop_duplicates(subset=['infinitive'], inplace=True)
        df_corpop_lexicon = df_lexicon.merge(df_corpop_lexicon, on=['infinitive'], how='left')
        df_corpop_lexicon.dropna(subset=['hits'], inplace=True)

        # consolida df_tep_lexicon e corpop para captar número de hits
        df_tep_lexicon['words'] = df_tep_lexicon['words'].str.strip()
        df_tep_hits = df_tep_lexicon.merge(df_corpop_lexicon, on=['infinitive', 'words', 'gender', 'number'], how='left')
        df_tep_hits.sort_values(by=['hits'], ascending=False, inplace=True)
        df_tep_hits.drop_duplicates(subset=['key', 'words'], inplace=True)
        df_tep_hits.reset_index(drop=True, inplace=True)

        # indica melhor substituição para cada palavra do dicionario de sinonimos com base nas palavras simples
        if type == "number_gender":
            for k in df_tep_hits['key'].unique():
                df_selection = df_tep_hits[df_tep_hits['key'] == k]

                if len(df_selection['infinitive'].unique())>1:
                        df_tep_hits.loc[(df_tep_hits['key'] == k) & (df_selection['gender'] == 'Fem') & (df_selection['number'] == 'Sing'), 'replace'] = ss.get_replacement_word(df_selection, 'Fem', 'Sing')
                        df_tep_hits.loc[(df_tep_hits['key'] == k) & (df_selection['gender'] == 'Masc') & (df_selection['number'] == 'Sing'), 'replace'] = ss.get_replacement_word(df_selection, 'Masc', 'Sing')
                        df_tep_hits.loc[(df_tep_hits['key'] == k) & (df_selection['gender'] == 'NA') & (df_selection['number'] == 'Sing'), 'replace'] = ss.get_replacement_word(df_selection, 'NA', 'Sing')

                        df_tep_hits.loc[(df_tep_hits['key'] == k) & (df_selection['gender'] == 'Fem') & (df_selection['number'] == 'Plur'), 'replace'] = ss.get_replacement_word(df_selection, 'Fem', 'Plur')
                        df_tep_hits.loc[(df_tep_hits['key'] == k) & (df_selection['gender'] == 'Masc') & (df_selection['number'] == 'Plur'), 'replace'] = ss.get_replacement_word(df_selection, 'Masc', 'Plur')
                        df_tep_hits.loc[(df_tep_hits['key'] == k) & (df_selection['gender'] == 'NA') & (df_selection['number'] == 'Plur'), 'replace'] = ss.get_replacement_word(df_selection, 'NA', 'Plur')

                        df_tep_hits.loc[(df_tep_hits['key'] == k) & (df_selection['gender'] == 'Fem') & (df_selection['number'] == 'NA'), 'replace'] = ss.get_replacement_word(df_selection, 'Fem', 'NA')
                        df_tep_hits.loc[(df_tep_hits['key'] == k) & (df_selection['gender'] == 'Masc') & (df_selection['number'] == 'NA'), 'replace'] = ss.get_replacement_word(df_selection, 'Masc', 'NA')
                        df_tep_hits.loc[(df_tep_hits['key'] == k) & (df_selection['gender'] == 'NA') & (df_selection['number'] == 'NA'), 'replace'] = ss.get_replacement_word(df_selection, 'NA', 'NA')

            # limpa dataframe com melhores indicações de substituições
            df_replacements = df_tep_hits.dropna(subset=['replace'])

            df_corpop_lexicon.rename(columns={"hits":"hits_replace", "words":"replace"}, inplace=True)
            df_corpop_lexicon.sort_values(by=['hits_replace'], ascending=False, inplace=True)
            df_corpop_lexicon.drop_duplicates(subset=['replace', 'gender', 'number'], inplace=True)

            df_replacements = df_replacements.merge(df_corpop_lexicon[['replace', 'gender', 'number', 'hits_replace']], on=['replace', 'gender', 'number'], how='left')        

            # df_replacements = df_tep_hits[['key', 'words', 'hits', 'gender', 'number']].merge(df_tep_hits[['key', 'words', 'hits', 'gender', 'number']], on=['key', 'gender', 'number'], how='left')

        else:
            df_replacements = df_tep_hits[['key', 'words', 'hits']].merge(df_tep_hits[['key', 'words', 'hits']], on=['key'], how='left')
            df_replacements.rename(columns={"words_x":"words", "words_y":"replace", "hits_x":"hits", "hits_y":"hits_replace"}, inplace=True)
            df_replacements.sort_values(by=['hits_replace'], ascending=False, inplace=True)
            df_replacements.drop_duplicates(subset=['words','replace'], inplace=True)
            df_replacements = df_replacements[df_replacements['hits_replace']>df_replacements['hits']]

        print(f"Number of words available for replacement: {len(df_replacements['words'].unique())}")

        print('Salvando dicionário de substituições em \\tools\\ss\\')
        df_replacements.to_parquet(os.getcwd()+f"\\tools\\ss\\context_{type}.parquet")
        df_replacements.to_excel(os.getcwd()+f"\\tools\\ss\\context_{type}.xlsx")

        return df_replacements

    def replace_words(original_text, sp, df_replacements, type="simple", nr_texts=5):
        # carrega corpus de substituições
        df_replacements['words'] = df_replacements['words'].str.upper()

        # converte o texto original para uma lista de sentenças
        sentences = original_text.splitlines()

        # processo de substituição de tokens quando existente em df_replacements
        output = [None]*nr_texts*nr_texts
        new_text = ""
        k = 0
        for sentence in sentences:
            # avaliação de token na senteça que sejam nomes próprios para não realizar substituição
            doc = sp(sentence)
            propn = [str(x) for x in list(doc) if x.pos_ == 'PROPN']
            ents = [str(x) for x in list(doc.ents)]
            ents = word_tokenize(" ".join(ents))
            ner = [x for x in ents if x in propn]
            for token in word_tokenize(sentence):
                if token not in ner:
                    # seleciona parte do token que seja texto
                    token_word_list = re.findall(r'[^\W]+', token)
                    token_word_list = token if token_word_list == [] else token_word_list
                    word_size = 0
                    ind = 0
                    for word in token_word_list:
                        if len(word) > word_size:
                            word_size = len(word)
                            ind = token_word_list.index(word)
                    token_word = token_word_list[ind]

                    # seleciona possibilidades de substituição para o token texto
                    df_selection = df_replacements.loc[df_replacements['words'].isin([token_word.upper()]),:]

                    if not df_selection.empty:  
                        # agrupa possíveis substituições por máximo de hits para todos os contextos
                        df_selection = df_selection.groupby(by=['replace']).max().reset_index()
                        num_replacements = len(df_selection['replace'].unique())
                        if num_replacements > 0:
                            # conta quantidade de substituições realizadas
                            k += 1

                            # aplica algoritmo da roleta
                            if type=="simple":
                                df_selection['probability'] = df_selection['hits_replace']/df_selection['hits_replace'].sum()
                                ascending=False
                            else:
                                df_selection['probability'] = df_selection['hits']/df_selection['hits'].sum() if df_selection['hits'].sum() != 0 else 0
                                ascending=True

                            for i in range(nr_texts):
                                if num_replacements > 1:
                                    selected_key = ss.random_selection(df_selection, ascending)
                                    selected_replace = df_selection.loc[selected_key, 'replace']
                                    replacement_word = token.replace(token_word, selected_replace.lower())
                                else:
                                    replacement_word = df_selection['replace'][0].lower()

                                try:
                                    if output[0] == None:
                                        new_text += replacement_word + " "
                                        output = [new_text]*nr_texts
                                    else:
                                        new_text = output[i]
                                        new_text += replacement_word + " "
                                        output[i] = new_text

                                except:
                                    if output[0] == None:
                                        new_text += token + " "
                                        output = [new_text]*nr_texts
                                    else:
                                        new_text = output[i]
                                        new_text += token + " "
                                        output[i] = new_text
                        else:
                            if output[0] == None:
                                new_text += token + " "
                                output = [new_text]*nr_texts
                            else:
                                for i, _ in enumerate(output):
                                    new_text = output[i]
                                    new_text += token + " "
                                    output[i] = new_text
                    else:
                        if output[0] == None:
                            new_text += token + " "
                            output = [new_text]*nr_texts
                        else:
                            for i, _ in enumerate(output):
                                new_text = output[i]
                                new_text += token + " "
                                output[i] = new_text
                else:
                    if output[0] == None:
                        new_text += token + " "
                        output = [new_text]*nr_texts
                    else:
                        for i, _ in enumerate(output):
                            new_text = output[i]
                            new_text += token + " "
                            output[i] = new_text
            # adiciona line break ao final de cada sentença
            if output[0] != None:
                for i, _ in enumerate(output):
                    output[i] = output[i] + "\n"
        
        print(f"total of replacements: {k}/{len(original_text.split())}")
        return output, df_replacements

    def replace_complex_words(original_text, sp, df_replacements, nr_texts=5):
        df_replacements.rename(columns={"words":"replace", "replace":"words"}, inplace=True)
        output, _ = ss.replace_words(original_text, sp, df_replacements, type="complex", nr_texts=nr_texts)
        df_replacements.rename(columns={"replace":"words", "words":"replace"}, inplace=True)
        return output, df_replacements

