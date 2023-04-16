import os
import pandas as pd

class read:
    def readFile(filename, charenc=""): 
        import chardet
        
        rawdata = open(filename, 'rb').read()
        result = chardet.detect(rawdata)
        charenc = result['encoding'] if charenc=="" else charenc
        with open(filename, "r", encoding=charenc) as f:
            rich_text = f.readlines()
            content = []
            for line in rich_text:
                try:
                    if len(line) > 0:
                        content.append(str(line).replace('\xa0', ''))
                except:
                    print(f"Linha não lida do {filename}: {line}")
            f.close()
        return content
    
    def readAllTxtsInFolder(name_identifier="", input_path=os.getcwd()+"\\inputs\\", charenc="utf-8"):
        new_texts = pd.DataFrame()
        i = 0
        for filename in os.listdir(input_path):
            if ".txt" in filename or ".tsv" in filename:
                if name_identifier == "" or name_identifier in filename:
                    new_texts.loc[i, 'Title'] = filename
                    content = read.readFile(input_path+filename, charenc)
                    new_texts.loc[i, 'Content'] = "".join(content) if content else ""
                    i += 1
        return new_texts

    def readXSVFile(filename, sep=""):
        if sep == "":
            x = filename[-3:]
        if x == 'csv':
            return pd.read_csv(filename, sep=',')
        elif x == 'tsv':
            return pd.read_csv(filename, sep='\t')
        else:
            return pd.read_csv(filename, sep=sep)

class write:
    def str2txtFile(filename, content):
        if content:
            f = open(filename, 'w', encoding="utf-8")
            f.write(content)
            f.close()

class utils:
    def clean_text(
        string: str, 
        stop_words = [],
        punctuations=r'''!()[]{};:'"\,<>./?@#$%^&*_~''') -> str:
        """
        A method to clean text 
        """
        import re

        if string:
            # Cleaning the urls
            string = re.sub(r'https?://\S+|www\.\S+', '', string)

            # Cleaning the html elements
            string = re.sub(r'<.*?>', '', string)

            # Removing the punctuations
            for x in string.lower(): 
                if x in punctuations: 
                    string = string.replace(x, "") 

            # Converting the text to lower
            string = string.lower()

            # Removing stop words
            if stop_words != []:
                string = ' '.join([word for word in string.split() if word not in stop_words])

            # Cleaning the whitespaces
            string = re.sub(r'\s+', ' ', string).strip()

            return string   

    def transform_text(input_text):
        text = str(input_text)
        text = text.strip()
        text = text.replace("  ", " ")
        text = text.replace(" . ", ". ")
        text = text.replace(" , ", ", ")
        text = text.replace(" ( ", " (")
        text = text.replace(" ) ", ") ")
        text = text.replace(" [ ", " [")
        text = text.replace(" ] ", "] ")
        text = text.replace(" ; ", "; ")
        text = text.replace(" ? ", "? ")    
        text = text.replace(" ! ", "! ")
        
        text = text.replace("\\r\\n", "{{enter}}")
        text = text.replace("\\n", "{{enter}}")
        text = text.replace("\\\"", "\"")
        text = text.replace("è", "e")
        text = text.replace("ì", "i")
        text = text.replace("ò", "o")
        text = text.replace("ù", "u")

        text = text.replace(" à ", "{{crase}}")
        text = text.replace("à", "a")
        text = text.replace("{{crase}}", " à ")
        text = text.replace(" À ", "{{CRASE}}")
        text = text.replace("À", "A")
        text = text.replace("{{CRASE}}", " À ")

        text = text.replace("``", "\"")
        text = text.replace("''", "\"")
        text = text.replace("`", "\"")
        text = text.replace("´", "\"")
        text = text.replace("º", "o")
        text = text.replace("ª", "a")
        text = text.replace("ĉ", "c")
        text = text.replace("ý", "y")
        text = text.replace("\\", "/")
        text = text.replace("♪", "")
        text = text.replace("ă", "ã")
        text = text.replace("ò", "o")
        text = text.replace("Ò", "O")
        text = text.replace("å", "a")
        text = text.replace("ř", "r")
        text = text.replace("ő", "o")
        text = text.replace("Û", "U")
        text = text.replace("û", "u")
        text = text.replace("ẽ", "e")

        return text

    def apply_PCA(df, variacao=0.9, n_components=0):
        from sklearn.decomposition import PCA 
        import numpy as np
        
        # Definição de componentes PCA conforme percentual de variação definido
        pca = PCA()
        pca.fit(df)
        
        if n_components!=0:
            n = n_components
        else:
            variancia = np.cumsum(pca.explained_variance_ratio_)
            n = np.argmax(variancia >= variacao)

        # PCA considerando qtde de componentes encontrada
        pca = PCA(n_components = n+1)
        pca.fit(df)
        return pd.DataFrame(pca.transform(df), index = df.index)

class supervised_grouping:
    def classify(self, config):
        input_filename = config['input_filename'] if 'input_filename' in config else 'full_gold'
        output_name = config['output_name']
        output_detail = config['output_detail']
        clf_list = config['clf_list'].split(',')
        validation_type = config['validation_type']
        corpus_selection_list = config['corpus_selection_list'].split(',')
        augmentation_label = config['augmentation_label'].split(',') if 'augmentation_label' in config else None
        SL_selection = config['SL_selection'].split(',') if 'SL_selection' in config else None
        type_metrics_cols = config['type_metrics_cols'] if 'type_metrics_cols' in config else None
        type_embeds_cols = config['type_embeds_cols'] if 'type_embeds_cols' in config else None
        training_selection_list = config['training_selection_list'].split(',') if 'training_selection_list' in config else None
        testing_selection_list = config['testing_selection_list'].split(',') if 'testing_selection_list' in config else None
        n_components_embeddings = int(config['n_components_embeddings']) if 'n_components_embeddings' in config else None

        # Carrega dados
        gold = pd.read_parquet(f"{os.getcwd()}\\outputs\\gold\\{input_filename}.parquet")
        attr_columns = ['key', 'Title', 'Corpus', 'Label', 'Content', 'target']
        
        # Seleção de textos para análise
        paired_gold = gold[gold['Corpus'].isin(corpus_selection_list)]
        # Seleciona apenas uma classe específica para aumento, quando aplicável
        if augmentation_label:
            paired_gold = pd.DataFrame()
            for idx, label in enumerate(augmentation_label):
                if label == 'all':
                    aux_paired_gold = gold[gold['Corpus'].str.match(corpus_selection_list[idx])]
                else:
                    aux_paired_gold = gold[(gold['Corpus'].str.match(corpus_selection_list[idx]) & gold['Label'].str.contains(label))]
                paired_gold = paired_gold.append(aux_paired_gold).reset_index(drop=True)
        paired_gold.reset_index(drop=True, inplace=True)
        paired_gold['target'] = 0
        paired_gold.loc[paired_gold['Label'].str.contains('Simples'), 'target'] = 1

        # Inicia seleção de métricas para teste
        metrics_cols = [x for x in gold.columns if (x not in attr_columns) and ('BERTimbau' not in x) and ('FB_CBOW' not in x) and ('NILC_SG' not in x)]
        ## Por seleção de métricas de embedding
        embeds_cols = [x for x in gold.columns if (x not in attr_columns) and (x not in metrics_cols)]
        if type_embeds_cols:
            if type_embeds_cols == "none":
                embeds_cols = []
            elif type_embeds_cols == "fb_cbow":
                embeds_cols = [x for x in embeds_cols if 'FB_CBOW' in x]
            elif type_embeds_cols == "nilc_sg":
                embeds_cols = [x for x in embeds_cols if 'NILC_SG' in x]
            elif type_embeds_cols == "bertimbau":
                embeds_cols = [x for x in embeds_cols if 'BERTimbau' in x]
                if n_components_embeddings:
                    df_pca = utils.apply_PCA(paired_gold[embeds_cols], n_components=n_components_embeddings)
                    df_pca.columns = ['bert_pca_'+str(x[0]) for x in enumerate(df_pca.columns)]
                    paired_gold = paired_gold.merge(df_pca, left_index=True, right_index=True, how='left')
                    embeds_cols = [x for x in df_pca.columns if 'bert_pca_' in x]
            elif type_embeds_cols == "bertimbau_nilc":
                embeds_cols = [x for x in embeds_cols if 'BERTimbau' in x or 'FB_CBOW' in x]
            elif type_embeds_cols != "all":
                raise Exception('type_embeds_cols inválido')

        ## Por seleção de métricas de leiturabilidade
        if type_metrics_cols:
            if type_metrics_cols == "none":
                metrics_cols = []
            elif type_metrics_cols == "teste":
                metrics_cols = ['lsa_adj_mean', 'lsa_adj_std', 'lsa_all_mean', 'lsa_all_std', 'lsa_paragraph_mean', 'lsa_paragraph_std']
                # metrics_cols = ['cross_entropy', 'concretude_1_25_ratio', 'concretude_25_4_ratio', 'concretude_4_55_ratio', 'concretude_55_7_ratio', 'concretude_mean', 'concretude_std', 'familiaridade_1_25_ratio', 'familiaridade_25_4_ratio', 'familiaridade_4_55_ratio', 'familiaridade_55_7_ratio', 'familiaridade_mean', 'familiaridade_std', 'idade_aquisicao_1_25_ratio', 'idade_aquisicao_25_4_ratio', 'idade_aquisicao_4_55_ratio', 'idade_aquisicao_55_7_ratio', 'idade_aquisicao_mean', 'idade_aquisicao_std', 'imageabilidade_1_25_ratio', 'imageabilidade_25_4_ratio', 'imageabilidade_4_55_ratio', 'imageabilidade_55_7_ratio', 'imageabilidade_mean', 'imageabilidade_std', 'content_density', 'punctuation_diversity', 'aux_plus_PCP_per_sentence', 'participle_verbs', 'verbal_time_moods_diversity', 'words_before_main_verb', 'adjunct_per_clause', 'clauses_per_sentence', 'non_svo_ratio', 'passive_ratio', 'sentences_with_one_clause', 'infinitive_verbs', 'punctuation_ratio', 'ratio_function_to_content_words', 'hypernyms_verbs', 'content_words_ambiguity', 'cw_freq', 'cw_freq_bra', 'cw_freq_brwac', 'freq_bra', 'freq_brwac', 'min_cw_freq', 'min_cw_freq_bra', 'min_cw_freq_brwac', 'min_freq_bra', 'min_freq_brwac', 'dalechall_adapted']
                # metrics_cols = ['cross_entropy', 'concretude_1_25_ratio', 'concretude_55_7_ratio', 'familiaridade_1_25_ratio', 'familiaridade_55_7_ratio', 'familiaridade_mean', 'familiaridade_std', 'content_density', 'punctuation_diversity', 'and_ratio', 'aux_plus_PCP_per_sentence', 'participle_verbs', 'verbal_time_moods_diversity', 'words_before_main_verb', 'adjunct_per_clause', 'clauses_per_sentence', 'non_svo_ratio', 'passive_ratio', 'sentences_with_one_clause', 'infinitive_verbs', 'punctuation_ratio', 'ratio_function_to_content_words', 'hypernyms_verbs', 'content_words_ambiguity', 'cw_freq', 'cw_freq_bra', 'cw_freq_brwac', 'freq_bra', 'freq_brwac', 'min_cw_freq', 'min_cw_freq_bra', 'min_cw_freq_brwac', 'min_freq_bra', 'min_freq_brwac', 'dalechall_adapted']
            elif type_metrics_cols == "ltb_fs_dt":
                metrics_cols = ['brunet', 'dalechall_adapted', 'flesch', 'gunning_fox', 'honore', 'cw_freq', 'min_cw_freq', 'words', 'dep_distance', 'gerund_verbs', 'ratio_coordinate_conjunctions', 'arg_ovl', 'pronoun_diversity', 'negative_words']
            elif type_metrics_cols == "dt_no_fuzzy":
                metrics_cols = ['idade_aquisicao_mean', 'sentences_with_five_clauses', 'honore', 'min_cw_freq_bra', 'abstract_nouns_ratio', 'cross_entropy', 'lsa_adj_mean', 'concretude_55_7_ratio', 'imageabilidade_4_55_ratio', 'function_word_diversity', 'sentences_with_four_clauses', 'pronoun_ratio', 'clauses_per_sentence', 'anaphoric_refs', 'nouns_ambiguity', 'imageabilidade_55_7_ratio', 'freq_bra', 'max_noun_phrase', 'cw_freq_bra', 'sentences_with_two_clauses', 'named_entity_ratio_sentence', 'adjunct_per_clause', 'verb_diversity']
            elif type_metrics_cols == "fs_var":
                metrics_cols = ['adjacent_refs', 'second_person_pronouns', 'lsa_givenness_std', 'subjunctive_imperfect_ratio', 'adverbs_standard_deviation', 'indicative_imperfect_ratio']
            elif type_metrics_cols == "fs":
                metrics_cols = ['idade_aquisicao_mean', 'output_fb_284', 'indicative_imperfect_ratio', 'output_fb_202', 'anaphoric_refs', 'output_bert_729']
            elif type_metrics_cols == "decision_tree_variance":
                metrics_cols = ['lsa_adj_std', 'adjectives_standard_deviation', 'adj_cw_ovl', 'min_cw_freq', 'subjunctive_imperfect_ratio', 'output_bert_579', 'second_person_pronouns', 'sentences_per_paragraph', 'demonstrative_pronoun_ratio', 'cross_entropy', 'arg_ovl', 'lsa_paragraph_std', 'sentence_length_standard_deviation', 'nouns_standard_deviation', 'output_bert_509', 'content_word_standard_deviation', 'indicative_preterite_perfect_ratio', 'output_bert_100', 'lsa_span_std', 'dialog_pronoun_ratio', 'verbs_standard_deviation', 'third_person_pronouns', 'output_bert_23', 'lsa_givenness_std', 'oblique_pronouns_ratio', 'verbs_max', 'stem_ovl', 'lsa_all_std']
            elif type_metrics_cols == "no_fuzzy":
                metrics_cols = ["adj_arg_ovl", "adj_cw_ovl", "adj_stem_ovl", "arg_ovl", "stem_ovl", "cross_entropy", "lsa_adj_mean", "lsa_all_mean", "lsa_givenness_mean", "lsa_paragraph_mean", "lsa_span_mean", "adjacent_refs", "anaphoric_refs", "adjunct_per_clause", "adverbs_before_main_verb_ratio", "apposition_per_clause", "clauses_per_sentence", "coordinate_conjunctions_per_clauses", "dep_distance", "frazier", "infinite_subordinate_clauses", "non_svo_ratio", "passive_ratio", "postponed_subject_ratio", "ratio_coordinate_conjunctions", "ratio_subordinate_conjunctions", "relative_clauses", "sentences_with_five_clauses", "sentences_with_four_clauses", "sentences_with_one_clause", "sentences_with_seven_more_clauses", "sentences_with_six_clauses", "sentences_with_three_clauses", "sentences_with_two_clauses", "sentences_with_zero_clause", "subordinate_clauses", "temporal_adjunct_ratio", "words_before_main_verb", "yngve", "add_neg_conn_ratio", "add_pos_conn_ratio", "and_ratio", "cau_neg_conn_ratio", "cau_pos_conn_ratio", "conn_ratio", "if_ratio", "log_neg_conn_ratio", "log_pos_conn_ratio", "logic_operators", "negation_ratio", "or_ratio", "gerund_verbs", "max_noun_phrase", "mean_noun_phrase", "min_noun_phrase", "adjective_diversity_ratio", "content_density", "content_word_diversity", "function_word_diversity", "indefinite_pronouns_diversity", "noun_diversity", "preposition_diversity", "pronoun_diversity", "punctuation_diversity", "relative_pronouns_diversity_ratio", "ttr", "verb_diversity", "cw_freq", "cw_freq_bra", "cw_freq_brwac", "freq_bra", "freq_brwac", "min_cw_freq", "min_cw_freq_bra", "min_cw_freq_brwac", "min_freq_bra", "min_freq_brwac", "adjective_ratio", "adjectives_max", "adjectives_min", "adverbs_diversity_ratio", "content_words", "first_person_possessive_pronouns", "first_person_pronouns", "function_words", "indefinite_pronoun_ratio", "indicative_condition_ratio", "indicative_future_ratio", "infinitive_verbs", "noun_ratio", "nouns_max", "nouns_min", "oblique_pronouns_ratio", "personal_pronouns", "prepositions_per_clause", "prepositions_per_sentence", "pronoun_ratio", "pronouns_max", "pronouns_min", "punctuation_ratio", "ratio_function_to_content_words", "relative_pronouns_ratio", "second_person_pronouns", "third_person_possessive_pronouns", "verbs", "abstract_nouns_ratio", "adjectives_ambiguity", "adverbs_ambiguity", "content_words_ambiguity", "hypernyms_verbs", "named_entity_ratio_sentence", "nouns_ambiguity", "verbs_ambiguity", "verbal_time_moods_diversity", "aux_plus_PCP_per_sentence", "indicative_imperfect_ratio", "indicative_pluperfect_ratio", "indicative_present_ratio", "participle_verbs", "subjunctive_future_ratio", "subjunctive_imperfect_ratio", "subjunctive_present_ratio", "tmp_neg_conn_ratio", "tmp_pos_conn_ratio", "paragraphs", "sentence_length_max", "sentences", "subtitles", "syllables_per_content_word", "words", "words_per_sentence", "concretude_1_25_ratio", "concretude_25_4_ratio", "concretude_4_55_ratio", "concretude_55_7_ratio", "concretude_mean", "familiaridade_1_25_ratio", "familiaridade_25_4_ratio", "familiaridade_4_55_ratio", "familiaridade_55_7_ratio", "familiaridade_mean", "idade_aquisicao_1_25_ratio", "idade_aquisicao_25_4_ratio", "idade_aquisicao_4_55_ratio", "idade_aquisicao_55_7_ratio", "idade_aquisicao_mean", "imageabilidade_1_25_ratio", "imageabilidade_25_4_ratio", "imageabilidade_4_55_ratio", "imageabilidade_55_7_ratio", "imageabilidade_mean", "sentence_length_min", "dialog_pronoun_ratio", "easy_conjunctions_ratio", "hard_conjunctions_ratio", "long_sentence_ratio", "medium_long_sentence_ratio", "medium_short_sentence_ratio", "short_sentence_ratio", "simple_word_ratio", "coreference_pronoun_ratio", "demonstrative_pronoun_ratio", "brunet", "dalechall_adapted", "flesch", "gunning_fox", "honore"]
            elif type_metrics_cols == "leiturabilidade":
                metrics_cols = ['brunet', 'dalechall_adapted', 'flesch', 'gunning_fox', 'honore']
            elif type_metrics_cols == "feature_selection":
                metrics_cols = ['cw_freq', 'min_cw_freq', 'honore', 'words', 'flesch', 'dep_distance']
            elif type_metrics_cols == "decision_tree":
                metrics_cols = ['flesch', 'gerund_verbs', 'ratio_coordinate_conjunctions', 'arg_ovl', 'pronoun_diversity', 'negative_words']
            elif type_metrics_cols != "all":
                raise Exception('type_metrics_cols inválido')
        
        ## Define relação de métricas e embeddings que serão usados para classificação
        value_columns = metrics_cols + embeds_cols

        # Realiza separação de textos para treino ou teste conforme o validation_type
        if validation_type == "loo":
            unique_keys_selection = paired_gold['key'].str.split('_', n=0, expand=True)[0].unique()
            X_fit = []
            y_fit = []
            X_eval = []
            y_eval = []
            df_eval = paired_gold[(paired_gold['key'].isin(unique_keys_selection))]
            for key in unique_keys_selection:
                df_train = paired_gold[~(paired_gold['Title'].str.contains(max(paired_gold.loc[paired_gold['key']==key,'Title'])[:-16]))]
                X_fit.append(df_train[value_columns])
                y_fit.append(df_train['target'])
                aux_eval = df_eval[(df_eval['key'] == str(key))]
                X_eval.append(aux_eval[value_columns])
                y_eval.append(aux_eval['target'])   
        elif validation_type == "random":
            from sklearn.model_selection import train_test_split
            X_fit, X_eval, y_fit, y_eval = train_test_split(paired_gold[value_columns], paired_gold['target'], test_size=0.15, random_state=1)
            df_eval = paired_gold[paired_gold.isin(y_eval.index)]
            X_fit = [X_fit]
        elif validation_type == "selection":
            if training_selection_list and testing_selection_list:
                df_fit = paired_gold[paired_gold['Corpus'].isin(training_selection_list)]
                X_fit = [df_fit[value_columns]]
                y_fit = df_fit['target']
                df_eval = paired_gold[paired_gold['Corpus'].isin(testing_selection_list)]
                X_eval = df_eval[value_columns]
                y_eval = df_eval['target']
            else:
                raise Exception('para validation_type do tipo selection é necessário informar training_selection_list e testing_selection_list por args')
        else:
                raise Exception('validation_type inválido')

        # Inicializa classificador conforme o clf_type
        for clf_type in clf_list:
            args={}
            if clf_type == "logistic_regression":
                args['random_state'] = 0
                args['max_iter'] = 400
                from sklearn.linear_model import LogisticRegression
                clf = LogisticRegression(**args)
            elif clf_type == "svm":
                args['probability'] = True
                from sklearn import svm
                clf = svm.SVC(**args)
            elif clf_type == "xgb":
                args['random_state'] = 0
                import xgboost as xgb
                clf = xgb.XGBClassifier(**args)
            else:
                print("CLASSIFICADOR INVÁLIDO! O PROCESSO ESTÁ SENDO ENCERRADO ...")
                return None
            
            # Realiza classificação para o conjunto especificado
            output = pd.DataFrame()
            indx = 0
            for _X_fit in X_fit:
                if len(X_fit) > 1:
                    _X_eval = X_eval[indx]
                    _y_fit = y_fit[indx]
                    _y_eval = y_eval[indx]
                else:
                    _X_eval = X_eval
                    _y_fit = y_fit
                    _y_eval = y_eval

                clf.fit(_X_fit, _y_fit)
                final_df = _X_eval.merge(df_eval[attr_columns], left_index=True, right_index=True, how="left")
                final_df['Prediction'] = pd.Series(clf.predict(_X_eval), index=final_df.index)
                final_df['Label'] = _y_eval

                try:
                    final_df['Complex_Probability'] = pd.Series(clf.predict_proba(_X_eval)[:,0]*100, index=final_df.index)
                    final_df['Simple_Probability'] = pd.Series(clf.predict_proba(_X_eval)[:,1]*100, index=final_df.index)
                    final_df = final_df[['key', 'Title', 'Label', 'Prediction', 'Complex_Probability', 'Simple_Probability']]
                except:
                    final_df = final_df[['key', 'Title', 'Label', 'Prediction']]
                final_df['Explanation'] = output_detail
                final_df['Classifier'] = clf_type
                final_df['Metrics_cols'] = type_metrics_cols
                final_df['Embeds_cols'] = type_embeds_cols
                final_df['Validation_type'] = validation_type
                final_df['Texts_selection'] = ', '.join(corpus_selection_list)
                output = pd.concat([final_df, output])
                indx += 1
            
            output.key = output.key.astype(str)
            output.to_parquet(f"{os.getcwd()}\\outputs\\others\\classifiers\\parquet\\{output_name}_{clf_type}.parquet")
            output.to_excel(f"{os.getcwd()}\\outputs\\others\\classifiers\\xlsx\\{output_name}_{clf_type}.xlsx")
