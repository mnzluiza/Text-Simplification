from datetime import datetime
import pandas as pd
import os

import re
from lxml import etree
from urllib.parse import unquote

from tools.utils import utils

print(f'PROCESSO DE ENRIQUECIMENTO DE DADOS POR WIKIBOOKS INICIALIZADO! - {datetime.now()}')

tree = etree.parse(f'{os.getcwd()}\\inputs\\crawling_by_readability\\wikibooks.vert', etree.HTMLParser(recover=True, encoding='utf-8'))
docs = tree.xpath('.//doc')
output = pd.DataFrame()
i = -1
for doc in docs:
    i+=1
    grade_level = str(doc.xpath(".//@docid"))
    output.loc[i, "Grade Level"] = grade_level
    output.loc[i, "Label"] = "Simples" if "fundamental" in grade_level else "Complexo"
    output.loc[i, "Corpus"] = "Wikibooks"
    output.loc[i, "Content"] = utils.transform_text("".join(doc.xpath(".//text()")).replace('\n\n\n\n\n', 'AAAAA').replace('\n\n.\n\n\n', '.AAAAA').replace('\n', ' ').replace('AAAAA', '\n'))

    try:
        uri = re.findall(r"'(.*)'", str(doc.xpath(".//@uri")))[0]
    except:
        uri = str(doc.xpath(".//@uri"))

    try:
        output.loc[i,"Title"] = unquote(re.findall(r"(.*)ContextTitle=(.*)%2FImprimir(.+)", uri)[0][1].encode('iso-8859-1', 'ignore').decode('utf8', 'ignore').replace("+", " ")).replace("\\xa0", "-")
    except:
        output.loc[i,"Title"] = re.findall(r"[\"'](.*)['\"]", unquote(str(doc.xpath(".//@title")).encode('iso-8859-1', 'ignore').decode('utf8', 'ignore')))[0]

output.reset_index(drop=True, inplace=True)
output['key'] = (output.index + 1000).astype(str)
output.to_parquet(os.getcwd() + '\\outputs\\bronze\\wikibooks.parquet')

print(f'PROCESSO DE ENRIQUECIMENTO DE DADOS POR WIKIBOOKS FINALIZADO! - {datetime.now()}')