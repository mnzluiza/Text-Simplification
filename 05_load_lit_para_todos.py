import os
from tools.utils import read
from datetime import datetime

print(f'PROCESSO DE CONSOLIDAÇÃO DO CORPUS LIT PARA TODOS INICIALIZADO! - {datetime.now()}')

output = read.readAllTxtsInFolder(input_path=os.getcwd()+"\\inputs\\lit_para_todos\\", charenc="")
output['key'] = (output.index + 2000).astype(str)
output['Corpus'] = "LitParaTodos"
output['Label'] = "Simples"
output.to_parquet(os.getcwd() + '\\outputs\\bronze\\lit_para_todos.parquet')

print(f'PROCESSO DE CONSOLIDAÇÃO DO CORPUS LIT PARA TODOS CONCLUÍDO! - {datetime.now()}')