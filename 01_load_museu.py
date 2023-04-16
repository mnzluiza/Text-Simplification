import os
from tools.utils import read
from datetime import datetime

print(f'PROCESSO DE CONSOLIDAÇÃO DO CORPUS PRINCIPAL INICIALIZADO! - {datetime.now()}')

p1 = read.readAllTxtsInFolder("Original", input_path=os.getcwd()+"\\inputs\\museu\\")
p1['Label'] = "Complexo"
p2 = read.readAllTxtsInFolder("Simplificado", input_path=os.getcwd()+"\\inputs\\museu\\")
p2['Label'] = "Simples"
output = p1.append(p2).reset_index(drop=True)

output['key'] = output.index.astype(str)
output['Corpus'] = "Museu"
output.to_parquet(os.getcwd() + '\\outputs\\bronze\\museu.parquet')

print(f'PROCESSO DE CONSOLIDAÇÃO DO CORPUS PRINCIPAL CONCLUÍDO! - {datetime.now()}')