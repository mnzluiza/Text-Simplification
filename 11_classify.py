import codecs
from configparser import ConfigParser
from tools.utils import supervised_grouping
from datetime import datetime
import os

CONFIG_FILE = f'{os.getcwd()}\\settings\\classify_config.ini'

def read_config():
    config_object = ConfigParser()
    config_object.read_file(codecs.open(CONFIG_FILE, "r", "utf8"))
    return config_object

sg = supervised_grouping()
config = read_config()

################################
##        EXPERIMENTOS        ##
################################
for item in config:
    if item != "DEFAULT":
        print(f"*** PROCESSO DE CLASSIFICAÇÃO {config[item]['output_name']} INICIALIZADO! - {datetime.now()} ***")
        sg.classify(config[item])
        print(f"--- PROCESSO ENCERRADO! - {datetime.now()} ---")
