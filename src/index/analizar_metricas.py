import os
os.chdir("../../")
import pandas as pd
import warnings
import json

warnings.filterwarnings('ignore')
from utils import load_config

pd.set_option('display.max_rows', 1000)


if __name__ == "__main__":
    feature = "Wage"
    todos_los_modelos = True
    modelo = "SVR"
    metrica = "SMAPE"
    config = load_config.config()
    nombre_reportes = config["nombre_dir_reportes_cv"]
    with open(nombre_reportes + feature + ".json", "r") as file:
        if todos_los_modelos:
            dic = json.load(file)
            for modelo in dic.keys():
                lista = dic[modelo]
                lista.sort(key=lambda t: t[metrica], reverse=False)
                print(modelo)
                print(lista[0])
                print("----")

        else:

            lista = json.load(file)[modelo]
            lista.sort(key=lambda t: t[metrica], reverse=False)
            print(lista[0])
