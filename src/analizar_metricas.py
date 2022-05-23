import os
import pickle
import logging.config
import pandas as pd
import re
import warnings
import numpy as np

from config.load_config import config
import json
warnings.filterwarnings('ignore')
from config import load_config

pd.set_option('display.max_rows', 1000)
# el usuario introduce parte de un nombre jugador o equipo y una variable y se muestran las predicciones y el valor real para diferentes modelos (todos o solo el mejor modelo) para los jugadores seleccionados

if __name__ == "__main__":
    feature="PositionSinLado"
    todos_los_modelos=True
    modelo="SVR"
    metrica="accuracy"
    config = load_config.config()
    nombre_reportes = config["nombre_dir_reportes_cv"]
    with open(nombre_reportes+feature+".json","r") as file:
        if  todos_los_modelos:
            dic=json.load(file)
            for modelo in dic.keys():
                lista = dic[modelo]
                #lista=[e for e in lista if "kernel" in e[ "parametros_fijos_no_por_defecto"].keys() and  e["parametros_fijos_no_por_defecto"]["kernel"]=="rbf"]# and e["parametros_optimizados"]["C"]==1 and e["parametros_optimizados"]["degree"]==3]
                lista.sort(key=lambda t: t[metrica],reverse=True)
                print(modelo)
                print(lista[0])
                print("----")

        else:


            lista=json.load(file)[modelo]
            lista.sort(key= lambda t:t[metrica],reverse=True)
            print(lista[0])





