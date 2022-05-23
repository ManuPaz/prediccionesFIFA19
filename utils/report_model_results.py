import os
from config import load_config
from functions import machine_learning
import json
config=load_config.config()
nombre_reportes=config["nombre_dir_reportes_cv"]
nombre_reportes_finales=config["nombre_dir_reportes_finales"]
for nombre in [nombre_reportes,nombre_reportes_finales]:
    if not os.path.isdir(nombre):
        os.makedirs(nombre)




def report_results(feature, nombre_modelo, resultados:dict, cv=True):
    if cv==True:
        nombre_dir = nombre_reportes
    else:
        nombre_dir = nombre_reportes_finales

    if  os.path.isfile(nombre_dir + feature + ".json"):
       with  open(nombre_dir+feature+".json","r") as file:
        dic=json.load(file)
    else:
        dic={}
    if cv:
        if nombre_modelo in dic.keys():
            dic[nombre_modelo].append(resultados)
        else:
            dic[nombre_modelo]=[resultados]
    else:
        dic[nombre_modelo] = resultados



    with  open(nombre_dir + feature + ".json", "w") as file:
        json.dump(dic,file)

