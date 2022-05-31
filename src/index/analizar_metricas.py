import os
os.chdir("../../")
import pandas as pd
import warnings
import json

warnings.filterwarnings('ignore')
from utils import load_config

pd.set_option('display.max_rows', 1000)
buscar_mejores_parametros=False
plotear_variacion_con_marametros=False

if __name__ == "__main__":
    feature = "PositionGrouped"
    todos_los_modelos = True
    modelo = "SVR"
    metrica = "SMAPE"
    config = load_config.config()
    nombre_reportes = config["nombre_dir_reportes_cv"]

    if buscar_mejores_parametros:

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

    else:

        nombre_dir_tablas = config["nombre_dir_tablas"]
        if not os.path.isdir(nombre_dir_tablas):
            os.makedirs(nombre_dir_tablas)

        nombre_reportes = config["nombre_dir_reportes_finales"]
        regresion=feature in config["entrenamiento"]["features_regresion"]
        if regresion :
            metricas=["SMAPE","MAPE"]

        else:
            metricas = ["accuracy"]
        columnas=["modelo"]
        for metrica in metricas:
            columnas.append(metrica+ " en train set")
            columnas.append(metrica + " en validation set")
        dataframe_resultados = pd.DataFrame(columns=columnas)
        with open(nombre_reportes + feature + ".json", "r") as file:
            dic = json.load(file)
            for e,value in dic.items():
                if regresion:
                    dataframe_resultados.loc[len(dataframe_resultados)]=[e,round(value["metricas_train"]["SMAPE"],3),
                                                        round(value["metricas_validation"]["SMAPE"],3),
                                                        round(value["metricas_train"]["MAPE"],3),
                                                        round(value["metricas_validation"]["MAPE"],3)
                                                        ]
                else:
                    dataframe_resultados.loc[len(dataframe_resultados)] = [e, round(value["metricas_train"]["accuracy"],3),
                                                                           round(value["metricas_validation"]["accuracy"],3)

                                                                           ]

        dataframe_resultados.to_csv(nombre_dir_tablas+feature+".csv",index=False)