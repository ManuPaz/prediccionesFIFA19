import os
import pickle
import logging.config
import pandas as pd
import re
import warnings
import numpy as np

pd.set_option('display.float_format', lambda x: '%.5f' % x)
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
warnings.filterwarnings('ignore')
from config import load_config

pd.set_option('display.max_rows', 1000)
# el usuario introduce parte de un nombre jugador o equipo y una variable y se muestran las predicciones y el valor real para diferentes modelos (todos o solo el mejor modelo) para los jugadores seleccionados

if __name__ == "__main__":
    logging.config.fileConfig('../logs/logging.conf')
    # create logger
    logger = logging.getLogger('forecasting')

    df = pd.read_csv("../data/preprocesed/dataFIFA.csv")
    df = df.loc[(df.Wage > 0) & (df.Value > 0)]
    config = load_config.config()
    todos_los_modelos = config["forecasting"]["todos_los_modelos"]


    if todos_los_modelos:
        dir =  config["nombre_dir_modelos"]
    else:
        dir = config["nombre_dir_mejores_modelos"]

    nombres_files = os.listdir(dir)

    dicEntradas = {1: "Wage", 2: "Value", 3: "PositionGrouped", 4: "Position", 5: "PositionSinLado"}
    cambiarJugadores = True

    while (1):
        if cambiarJugadores:
            nombre_jugador = input("Nombre del jugador o equipo:\n")
            rows = df.loc[(df.Name.str.contains(nombre_jugador, regex=True, na=True, flags=re.IGNORECASE)) \
                          | df.Club.str.contains(nombre_jugador, regex=True, na=True, flags=re.IGNORECASE)]

            rows.index = rows.Name

        if len(rows) > 0:
            print(rows.loc[:, ["Club", "Nationality", "PositionLongName", "PositionGrouped"]])
            try:
                variable_a_predecir = int(input("1:Salario\n2:Valor de mercado\n"
                                                "3:Position agrupada en P,D,M,Del\n4.Posicion\n"
                                                "5.Posicion sin lado  "))
                variable_a_predecir = dicEntradas[variable_a_predecir]
            except Exception as e:
                logger.error("Seleccion de variable invalida")
                continue

            cols = [variable_a_predecir]
            for nombre_file in nombres_files:
                if nombre_file.split("_")[0] == variable_a_predecir:
                    with open(dir + nombre_file, "rb") as file:
                        modeloEncapsulado = pickle.load(file)


                        jugadores_train = modeloEncapsulado.jugadores_train

                        rows_x = rows.loc[:, modeloEncapsulado.variables]
                        rows_y = rows.loc[:, variable_a_predecir]
                        #añadimos una columna para saber si el jugador estaba en train o test
                        rows["test_or_train"] = rows.Name.transform(lambda x: "train" if x in jugadores_train else "test")


                        pred = modeloEncapsulado.predict(rows_x, rows_y)
                        rows[modeloEncapsulado.name] = pred
                        cols.append(modeloEncapsulado.name)
                        cols.append("test_or_train")


            print((rows.loc[:, cols]))
        else:
            logger.error("Seleccion de jugador o equipo invalido {}".format(nombre_jugador))
        cambiarJugadores = True if (input("Cambiar jugadores:\n1. Si\n,2. No\n")) == "1" else False
