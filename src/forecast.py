
import os
import pickle
import logging.config
import pandas as pd
import re
import warnings
pd.set_option('display.float_format', lambda x: '%.5f' % x)
warnings.filterwarnings('ignore')
from config import load_config
pd.set_option('display.max_rows',1000)
# el usuario introduce parte de un nombre jugador o equipo y una variable y se muestran las predicciones y el valor real para diferentes modelos (todos o solo el mejor modelo) para los jugadores seleccionados

if __name__=="__main__":
    logging.config.fileConfig('../logs/logging.conf')
    # create logger
    logger = logging.getLogger('forecasting')

    df = pd.read_csv("../data/preprocesed/dataFIFA.csv")
    df=df.loc[(df.Wage>0)&(df.Value>0)]
    config=load_config.config()
    todos_los_modelos= config["forecasting"]["todos_los_modelos"]
    if todos_los_modelos:
        dir="../assets/pruebasModelos/"
    else:
        dir = "../assets/modelosFinales/"

    nombres_files=os.listdir(dir)


    dicEntradas={1:"Wage",2:"Value",3:"PositionGrouped",4:"Position",5:"PositionSinLado"}
    while(1):
        nombre_jugador=input("Nombre del jugador o equipo:\n")
        rows=df.loc[(df.Name.str.contains(nombre_jugador, regex=True, na=True, flags=re.IGNORECASE))\
            |df.Club.str.contains(nombre_jugador, regex=True, na=True, flags=re.IGNORECASE)]

        rows.index=rows.Name
        if len(rows)>0:
            print(rows.loc[:,["Club","Nationality","PositionLongName","PositionGrouped"]])
            try:
                variable_a_predecir=int(input("1:Salario\n2:Valor de mercado\n"
                                              "3:Position agrupada en P,D,M,Del\n4.Posicion\n"
                                              "5.Posicion sin lado  "))
                variable_a_predecir= dicEntradas[variable_a_predecir]
            except Exception as e:
                logger.error("Seleccion de variable invalida")
                continue

            cols=[variable_a_predecir]
            for nombre_file in nombres_files:
                if nombre_file.split("_")[0]==variable_a_predecir:
                    with open(dir+nombre_file,"rb") as file:
                        modeloEncapsulado=pickle.load(file)

                        rows_x=rows.loc[:,modeloEncapsulado.variables]
                        rows_y=rows.loc[:,variable_a_predecir]

                        pred = modeloEncapsulado.predict(rows_x, rows_y)
                        rows[modeloEncapsulado.name]=pred
                        cols.append(modeloEncapsulado.name)


            print((rows.loc[:,cols]))
        else:
            logger.error("Seleccion de jugador o equipo invalido {}".format(nombre_jugador))

