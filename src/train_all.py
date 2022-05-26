import os.path
import sys

import pandas as pd
from config import load_config
from functions import obtener_variables_predictoras, entrenamiento
import numpy as np
from functions import machine_learning
import logging
import pickle
import warnings

warnings.filterwarnings('ignore')
logging.config.fileConfig('../logs/logging.conf')

if __name__ == '__main__':
    logger = logging.getLogger('training')
    config = load_config.config()
    nombre_dir_modelos = config["nombre_dir_modelos_pruebas"]

    df = pd.read_csv("../data/preprocesed/dataFIFA.csv", index_col=0)

    features_tipo = {
        # "Wage": machine_learning.Regresor,
        # "Value": machine_learning.Regresor,
        "PositionGrouped": machine_learning.Clasificador,
        "PositionSinLado": machine_learning.Clasificador,
        "Position": machine_learning.Clasificador
    }

    # entrenamiento para cada variable de las 3 con todos sus modelos (clasificacin o regresion)
    for feature, tipo_modelo in features_tipo.items():

        if tipo_modelo == machine_learning.Clasificador:
            array_nombres_modelo = config["entrenamiento"]["classification"]["modelos"]

        else:
            array_nombres_modelo = config["entrenamiento"]["regression"]["modelos"]

        for nombre_modelo in array_nombres_modelo:

            # definir conjuntos de variables para entrenar
            grupos_variables = [
                obtener_variables_predictoras.obtenerVariablesPredictoras("todas"),
                # obtener_variables_predictoras.obtenerVariablesPredictoras(feature),
            ]
            # entrenamiento con todas las variables o solo las seleccionadas
            for variables, info_variables in zip(grupos_variables, ["todas_las_variables", "variables_seleccionadas"]):

                # obtenemos el objeto encapsulando el modelo entrenado, las metricas y las predicciones
                if tipo_modelo == machine_learning.Regresor:
                    # si es de regression eliminamos los jugadores con valor de ese feature=0 porque esto no depende de
                    # las caracteristicas.  Ocurre para los jugadores que no tienen equipo
                    df = df.loc[df[feature] > 0]
                    if feature == "Wage":
                        df = df.loc[df.Wage != 1000]

                modelo_encapsulado, X_train, X_test, y_train, y_test = entrenamiento.ejecutarModelo(
                    nombre_modelo, feature, tipo_modelo, df,
                    variables)

                # log de las metricas
                logger.info("Prediccion para {} con modelo {} con {}: {} ".format(feature, nombre_modelo,
                                                                                  info_variables,
                                                                                  modelo_encapsulado.metrics()))
                # guardamos el modelo
                if not os.path.isdir(nombre_dir_modelos):
                    os.makedirs(nombre_dir_modelos)
                with open(nombre_dir_modelos + feature + "_" + nombre_modelo, 'wb') as handle:
                    pickle.dump(modelo_encapsulado, handle, protocol=pickle.HIGHEST_PROTOCOL)

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
