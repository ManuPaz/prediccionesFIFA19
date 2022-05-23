import pandas as pd
from config import load_config
from functions import obtener_variables_predictoras, entrenamiento
import numpy as np
from functions import machine_learning,machine_learninge_utils
import logging
import warnings
import pickle
import os
from utils import report_model_results
warnings.filterwarnings('ignore')
logging.config.fileConfig('../logs/logging.conf')
logger = logging.getLogger('training')


if __name__ == '__main__':
    config = load_config.config()
    nombre_dir_modelos=config["nombre_dir_modelos"]
    df = pd.read_csv("../data/preprocesed/dataFIFA.csv",index_col=0)
    nombre_modelo="linear"
    feature =config["entrenamiento"]["feature_regresion"]
    df = df.loc[(df[feature] >0)]
    if feature=="Wage":
        df=df.loc[(df[feature] !=1000)]
    print(len(df), len(df.loc[df.Wage == 0]), len(df.loc[df.Value == 0]))


    columnas = obtener_variables_predictoras.obtenerVariablesPredictoras("todas")



    modelo_encapsulado,X_train, X_test, y_train, y_test = entrenamiento.ejecutarModelo(nombre_modelo, feature,
                                                      machine_learning.Regresor, df,
                                                      columnas)
    metricas=modelo_encapsulado.metrics()
    print( metricas)
    modelo_encapsulado.predict(X_train, y_train)
    metricas_train = modelo_encapsulado.metrics()
    print(metricas_train)

    resultado = {}
    resultado["params_no_default"]=modelo_encapsulado.params
    resultado["metricas_validation"]=metricas
    resultado["metricas_train"] =metricas_train
    resultado["normalizacion_X"] = modelo_encapsulado.normalize_X
    resultado["reduccionDimensionalidad"] = modelo_encapsulado.dimension_reduction
    resultado["transformacion_log_y"] = modelo_encapsulado.function_transform
    resultado["normalizacion_y"] = modelo_encapsulado.normalize_y
    report_model_results.report_results(modelo_encapsulado.feature, modelo_encapsulado.name, resultado, cv=False)

    if not os.path.isdir(nombre_dir_modelos):

        os.makedirs(nombre_dir_modelos)
    with open(nombre_dir_modelos + feature + "_" + nombre_modelo, 'wb') as handle:
        pickle.dump(modelo_encapsulado, handle, protocol=pickle.HIGHEST_PROTOCOL)

















# See PyCharm help at https://www.jetbrains.com/help/pycharm/