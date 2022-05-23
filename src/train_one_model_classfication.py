import pandas as pd
from config import load_config
from functions import obtener_variables_predictoras, entrenamiento
import numpy as np
from functions import machine_learning,machine_learninge_utils
import logging
import warnings
import pickle
from sklearn.decomposition import PCA
warnings.filterwarnings('ignore')
logging.config.fileConfig('../logs/logging.conf')
logger = logging.getLogger('training')
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
from utils import report_model_results
import os
if __name__ == '__main__':
    config = load_config.config()
    nombre_dir_modelos = config["nombre_dir_modelos"]

    df = pd.read_csv("../data/preprocesed/dataFIFA.csv",index_col=0)
    nombre_modelo="SVC"
    nombres = [
               "gradient_boosting", "ada_boosting",
                ]
    for nombre_modelo in nombres:
        feature =config["entrenamiento"]["feature_clasificacion"]



        columnas = obtener_variables_predictoras.obtenerVariablesPredictoras("todas")




        modelo_encapsulado, X_train, X_test, y_train, y_test = entrenamiento.ejecutarModelo(nombre_modelo, feature,
                                                                                            machine_learning.Clasificador,
                                                                                            df,

                                                                                        columnas)
        metricas=modelo_encapsulado.metrics()
        cm=modelo_encapsulado.confusion_matrix()
        print(metricas)

        modelo_encapsulado.predict(X_train, y_train)
        metricas_train = modelo_encapsulado.metrics()
        print(metricas)

        resultado = {}
        resultado["params_no_default"] = modelo_encapsulado.params
        resultado["metricas_validation"] = metricas
        resultado["metricas_train"] = metricas_train
        resultado["normalizacion_X"] = modelo_encapsulado.normalize_X
        resultado["reduccionDimensionalidad"] = modelo_encapsulado.dimension_reduction
        report_model_results.report_results(modelo_encapsulado.feature, modelo_encapsulado.name, resultado, cv=False)

        if not os.path.isdir(nombre_dir_modelos):
            os.makedirs(nombre_dir_modelos)
        with open(nombre_dir_modelos + feature + "_" + nombre_modelo, 'wb') as handle:
            pickle.dump(modelo_encapsulado, handle, protocol=pickle.HIGHEST_PROTOCOL)


















# See PyCharm help at https://www.jetbrains.com/help/pycharm/