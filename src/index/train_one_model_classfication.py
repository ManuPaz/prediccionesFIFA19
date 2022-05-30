import pandas as pd
import os
os.chdir("../../")
from functions import obtener_variables_predictoras, entrenamiento
from functions import machine_learning
import logging
import warnings
import pickle
warnings.filterwarnings('ignore')
logging.config.fileConfig('logs/logging.conf')
logger = logging.getLogger('training')
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
from utils import report_model_results, load_config


if __name__ == '__main__':
    config = load_config.config()
    nombre_dir_modelos = config["nombre_dir_modelos"]

    df = pd.read_csv("data/preprocesed/dataFIFA.csv", index_col=0)
    nombre_modelo = "SVC"
    nombres = [ "random_forest",
               "gradient_boosting", "ada_boosting",
               "linear_SVC", "logistic",
               "k_neighbors", "lda", "SVC" ]
    for nombre_modelo in nombres:
        feature = config["entrenamiento"]["feature_clasificacion"]

        columnas = obtener_variables_predictoras.obtenerVariablesPredictoras("todas")

        modelo_encapsulado, X_train, X_test, y_train, y_test = entrenamiento.entrenar_modelo(nombre_modelo, feature,
                                                                                             machine_learning.Clasificador,
                                                                                             df,

                                                                                             columnas)
        metricas = modelo_encapsulado.metrics()
        cm = modelo_encapsulado.confusion_matrix()
        print(metricas)

        modelo_encapsulado.predict(X_train, y_train)
        probs=modelo_encapsulado.predict(X_test, y_test)
        metricas_train = modelo_encapsulado.metrics()
        print(metricas)
        resultado = {"params_no_default": modelo_encapsulado.params,
                     "metricas_validation": metricas,
                     "metricas_train": metricas_train,
                     "normalizacion_X": modelo_encapsulado.normalize_X,
                     "reduccionDimensionalidad": modelo_encapsulado.dimension_reduction}

        report_model_results.report_results(modelo_encapsulado.feature, modelo_encapsulado.name, resultado, cv=False)

        if not os.path.isdir(nombre_dir_modelos):
            os.makedirs(nombre_dir_modelos)
        with open(nombre_dir_modelos + feature + "_" + nombre_modelo, 'wb') as handle:
            pickle.dump(modelo_encapsulado, handle, protocol=pickle.HIGHEST_PROTOCOL)

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
