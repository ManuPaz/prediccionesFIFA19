import pandas as pd
import os
os.chdir("../../")
from functions import obtener_variables_predictoras, entrenamiento
from functions import machine_learning
import logging
import warnings
import pickle
from utils import report_model_results, load_config

warnings.filterwarnings('ignore')
logging.config.fileConfig('logs/logging.conf')
logger = logging.getLogger('training')

if __name__ == '__main__':
    config = load_config.config()
    nombre_dir_modelos = config["nombre_dir_modelos"]
    df = pd.read_csv("data/preprocesed/dataFIFA.csv", index_col=0)
    nombre_modelo = "ridge"
    MODELOS = ["ridge", "lasso", "linear",
              "k_neighbors", "random_forest",
              "gradient_boosting","linear_SVR",
              "elastic_net","SVR"]
    for nombre_modelo in MODELOS:
        feature = config["entrenamiento"]["feature_regresion"]
        df = df.loc[(df[feature] > 0)]
        if feature == "Wage":
            df = df.loc[(df[feature] != 1000)]
        print(len(df), len(df.loc[df.Wage == 0]), len(df.loc[df.Value == 0]))

        columnas = obtener_variables_predictoras.obtenerVariablesPredictoras("todas")

        modelo_encapsulado, X_train, X_test, y_train, y_test = entrenamiento.entrenar_modelo(nombre_modelo, feature,
                                                                                             machine_learning.Regresor,
                                                                                             df,
                                                                                             columnas)
        metricas = modelo_encapsulado.metrics()
        print(metricas)
        modelo_encapsulado.predict(X_train, y_train)
        metricas_train = modelo_encapsulado.metrics()
        print(metricas_train)

        resultado = {"params_no_default": modelo_encapsulado.params,
                    "metricas_validation": metricas,
                    "metricas_train":metricas_train,
                    "normalizacion_X": modelo_encapsulado.normalize_X,
                    "reduccionDimensionalidad": modelo_encapsulado.dimension_reduction,
                    "transformacion_log_y": modelo_encapsulado.function_transform,
                    "normalizacion_y": modelo_encapsulado.normalize_y
                     }
        report_model_results.report_results(modelo_encapsulado.feature, modelo_encapsulado.name, resultado, cv=False)

        if not os.path.isdir(nombre_dir_modelos):
            os.makedirs(nombre_dir_modelos)
        with open(nombre_dir_modelos + feature + "_" + nombre_modelo, 'wb') as handle:
            pickle.dump(modelo_encapsulado, handle, protocol=pickle.HIGHEST_PROTOCOL)

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
