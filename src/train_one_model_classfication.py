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

if __name__ == '__main__':
    config = load_config.config()
    df = pd.read_csv("../data/preprocesed/dataFIFA.csv")
    nombre_modelo="random_forest"
    feature =config["entrenamiento"]["feature_clasificacion"]



    columnas = obtener_variables_predictoras.obtenerVariablesPredictoras("todas")




    modelo_encapsulado, X_train, X_test, y_train, y_test = entrenamiento.ejecutarModelo(nombre_modelo, feature,
                                                                                        machine_learning.Clasificador,
                                                                                        df,

                                                                                    columnas)
    metricas=modelo_encapsulado.metrics()
    cm=modelo_encapsulado.confusion_matrix()
    print(metricas)
    print(cm)
    modelo_encapsulado.predict(X_train, y_train)
    metricas = modelo_encapsulado.metrics()
    print(metricas )



    prob=modelo_encapsulado.predict_probabilities(X_test)

    with open("../assets/modelosFinales/" + feature + "_" + nombre_modelo, 'wb') as handle:
        pickle.dump(modelo_encapsulado, handle, protocol=pickle.HIGHEST_PROTOCOL)

















# See PyCharm help at https://www.jetbrains.com/help/pycharm/