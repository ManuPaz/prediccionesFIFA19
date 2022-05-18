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


if __name__ == '__main__':
    config = load_config.config()
    df = pd.read_csv("../data/preprocesed/dataFIFA.csv",index_col=0)
    nombre_modelo="SVR"
    feature =config["entrenamiento"]["feature_regresion"]

    df=df.loc[(df[feature] >1000) & (df[feature]<=5000)]
    print(len(df), len(df.loc[df.Wage == 0]), len(df.loc[df.Value == 0]))


    columnas = obtener_variables_predictoras.obtenerVariablesPredictoras("todas")



    modelo_encapsulado,X_train, X_test, y_train, y_test = entrenamiento.ejecutarModelo(nombre_modelo, feature,
                                                      machine_learning.Regresor, df,
                                                      columnas)
    print( modelo_encapsulado.metrics())
    modelo_encapsulado.predict(X_train,y_train)
    print(modelo_encapsulado.metrics())

    with open("../assets/modelosFinales/" + feature + "_" + nombre_modelo, 'wb') as handle:
        pickle.dump(modelo_encapsulado, handle, protocol=pickle.HIGHEST_PROTOCOL)

















# See PyCharm help at https://www.jetbrains.com/help/pycharm/