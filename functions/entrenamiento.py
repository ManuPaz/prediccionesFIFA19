import numpy as np

from functions import obtener_variables_predictoras
from config import load_config
import logging
from functions import machine_learninge_utils, machine_learning
config = load_config.config()
logging.config.fileConfig('../logs/logging.conf')
logger = logging.getLogger('training')
def ejecutarModelo(name, feature,Modelo, df,columnas):
    """
    Metodo que divide datos en train y test y crea y entrena el objeto que encapsula los modelos
    :param name: nombre del modelo a entrenar
    :param feature: nombre de la variable a predecir
    :param Modelo: clase para instanciar que encapsula al modelo: Clasificador o Regresor
    :param df: dataframe
    :param columnas: variables predictoras
    :return: objeto que encapsula al modelo, mX_train, X_test, y_train, y_test
    """
    X = df.loc[:, columnas]
    y = df.loc[:, feature]

    size = config["entrenamiento"]["train_test_split"]
    np.random.seed( config["entrenamiento"]["random_seed"])
    X_train, X_test, y_train, y_test = machine_learninge_utils.get_train_test(X, y, train_size=size, shuffle=True)

    modeloEncapsulado = Modelo(name, feature)
    modeloEncapsulado.fit_predict(X_train, y_train, X_test, y_test)

    return  modeloEncapsulado,X_train, X_test, y_train, y_test

