import numpy as np
from sklearn.model_selection import train_test_split
from utils import load_config
from functions import machine_learning
config = load_config.config()
from typing import Type

def entrenar_modelo(name, feature, modelo: Type[machine_learning.HyperParameterTuning], df, columnas):
    """
    Metodo que divide datos en train y test y crea y entrena el objeto que encapsula los modelos

    :param name: nombre del modelo a entrenar
    :type name: str
    :param feature: nombre de la variable a predecir
    :type feature: str
    :param modelo: clase que extiende  HyperParameterTuning: Clasificador o Regresor

    :param df: dataframe
    :type df: pandas.DataFrame
    :param columnas: variables predictoras
    :type columnas: []
    :return: objeto que encapsula al modelo, X_train, X_test, y_train, y_test
    :rtype: machine_learning.HyperParameterTuning
    """

    X = df.loc[:, columnas]
    y = df.loc[:, feature]

    size = config["entrenamiento"]["train_test_split"]
    np.random.seed(config["entrenamiento"]["random_seed"])
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        train_size=float(size),
        random_state=1234,
        shuffle=True
    )

    modelo_encapsulado = modelo(name, feature)
    modelo_encapsulado.fit_predict(X_train, y_train, X_test, y_test)

    return modelo_encapsulado, X_train, X_test, y_train, y_test
