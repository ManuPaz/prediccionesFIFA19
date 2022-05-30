import numpy as np
from sklearn.model_selection import train_test_split
from utils import load_config
import logging
config = load_config.config()
def entrenar_modelo(name, feature, modelo, df, columnas):
    """
    Metodo que divide datos en train y test y crea y entrena el objeto que encapsula los modelos
    :param name: nombre del modelo a entrenar
    :param feature: nombre de la variable a predecir
    :param modelo: clase para instanciar que encapsula al modelo: Clasificador o Regresor
    :param df: dataframe
    :param columnas: variables predictoras
    :return: objeto que encapsula al modelo, mX_train, X_test, y_train, y_test
    """

    X = df.loc[:, columnas]
    y = df.loc[:, feature]

    size = config["entrenamiento"]["train_test_split"]
    np.random.seed( config["entrenamiento"]["random_seed"])
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        train_size=float(size),
        random_state=1234,
        shuffle=True
    )

    modelo_encapsulado = modelo(name, feature)
    modelo_encapsulado.fit_predict(X_train, y_train, X_test, y_test)

    return  modelo_encapsulado,X_train, X_test, y_train, y_test

