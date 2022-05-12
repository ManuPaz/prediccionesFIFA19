import numpy as np

from functions import obtener_variables_predictoras
from config import load_config
import logging

from functions import machine_learninge_utils, machine_learning
config = load_config.config()
logging.config.fileConfig('../logs/logging.conf')
logger = logging.getLogger('training')
def ejecutarModelo(name, feature,Modelo, df,columnas, transformacion=None):
    """

    :param name: nombre del modelo a entrenar
    :param feature: nombre de la variable a predecir
    :param Modelo: clase para instanciar que encapsula al modelo: Clasificador o Regresor
    :param df: dataframe
    :param columnas: variables predictoras
    :param transformacion: funcion que se quiera aplicar a la variable a predecir (log o None normalmente)
    :return: objeto que encapsula al modelo, metricas, predicciones y probabilidades para cada clase en el clasificador
    (None para el regresor)
    """
    X = df.loc[:, columnas]
    y = df.loc[:, feature]
    if transformacion is not None:
        y = transformacion(y)
    size = config["entrenamiento"]["train_test_split"]
    X_train, X_test, y_train, y_test = machine_learninge_utils.get_train_test(X, y, train_size=size, shuffle=True)

    transform = None
    if Modelo == machine_learning.Regresor and transformacion is not None:
        transform=np.exp
        modeloEncapsulado = Modelo(name, feature,transform)
    else:
        modeloEncapsulado = Modelo(name, feature)
    modeloEncapsulado.fit_predict(X_train, y_train, X_test, y_test)

    #si es clasificadaor obtenemos tambien las probabilidades de cada clase para cada dato
    prediction_prob=None
    if Modelo==machine_learning.Clasificador:
        try:
            prediction_prob = modeloEncapsulado.predict_probabilities(X_test)
        except Exception as e:
            logger.error(e)


    metricas = modeloEncapsulado.metrics()
    return  modeloEncapsulado,metricas, modeloEncapsulado.y_pred, prediction_prob

