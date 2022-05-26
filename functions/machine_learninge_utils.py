from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, make_scorer
import numpy as np
from functions import machine_learning
from utils import report_model_results


def get_train_test(X, y, train_size=0.7, shuffle=False):
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        train_size=float(train_size),
        random_state=1234,
        shuffle=shuffle
    )
    return X_train, X_test, y_train, y_test


def parameter_search(modelo_encapsulado, param_grid, X_train, y_train, scorer, n_iter=10, cv=10, tipo="random"):
    """
    Optmizacion de parametros
    :param modelo: modelo de sklearn
    :param param_grid: espacio de parametros
    :param X_train:
    :param y_train:
    :param scorer: scorer con la metrica que se quiere optimizar
    :param n_iter: número de iteraciones (necesario solo para random search)
    :param cv: número de grupos de cross validation
    :param tipo: random o grid
    :return: parametros seleccionados
    """
    modelo = modelo_encapsulado.modelo()
    if tipo == "random":
        search = RandomizedSearchCV(modelo, param_grid, n_iter=n_iter, cv=cv, scoring=scorer)
    else:
        search = GridSearchCV(modelo, param_grid, cv=cv, scoring=scorer)
    search.fit(X_train, y_train)
    params = search.best_params_

    # guardar los resultados de cross validation en un archivo
    for i, combinacion in enumerate(search.cv_results_["params"]):

        resultado = {}

        resultado["parametros_optimizados"] = combinacion
        resultado["parametros_fijos_no_por_defecto"] = modelo_encapsulado.params
        resultado["normalizacion_X"] = modelo_encapsulado.normalize_X
        resultado["reduccionDimensionalidad"] = modelo_encapsulado.dimension_reduction

        if isinstance(modelo_encapsulado, machine_learning.Regresor):
            resultado["SMAPE"] = -search.cv_results_["mean_test_score"][i]
            resultado["transformacion_log_y"] = modelo_encapsulado.function_transform
            resultado["normalizacion_y"] = modelo_encapsulado.normalize_y
        else:
            resultado["accuracy"] = search.cv_results_["mean_test_score"][i]
            resultado["metodo_metricas_multiclass"] = modelo_encapsulado.compute_metrics
            resultado["numero_de_clases"] = len(search.classes_)
        resultado["tipo_optimizacion"] = modelo_encapsulado.tipo_busqueda

        report_model_results.report_results(modelo_encapsulado.feature, modelo_encapsulado.name, resultado, cv=True)

    return params


def SMAPE(real, fitted, transformation=None):
    """

    :param real:
    :param fitted:
    :param transformation: para cuando se llame a este  metodo con el make_scorer en el random search
        si random_search trabaja con datos transformados  aqui se hace la inversa para usar la metrica sobre los originales
        vale para cualquier otro caso en que no se quieran mandar los datos ya transformados a los originales
    :return:
    """
    if transformation is not None:
        real = transformation.pipeline_y.inverse_transform(real.reshape(-1, 1))
        fitted = transformation.pipeline_y.inverse_transform(fitted.reshape(-1, 1))
    if isinstance(real, np.ndarray):
        real = real.reshape(-1)
    if isinstance(fitted, np.ndarray):
        fitted = fitted.reshape(-1)
    return np.mean(abs(real - fitted) * 2 / (abs(real) + abs(fitted)))
