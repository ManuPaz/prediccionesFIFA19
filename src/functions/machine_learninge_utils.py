from sklearn.model_selection import RandomizedSearchCV, GridSearchCV

import numpy as np
from functions import machine_learning
from utils import report_model_results




def parameter_search(modelo_encapsulado, param_grid, X_train, y_train, scorer, n_iter=10, cv=10, tipo="random"):
    """
    Optmizacion de parametros
    :param modelo_encapsulado: objeto que encapsula modelo
    :type: HyperParameterTuning
    :param param_grid: espacio de parametros
    :type: dict
    :param X_train:
    :type: pandas dataframe
    :param y_train:
    :type: pandas dataframe o pandas series
    :param scorer: scorer con la metrica que se quiere optimizar
    :type: callable
    :param n_iter: número de iteraciones (necesario solo para random search)
    :type: int
    :param cv: número de grupos de cross validation
    :type: int
    :param tipo: random o grid
    :type: str
    :return: parametros seleccionados
    :type: dict
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

        resultado = {"parametros_optimizados": combinacion,
                     "parametros_fijos_no_por_defecto":modelo_encapsulado.params,
                     "normalizacion_X":modelo_encapsulado.normalize_X,
                     "reduccionDimensionalidad":modelo_encapsulado.dimension_reduction}

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


def smape(real, fitted, transformation=None):
    """

    :param real: valores reales de y
    :type: pandas series o numpy array
    :param fitted: predicciondes de y
    :type: pandas series o numpy array
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
