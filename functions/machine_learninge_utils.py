from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, make_scorer
import numpy as np

def get_train_test(X, y, train_size =0.7, shuffle=False):
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        train_size=float(train_size),
        random_state=1234,
        shuffle=shuffle
    )
    return X_train, X_test, y_train, y_test


def parameter_search(modelo, param_grid, X_train, y_train,scorer,n_iter=10,cv=10,tipo="random"):
    """

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
    if tipo == "random":
        search = RandomizedSearchCV(modelo, param_grid,n_iter=n_iter, cv=cv, scoring=scorer)
    else:
        search = GridSearchCV(modelo, param_grid, cv=cv, scoring=scorer)
    search.fit(X_train, y_train)
    params = search.best_params_
    return params

def SMAPE(real, fitted):
    return np.mean(abs(real-fitted)*2/(abs(real)+abs(fitted)))



