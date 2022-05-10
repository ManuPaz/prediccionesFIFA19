from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, make_scorer
scorer = make_scorer(f1_score, average="macro")
def get_train_test(X, y, train_size =0.7, shuffle=False):
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        train_size=float(train_size),
        random_state=1234,
        shuffle=shuffle
    )
    return X_train, X_test, y_train, y_test


def parameter_search(gbrt, param_grid, X_train, y_train,n_iter=10,cv=10,tipo="random"):
    if tipo == "random":
        search = RandomizedSearchCV(gbrt, param_grid,n_iter=n_iter, cv=cv, scoring=scorer)
    else:
        search = GridSearchCV(gbrt, param_grid, cv=cv, scoring=scorer)
    search.fit(X_train, y_train)
    params = search.best_params_
    return params



