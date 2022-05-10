from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import GradientBoostingClassifier, AdaBoostClassifier, RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, make_scorer
from sklearn.svm import LinearSVC
from  functions import machine_learninge_utils
import numpy as np
from config import load_config
import logging.config

config = load_config.config()

logging.config.fileConfig('../logs/logging.conf')
logger = logging.getLogger('training')


scorer = make_scorer(f1_score, average="macro")

modelos={"decision_tree":DecisionTreeClassifier, "random_forest":RandomForestClassifier,
         "gradient_boosting": GradientBoostingClassifier, "ada_boosting": AdaBoostClassifier,
         "linear_SVC":LinearSVC, "logistic_regression":LogisticRegression,"k_neighbors":KNeighborsClassifier,
         "lda":LinearDiscriminantAnalysis}


class Classifier_HyperParameterTuning():

    def __init__(self, name):
        self.name = name
        self.param_grid = {} # param_grid para optimizacion de hiperparametros
        self.params = {}  # parametros seleccionados o parametros fijos para entrenar el modelo
        self.optimizar =  True

        #paramatros para pasar a random search o grid search
        self.cv = config["entrenamiento"]["cv"]
        self.n_iter = config["entrenamiento"]["random_search"]["n_iter"]
        self.tipo_busqueda=config["entrenamiento"]["search"]

        #cargar variables de config.yaml en diccionaraios params y param_grid
        if name in config["entrenamiento"]["param_tunning"]:
            for param in config["entrenamiento"]["param_tunning"][name]:
                l = config["entrenamiento"]["param_tunning"][name][param]
                if len(l)==3:
                    values = np.arange(l[0], l[1], l[2])
                else:
                    values=l
                self.param_grid[param] = values

        if name in config["entrenamiento"]["params"]:
            for param in  config["entrenamiento"]["params"][name]:
                l = config["entrenamiento"]["params"][name][param]
                self.params[param] = l

        #definir modelo
        self.modelo = modelos[name]


    def fit_predict(self,X_train, y_train, X_test, y_test):

        # si hay parametros para optimizar y se quiere optimizar se opmitizan con random search p con grid search
        if  self.optimizar and len(self.param_grid.keys())>0:

            parametros= machine_learninge_utils.parameter_search(self.modelo(), self.param_grid, X_train, y_train, cv=self.cv, n_iter=self.n_iter,tipo=self.tipo_busqueda)
            self.params.update(parametros)

        logger.info("Entrenamiento de modelo {} para variable {}"
                    "con parametros {}, busqueda de  parametros {} search".format(self.name,config["entrenamiento"]["feature"], str(self.params),self.tipo_busqueda))

        print(self.params)
        self.modelo = self.modelo(**self.params)
        self.modelo.fit(X_train, y_train)
        y_pred =  self.modelo.predict(X_test)
        return y_pred

    def metrics(self, y, y_pred):
        p = precision_score(y, y_pred, average="macro")
        r = recall_score(y, y_pred, average="macro")
        f1 = f1_score(y, y_pred, average="macro")
        ac = accuracy_score(y, y_pred)
        return p, r, f1, ac




