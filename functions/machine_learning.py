import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import GradientBoostingClassifier, AdaBoostClassifier, RandomForestClassifier,\
    RandomForestRegressor,GradientBoostingRegressor
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, make_scorer, mean_squared_error, \
    mean_absolute_percentage_error, confusion_matrix
from sklearn.svm import LinearSVC,LinearSVR,SVR,SVC
from sklearn.linear_model import RidgeCV, LassoCV, LinearRegression, ElasticNetCV,LogisticRegression
from functions import machine_learninge_utils
import numpy as np
from utils import funcionesMatematicas
from config import load_config
import logging.config
from sklearn import preprocessing
from sklearn.pipeline import Pipeline
config = load_config.config()

logging.config.fileConfig('../logs/logging.conf')
logger = logging.getLogger('training')

scorer = make_scorer(f1_score, average="macro")

modelosClasificacion = {"decision_tree": DecisionTreeClassifier, "random_forest": RandomForestClassifier,
                        "gradient_boosting": GradientBoostingClassifier, "ada_boosting": AdaBoostClassifier,
                        "linear_SVC": LinearSVC, "logistic": LogisticRegression,
                        "k_neighbors": KNeighborsClassifier,"SVC":SVC,
                        "lda": LinearDiscriminantAnalysis}
modelosRegresion = {"ridge": RidgeCV, "lasso": LassoCV, "linear": LinearRegression,
                    "k_neighbors": KNeighborsRegressor,"random_forest":RandomForestRegressor,
                    "gradient_boosting": GradientBoostingRegressor,"linear_SVR":LinearSVR,
                    "logistic":LogisticRegression, "elastic_net":ElasticNetCV,"SVR":SVR}


class HyperParameterTuning():

    @property
    def name(self):
        """

        :return: nombre del modelo
        """
        return self.__name
    @property
    def feature(self):
        """

        :return: variable que se quiere predecir
        """
        return self.__feature
    @property
    def modelo(self):
        """

        :return: modelo de sklearn
        """
        return self.__modelo
    @property
    def param_grid(self):
        """

        :return: espacio de parametros para optimizar
        """
        return self.__param_grid
    @property
    def params(self):
        """

        :return: parametros del modelo
        """
        return self.__params
    @property
    def optimizar(self):
        """

        :return: si se quiere optimizar los parametros o no
        """
        return self.__optimizar
    @property
    def cv(self):
        """

        :return: grupos para el cross validation
        """
        return self.__cv
    @property
    def n_iter(self):
        """

        :return: numero de iteraciones si se hace random search
        """
        return self.__n_iter
    @property
    def tipo_busqueda(self):
        """

        :return: busqueda para la optimizacion de parametros: random o grid
        """
        return self.__tipo_busqueda
    @property
    def variables(self):
        """

        :return: variables predictoras del modelo
        """
        return self.__variables

    @property
    def scorer(self):
        """

        :return: scorer para la metrica a optimizar en random search o grid search si se hacen: f1-score, accuracy, mse ..
        """
        return self.__scorer
    @property
    def X_train(self):
        return self.__X_train
    @property
    def X_test(self):
        return self.__X_test

    @property
    def y_train(self):
        return self.__y_train

    @property
    def y_test(self):
        return self.__y_test

    @property
    def y_pred(self):
        return self.__y_pred

    @property
    def normalize_X(self):
        """

        :return: si se normaliza la X o no
        """
        return self.__normalize_X

    @property
    def dimension_reduction(self):
        """

        :return: si se reduce las dimensiones o no
        """
        return self.__dimension_reduction


    @property
    def pipeline(self):
        """

        :return: pipeline para transformar la X
        """
        return self.__pipeline
    @property
    def usar_transform_pipeline(self):
        """

        :return: si se usa alguna transformacion o no
        """
        return self.__usar_transform_pipeline

    @property
    def jugadores_train(self):
        return self.__jugadores_train
    @modelo.setter
    def modelo(self,modelo):
        self.__modelo=modelo

    @y_test.setter
    def y_test(self, y_test):
        self.__y_test=y_test

    @X_test.setter
    def X_test(self, X_test):
        self.__X_test = X_test

    @y_pred.setter
    def y_pred(self, y_pred):
        self.__y_pred = y_pred

    @scorer.setter
    def scorer(self,scorer):
            self.__scorer = scorer

    def __init__(self, name, feature,tipo):
        """

        :param tipo: regresor o clasificacion
        :param name: nombre del modelo. En funcion del nombre se coge el modelo de sklearn
        :param feature: variable a predecir
        """
        self.__name = name
        self.__feature = feature
        self.__param_grid = {}  # param_grid para optimizacion de hiperparametros
        self.__params = {}  # parametros seleccionados o parametros fijos para entrenar el modelo
        self.__optimizar = config["entrenamiento"]["optimizar"]

        # paramatros para pasar a random search o grid search
        self.__cv = config["entrenamiento"]["cv"]
        self.__n_iter = config["entrenamiento"]["random_search"]["n_iter"]
        self.__tipo_busqueda = config["entrenamiento"]["search"]
        self.__normalize_X = config["entrenamiento"]["normalize_X"]
        self.__dimension_reduction=config["entrenamiento"]["reducir_dimensionalidad"]
        self.__usar_transform_pipeline=False
        arrayPipeline=[]
        if self.__normalize_X:
            self.__usar_transform_pipeline = True
            if config["entrenamiento"]["scale_X"]:
                arrayPipeline.append(("scaler",preprocessing.MinMaxScaler()))


            else:
                arrayPipeline.append(("scaler", preprocessing.StandardScaler()))

        if self.__dimension_reduction:
            self.__usar_transform_pipeline = True
            arrayPipeline.append(("pca", PCA(n_components=6)))

        if self.__usar_transform_pipeline:
            self.__pipeline = Pipeline(arrayPipeline)





        # cargar variables de config.yaml en diccionaraios params y param_grid
        if name in config["entrenamiento"][tipo]["param_tunning"]:
            for param in config["entrenamiento"][tipo]["param_tunning"][name]:
                l = config["entrenamiento"][tipo]["param_tunning"][name][param]
                if len(l) == 3:
                    try:
                        values =np.arange(l[0], l[1], l[2]).tolist()
                    except Exception as e:
                        values=l

                else:

                    values = l
                self.__param_grid[param] = values

        if name in config["entrenamiento"][tipo]["params"]:
            for param in config["entrenamiento"][tipo]["params"][name]:
                l = config["entrenamiento"][tipo]["params"][name][param]
                self.__params[param] = l
                if l == "None":
                    self.__params[param]=None





    def __mensaje_log(self):
        log_text="Entrenamiento de modelo {} para variable {}".format(self.name, self.feature)
        if len(self.params)>0:
            log_text+=" con parametros {}".format(str(self.params))
        if self.optimizar and len(self.param_grid.keys()) > 0:
            log_text += " y optimizacion con tipo de busqueda {}".format(self.tipo_busqueda)
        logger.info(log_text)

    def predict(self,X_test=None,y_test=None):
        """
        Si X_test o y_test son None se utilizan los que ya están guardados guardados en atributos
        :param X_test:
        :param y_test:
        :return: las predicciones realizadas
        """
        if X_test is not None:
            self.__X_test = X_test.copy()
            if self.__usar_transform_pipeline:
                self.__X_test = self.__pipeline.transform(self.__X_test)


        if y_test is not None:
            self.__y_test = y_test.copy()
        self.__y_pred = self.__modelo.predict(self.__X_test)


        return  self.__y_pred

    def __fit_predict(self):
        # si hay parametros para optimizar y se quiere optimizar se opmitizan con random search p con grid search
        if self.__optimizar and len(self.__param_grid.keys()) > 0:
            parametros = machine_learninge_utils.parameter_search(self, self.__param_grid, self.__X_train, self.__y_train,
                                                                  cv=self.__cv, n_iter=self.__n_iter,
                                                                  tipo=self.__tipo_busqueda, scorer=self.__scorer)
            self.__params.update(parametros)

        self.__mensaje_log()

        self.__modelo = self.__modelo(**self.__params)

        self.__modelo.fit(self.__X_train, self.__y_train)
        self.__y_pred = self.__modelo.predict(self.__X_test)


    def fit_predict(self, X_train, y_train, X_test, y_test):
        """
        Entrena el modelo haciendo optimizacion de parametros antes o no dependiendo de la configuracion
        Predice para los datos de test.
        Hace las transformaciones sobre X que se determinen en la configuracion
        Guarda X_train, X_test preprocesados y y_train e y_test
        :param X_train:
        :param y_train:
        :param X_test:
        :param y_test:
        """

        self.__variables=X_train.columns
        self.__X_train=X_train.copy()
        self.__X_test = X_test.copy()
        self.__y_train= y_train.copy()
        self.__y_test = y_test.copy()

        #guardamos los nombres de los jugadores del train set
        self.__jugadores_train= self.__X_train.index

        if self.__usar_transform_pipeline:
            self.__X_train = self.__pipeline.fit_transform(self.__X_train)
            self.__X_test = self.__pipeline.transform(self.X_test)
        self.__fit_predict()




class Clasificador(HyperParameterTuning):

    @property
    def compute_metrics(self):
        """

        :return: forma de computar las metricas para multiclase: macro, micro o average
        """
        return self.__compute_metrics

    @compute_metrics.setter
    def compute_metrics(self, compute_metrics):
        self.__compute_metrics = compute_metrics

    def __init__(self, name, feature):
        super().__init__(name,feature,"classification")
        self.modelo = modelosClasificacion[name]
        self.__compute_metrics = config["entrenamiento"]["classification"]["multi_class_score"]
        self.scorer = make_scorer(accuracy_score,greater_is_better=True)


    def predict_probabilities(self,  X_test):
        """
        funcion que devuelve la probabilidade para cada clase en cada prediccion
        :param X_test:
        :return: propbabilidades de cada clase para cada prediccion
        """
        self.X_test=X_test
        if self.usar_transform_pipeline:
            self.X_test = self.pipeline.transform(self.X_test)



        y_proba = self.modelo.predict_proba(self.X_test)
        return y_proba

    def metrics(self):
        p = precision_score(self.y_test, self.y_pred, average=self.__compute_metrics)
        r = recall_score(self.y_test, self.y_pred, average=self.__compute_metrics)
        f1 = f1_score(self.y_test, self.y_pred, average=self.__compute_metrics)
        ac = accuracy_score(self.y_test, self.y_pred)

        return {"precission": p, "recall": r, "f1_score": f1, "accuracy": ac}
    def confusion_matrix(self):
        cm = confusion_matrix(self.y_test, self.y_pred, labels=self.modelo.classes_)
        cm = pd.DataFrame(cm, columns=self.modelo.classes_, index=self.modelo.classes_)
        return cm


class Regresor(HyperParameterTuning):

    @property
    def function_transform(self):
        """

        :return: si se transforma la y con una funcion o no (logaritmica por ejemplo)
        """
        return self.__function_transform

    @property
    def normalize_y(self):
        """

        :return: si se normaliza la y o no
        """
        return self.__normalize_y
    @property
    def pipeline_y(self):
        """

        :return: pipeline to transform y
        """
        return self.__pipeline_y



    def __init__(self, name, feature):
        """

        :param name:
        :param feature variable a predecir
        """
        super().__init__(name,feature,"regression")
        self.modelo = modelosRegresion[name]

        self.__normalize_y = config["entrenamiento"]["regression"]["normalize_y"]
        self.__function_transform = config["entrenamiento"]["regression"]["transformacion_logaritmica"]

        if self.__function_transform:
            self.scorer = make_scorer(machine_learninge_utils.SMAPE, greater_is_better=False,transformation=self)
        else:
            self.scorer = make_scorer(machine_learninge_utils.SMAPE, greater_is_better=False)

        self.__usar_pipeline_y=False
        arrayPipeline=[]
        if self.__normalize_y:
            self.__usar_pipeline_y=True
            arrayPipeline.append(("scaler",preprocessing.StandardScaler()))
        if self.__function_transform :
            self.__usar_pipeline_y = True
            arrayPipeline.append(("log",preprocessing.FunctionTransformer \
                (func=funcionesMatematicas.log_1,inverse_func=funcionesMatematicas.exp_1)))

        if self.__usar_pipeline_y:
            self.__pipeline_y=Pipeline(arrayPipeline)
    def fit_predict(self, X_train, y_train, X_test, y_test):
        """
        Extiende la clase del padre añadiendo las transformaciones numericas sobre la y si son necesarias
        :param X_train: pandas dataframe
        :param y_train:   pandas series
        :param X_test: pandas dataframe
        :param y_test:  pandas series
        """

        y_train=y_train.to_numpy().reshape(-1,1)

        if self.__usar_pipeline_y:
            y_train = self.__pipeline_y.fit_transform(y_train)


        # genera self.y_pred
        super().fit_predict( X_train, y_train, X_test, y_test)

        #recuperamos las predicciones originales para comparar con y_test
        if self.__usar_pipeline_y:
            self.y_pred = self.__pipeline_y.inverse_transform(self.y_pred.reshape(-1,1))


    def metrics(self):


        mape = mean_absolute_percentage_error(self.y_test, self.y_pred)
        smape= machine_learninge_utils.SMAPE(self.y_test, self.y_pred)
        return { "mape": mape, "smape": smape}

    def predict(self,X_test=None,y_test=None):
        """
        Extiende la clase del padre añadiendo las transformaciones inversas sobre las predicciones si son necesarias

        :param X_test:
        :param y_test:
        :return:
        """
        # genera self.y_pred
        super().predict(X_test,y_test)

        # recuperamos las predicciones originales para comparar con y_test
        if self.__usar_pipeline_y:
            self.y_pred = self.__pipeline_y.inverse_transform(self.y_pred.reshape(-1,1))


        return self.y_pred

