import pandas as pd
from config import load_config
from functions import obtener_variables_predictoras, entrenamiento
import numpy as np
from functions import machine_learning,machine_learninge_utils
import logging
import warnings
import pickle
from sklearn.decomposition import PCA
warnings.filterwarnings('ignore')
logging.config.fileConfig('../logs/logging.conf')
logger = logging.getLogger('training')


if __name__ == '__main__':
    config = load_config.config()
    df = pd.read_csv("../data/preprocesed/dataFIFA.csv")
    nombre_modelo="random_forest"
    feature =config["entrenamiento"]["feature_regresion"]



    columnas = obtener_variables_predictoras.obtenerVariablesPredictoras("todas")
    df=df.loc[df[feature]>0]
    X = df.loc[:, columnas]
    if config["entrenamiento"]["reducir_dimensionalidad"]:
        pca = PCA(n_components=6)
        pca.fit(X)
        X=pd.DataFrame(pca.transform(X))
        print(X.shape)
        print(np.cumsum(pca.explained_variance_ratio_))
    y = df.loc[:, feature]
    y=np.log(y)
    size = config["entrenamiento"]["train_test_split"]
    X_train, X_test, y_train, y_test = machine_learninge_utils.get_train_test(X, y, train_size=size, shuffle=True)
    transform =np.exp
    modelo_encapsulado = machine_learning.Regresor(nombre_modelo, feature, transform)
    modelo_encapsulado.fit_predict(X_train,  y_train,X_test, y_test)
    modelo_encapsulado.predict(X_train,y_train)
    print( modelo_encapsulado.metrics())
    modelo_encapsulado.predict(X_test, y_test)
    print(modelo_encapsulado.metrics())
    with open("../assets/modelosFinales/" + feature + "_" + nombre_modelo, 'wb') as handle:
        pickle.dump(modelo_encapsulado, handle, protocol=pickle.HIGHEST_PROTOCOL)

















# See PyCharm help at https://www.jetbrains.com/help/pycharm/