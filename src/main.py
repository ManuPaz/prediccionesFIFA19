import pandas as pd
from functions import obtener_variables_predictoras
from config import load_config
from functions import machine_learninge_utils, machine_learning_classification


if __name__ == '__main__':
    config = load_config.config()


    feature=config["entrenamiento"]["feature"]
    size=config["entrenamiento"]["train_test_split"]

    df =  pd.read_csv("../data/preprocesed/dataFIFA.csv")
    columnas = obtener_variables_predictoras.obtenerVariablesPredictoras(feature)

    X = df.loc[:, columnas]
    y=df.loc[:,feature]
    X_train, X_test, y_train, y_test=machine_learninge_utils.get_train_test(X, y, train_size=size, shuffle=False)
    a=machine_learning_classification.Classifier_HyperParameterTuning("lda")
    y_pred=a.fit_predict(X_train, y_train, X_test, y_test)
    metricas=a.metrics(y_test, y_pred)
    print(metricas)










# See PyCharm help at https://www.jetbrains.com/help/pycharm/