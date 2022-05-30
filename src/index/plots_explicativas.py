import matplotlib.pyplot as plt
import logging
import os
os.chdir("../../")
logging.getLogger('matplotlib.font_manager').disabled = True
from utils import load_config
import pandas as pd
from functions import obtener_variables_predictoras
import numpy as np
def scatter(X,columna,y, feature, transform=None ):
    plt.figure(figsize=(15, 15))
    titulo = feature + " vs " + columna
    if transform is not None:
        y=transform(y)
        titulo+= " "+str(transform)

    plt.scatter(X[columna], y)

    plt.title(titulo)
    plt.xlabel(columna)
    plt.ylabel(feature)
    plt.show()
if __name__ == '__main__':

        config = load_config.config()
        plot = config["entrenamiento"]["plot"]
        df = pd.read_csv("data/preprocesed/dataFIFA.csv")
        df=df.loc[(df.Wage>0) & (df.Value>0)]
        feature="Wage"
        columnas = obtener_variables_predictoras.obtenerVariablesPredictoras(feature)
        X=df.loc[:,columnas]

        if plot:
            for columna in X.columns:
                y = df.loc[:, feature]
                print(columna)
                scatter(X, columna,  y,feature, transform=None)
                scatter(X, columna, y,feature, transform=np.log)
