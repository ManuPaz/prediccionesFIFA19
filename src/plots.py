import matplotlib.pyplot as plt
import logging
logging.getLogger('matplotlib.font_manager').disabled = True
from config import load_config
import pandas as pd
from functions import obtener_variables_predictoras
import numpy as np
def scatter(df,columna, feature, transform=None ):
    plt.figure(figsize=(15, 15))
    y=df.loc[:, feature]
    titulo = feature + " vs " + columna
    if transform is not None:
        y=transform(y)
        titulo+= " "+str(transform)
    plt.scatter(df[columna], y)

    plt.title(titulo)
    plt.xlabel(columna)
    plt.ylabel(feature)
    plt.show()
if __name__ == '__main__':

        config = load_config.config()
        plot = config["entrenamiento"]["plot"]
        df = pd.read_csv("../data/preprocesed/dataFIFA.csv")
        df=df.loc[(df.Wage>0) & (df.Value>0)]
        if plot:
            columnas = obtener_variables_predictoras.obtenerVariablesPredictoras("Wage")
            for columna in columnas:
                scatter(df, columna, "Value", transform=None)
                scatter(df, columna, "Value", transform=np.log)
