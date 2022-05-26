
import pickle

import seaborn
from IPython.core.pylabtools import figsize
import plotly.express as px
from config import load_config
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings
import numpy as np
PLOT="heatmap"
plot_regression=False
dataframe_completo=False
warnings.filterwarnings("ignore")
if __name__=="__main__":
    df = pd.read_csv("../data/preprocesed/dataFIFA.csv")
    config = load_config.config()
    nombre_dir_modelos = config["nombre_dir_modelos"]
    if dataframe_completo:

        nombre_dir_plots=config["nombre_dir_reportes_plots_all"]
    else:
        nombre_dir_plots = config["nombre_dir_reportes_plots_test"]

    for tipo in ["regression","classification","classification_sin_texto"]:
        if not os.path.isdir(nombre_dir_plots+tipo):
            os.makedirs(nombre_dir_plots+tipo)




    for feature in config["entrenamiento"]["features_regresion"]+config["entrenamiento"]["features_clasificacion"]:
        if feature in   config["entrenamiento"]["features_regresion"]:
            tipo="regression"
        else:
            tipo="classification"

        data_aux = df

        if tipo=="regression":
            data_aux=data_aux.loc[data_aux[feature]>0]

        if feature =="Wage":
            data_aux=data_aux.loc[data_aux.Wage!=1000]
        if tipo=="regression" and not plot_regression:
            continue
        for modelo in config["entrenamiento"][tipo]["todos"]:
            with open(nombre_dir_modelos+feature+"_"+modelo,"rb") as file:
                modelo_encapsulado=pickle.load(file)
            if not dataframe_completo:
                jugadores_train = list(modelo_encapsulado.jugadores_train)
                data_aux=data_aux.loc[~data_aux.Name.isin(jugadores_train)]

            X = data_aux.loc[:, modelo_encapsulado.variables]
            y = data_aux.loc[:, feature]
            print(data_aux.shape)

            modelo_encapsulado.predict(X,y)
            y_pred= modelo_encapsulado.y_pred.reshape(-1)
            print(modelo_encapsulado.metrics())


            if tipo=="regression" and plot_regression:

                ax=sns.scatterplot(y,y_pred,color="green")
                figure = ax.get_figure()
                plt.title("Real vs prediccion: variable {}, modelo {}".format(feature, modelo))
                plt.xlabel("Real")
                plt.ylabel("Prediccion")

            elif tipo == "classification":
                y_pred=modelo_encapsulado.y_pred.reshape(-1)
                df_results=pd.DataFrame([y.to_numpy().reshape(-1),y_pred]).transpose()
                df_results.columns=["y","y_pred"]
                categories=modelo_encapsulado.modelo.classes_

                for k in ["y","y_pred"]:
                    df_results[k]=pd.Categorical(df_results[k],categories=categories)



                if PLOT=="heatmap":

                    cm = modelo_encapsulado.confusion_matrix()
                    figure, ax = plt.subplots()

                    im = ax.imshow(cm, cmap=plt.get_cmap("cool"))
                    ax.set_xticks(np.arange(len(cm.index)), labels=cm.index)
                    plt.setp(ax.get_xticklabels(), rotation=90, ha="right", rotation_mode="anchor")
                    ax.set_yticks(np.arange(len(cm.index)), labels=cm.index)
                    plt.title("Real vs prediccion: variable {}, modelo {}".format(feature, modelo))
                    plt.xlabel("Real",labelpad=0.2)
                    plt.ylabel("Prediccion")

                    for i, l in zip(range(len(cm.index)), cm.index):
                        for j, k in zip(range(len(cm.index)), cm.index):
                            if cm.loc[l, k] != 0:
                                text = ax.text(j, i, cm.loc[l, k], ha="center", va="center", color="black",fontsize=6)


                if PLOT=="seaborn":

                         g=seaborn.displot(data=df_results,x="y", y="y_pred",color="green",rug=True)
                         clases=modelo_encapsulado.modelo.classes_
                         for ax in g.axes.flat:
                                           for label in ax.get_xticklabels():
                                               label.set_rotation(90)
                         plt.title("Real vs prediccion: variable {}, modelo {}".format(feature, modelo))
                         plt.xlabel("Real")
                         plt.ylabel("Prediccion")

                         g.savefig(nombre_dir_plots + tipo + "_sin_texto/" + feature + "_" + modelo + ".jpg")
                         # fig = px.density_heatmap(df_results, x="y", y="y_pred", text_auto=True)

            if (tipo=="regression" and plot_regression) or (tipo=="classification" and PLOT!="seaborn"):


                figure.savefig(nombre_dir_plots + tipo + "/" + feature + "_" + modelo + ".jpg")

                plt.show()




