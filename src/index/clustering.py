import pandas as pd
import os
os.chdir("../../")
from functions import obtener_variables_predictoras, entrenamiento
import logging
import warnings
from utils import  load_config
from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix
import numpy as np
from functions import functions_clustering, plots
warnings.filterwarnings('ignore')
logging.config.fileConfig('logs/logging.conf')
logger = logging.getLogger('training')


if __name__ == '__main__':
    config=load_config.config()
    nombre_dir_plots=config["nombre_dir_reportes_plots_clustering"]
    nombre_dir_tablas=config["nombre_dir_reportes_tablas"]
    for dir in [ nombre_dir_plots, nombre_dir_tablas]:
        if not os.path.isdir(dir):
            os.makedirs(dir)


    #variable categorica que se va a ver si coincide con los grupos
    feature="PositionGrouped"
    df = pd.read_csv("data/preprocesed/dataFIFA.csv", index_col=0)


    #obtenemos las columnas que queremos usar
    columnas = obtener_variables_predictoras.obtenerVariablesPredictoras("todas")
    position = df[feature].values.reshape(-1)
    df= df.loc[:, columnas]

    #si se baja la dimension con pca
    dimension_reduction=True
    #numero de componentes si se usa pca
    n_comps = 3

    #si se hace kmeans con varias k y se visualizan usando el metodo del codo
    buscar_k=False

    #parametros para k_means
    max_iter = 30
    n_init= 100
    k_max = 20
    k_min=2



    # k elegido usando el metodo del codo
    k_elegido = 4

    #pca
    if dimension_reduction:
       df_transformed=plots.pca_con_plots(df,n_comps, nombre_dir_tablas)


    #optimizacion de k
    if buscar_k:
        costes_finales, centroides_finales=functions_clustering.optimizar_k(k_min, k_max, df_transformed, n_init, max_iter)

        functions_clustering.elbow_method(k_min, k_max, costes_finales, nombre_archivo=nombre_dir_plots  + "elbow_method_"+str(n_comps)+"comps_.jpg")


    #clustering con el k elegido

    model, coste, centroides = functions_clustering.train_k_means(k_elegido, df_transformed, inicio="random", n_init=n_init,
                                                                  max_iter=max_iter)




    nombre_archivo1 = nombre_dir_plots + feature + "_" + str(n_comps) + "comps_.jpg"
    nombre_archivo2 = nombre_dir_plots +"clusters_k_" + str(k_elegido) + "_" + str(n_comps) + "comps_.jpg"

    df_transformed["position"]=position
    grupos = model.labels_
    df_transformed["cluster"] = grupos

    position_clusters = functions_clustering.asignar_centroides_a_clusters(df_transformed, centroides)

    if n_comps==2:
        funcion_clustering= plots.plot_scatter_2d_with_classes
    elif n_comps == 3:
        funcion_clustering=  plots.plot_scatter_3d_with_classes



    funcion_clustering(df_transformed, centroides,
                     nombre_archivo=nombre_archivo1,position_clusters=position_clusters,girar=False)
    df_transformed = df_transformed.drop("position", axis=1)
    funcion_clustering(df_transformed, centroides,
                                                      nombre_archivo=nombre_archivo2, position_clusters=None,
                                                      girar=False)







    #si utilizamos un numero de cluster que corresponde con el numero de posiciones, asignamos clusters a posiciones para ver si coinciden el cluster
    df_transformed["position"] = position
    df_transformed["position"] = df_transformed.position.transform(lambda x: position_clusters[x])
    cm = confusion_matrix(df_transformed.position.values.reshape(-1), df_transformed.cluster.values.reshape(-1),labels=list(position_clusters.values()))
    cm = pd.DataFrame(cm, columns=list(position_clusters.values()), index=list(position_clusters.keys()))

    plots.plot_heat_map(cm,nombre_dir_plots, "Posicion vs cluster", "Cluster"," Posicion")





