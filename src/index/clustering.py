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


def clustering_kmeans(df_transformed,  position, nombre_dir_plots ):
    # si se hace kmeans con varias k y se visualizan usando el metodo del codo
    buscar_k =False

    # parametros para k_means
    max_iter = 100
    n_init = 100
    k_max = 20
    k_min = 2

    # k elegido usando el metodo del codo
    k_elegido = 8
    # optimizacion de k
    if buscar_k:
        costes_finales, centroides_finales,metricas = functions_clustering.optimizar_k(k_min, k_max, df_transformed, n_init,
                                                                              max_iter,position)

        functions_clustering.k_vs_metric(k_min, k_max, costes_finales,
                                         nombre_archivo=nombre_dir_plots + "/optimizacion/" + "elbow_method_" + str(n_comps) + "comps+_metrica_coste.jpg", title="Variacion del coste")
        for key,array in metricas.items():
            functions_clustering.k_vs_metric(k_min, k_max, array,
                                             nombre_archivo=nombre_dir_plots+ "/optimizacion/"  + str(
                                                  n_comps) + "comps+_metrica_{}.jpg".format(key), title=key)



        # clustering con el k elegido

    model, coste, centroides = functions_clustering.train_k_means(k_elegido, df_transformed, inicio="random", n_init=n_init,
                                                                  max_iter=max_iter)

    nombre_archivo1 = nombre_dir_plots + feature + "_" + str(n_comps) + "comps_.jpg"
    nombre_archivo2 = nombre_dir_plots + "clusters_k_" + str(k_elegido) + "_" + str(n_comps) + "comps_.jpg"

    df_transformed["position"] = position
    grupos = model.labels_
    df_transformed["cluster"] = grupos

    position_clusters = functions_clustering.asignar_centroides_a_clusters(df_transformed, centroides,
                                                                               uno_a_uno=uno_a_uno)

    if n_comps == 2:
        funcion_clustering = plots.plot_scatter_2d_with_classes
    elif n_comps == 3:
        funcion_clustering = plots.plot_scatter_3d_with_classes

    if k_elegido in num_clases:
        funcion_clustering(df_transformed, centroides,
                           nombre_archivo=nombre_archivo1, position_clusters=position_clusters, girar=False,
                           asignar_posiciones=uno_a_uno)
    df_transformed = df_transformed.drop("position", axis=1)
    funcion_clustering(df_transformed, centroides,
                       nombre_archivo=nombre_archivo2, position_clusters=None,
                       girar=False, asignar_posiciones=uno_a_uno)

    if k_elegido in num_clases:
        # si utilizamos un numero de cluster que corresponde con el numero de posiciones, asignamos clusters a posiciones para ver si coinciden el cluster
        df_transformed["position"] = position
        df_transformed["position"] = df_transformed.position.transform(lambda x: position_clusters[x])
        cm = confusion_matrix(df_transformed.position.values.reshape(-1), (df_transformed.cluster.values.reshape(-1)),
                              labels=np.unique(df_transformed.cluster.values.reshape(-1)))
        aux = pd.DataFrame(0, index=position_clusters.keys(), columns=range(k_elegido))
        for i in aux.index:
            aux.loc[i] = cm[position_clusters[i]]
        cm = aux
        plots.plot_heat_map(cm, nombre_dir_plots+"/heatmaps/" + "heat_map_k" + str(k_elegido) + "_ncomps" + str(n_comps),
                            "{} vs cluster".format(feature), "Cluster", " {}".format(feature))


    functions_clustering.metrics_clustering(position , df_transformed["cluster"])

def clustering_dbscan(df_trasnformed,  position, nombre_dir_plots ):
    optimizar =False
    if optimizar:
        resultados=functions_clustering.optimize_dbscan(df_trasnformed,position, eps=np.arange(0.1,0.7,0.1),min_samples=np.arange(10,300,30))


        min_index = max(resultados, key=resultados.get)
        print(min_index)

    eps=0.1
    min_samples=100
    db=functions_clustering.train_dbscan(df_trasnformed,eps=eps,min_samples=min_samples)
    labels=db.labels_
    functions_clustering.metrics_clustering(labels, position)
    if n_comps == 2:
        funcion_clustering = plots.plot_scatter_2d_with_classes

    elif n_comps == 3:
        funcion_clustering = plots.plot_scatter_3d_with_classes
    nombre_archivo1 = nombre_dir_plots + "_" + str(n_comps) + "comps_.jpg"
    nombre_archivo2 = nombre_dir_plots + "_"+feature+"_" + str(n_comps) + "comps_.jpg"

    df_trasnformed["position"]=position
    df_trasnformed["cluster"] = db.labels_
    funcion_clustering(df_trasnformed,nombre_archivo=nombre_archivo2, position_clusters=None,
                       girar=False, asignar_posiciones=False)

    df_trasnformed= df_trasnformed.drop("position",axis=1)
    funcion_clustering(df_trasnformed, nombre_archivo=nombre_archivo1, position_clusters=None,
                       girar=False, asignar_posiciones=False)


if __name__ == '__main__':
    config=load_config.config()

    nombre_dir_plots_kmeans=config["nombre_dir_reportes_plots_kmeans"]
    nombre_dir_plots_dbscan = config["nombre_dir_reportes_plots_dbscan"]
    nombre_dir_tablas=config["nombre_dir_reportes_tablas"]
    for dir in [ nombre_dir_plots_kmeans,  nombre_dir_plots_dbscan, nombre_dir_tablas]:
        if not os.path.isdir(dir):
            os.makedirs(dir)
        if not os.path.isdir(dir+"/heatmaps"):
            os.makedirs(dir+"/heatmaps")
        if not os.path.isdir(dir + "/optimizacion"):
            os.makedirs(dir + "/optimizacion")


    #variable categorica que se va a ver si coincide con los grupos
    feature="PositionGrouped"
    num_clases=[4, 14, 27]
    df = pd.read_csv("data/preprocesed/dataFIFA.csv", index_col=0)


    #obtenemos las columnas que queremos usar
    columnas = obtener_variables_predictoras.obtenerVariablesPredictoras("todas")
    position = df[feature].values.reshape(-1)

    df= df.loc[:, columnas]

    clustering="dbscan"

    #si se baja la dimension con pca
    dimension_reduction=True
    #numero de componentes si se usa pca
    n_comps = 2

    #parametros para la asignacion de clusters a posiciones
    uno_a_uno=True



    #pca
    if dimension_reduction:
       df_transformed=plots.pca_con_plots(df,n_comps, nombre_dir_tablas)


    if clustering=="kmeans":
        clustering_kmeans(df_transformed, position,  nombre_dir_plots_kmeans)
    elif clustering=="dbscan":
        clustering_dbscan(df_transformed, position,  nombre_dir_plots_dbscan )








