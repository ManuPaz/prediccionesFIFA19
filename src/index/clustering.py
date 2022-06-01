import pandas as pd
import os
os.chdir("../../")
from functions import obtener_variables_predictoras, entrenamiento
import logging
import warnings
from utils import  load_config
from sklearn.cluster import KMeans
from functions import functions_clustering
warnings.filterwarnings('ignore')
logging.config.fileConfig('logs/logging.conf')
logger = logging.getLogger('training')
import numpy as np
from sklearn.decomposition import PCA
if __name__ == '__main__':
    config=load_config.config()
    nombre_dir_plots=config["nombre_dir_reportes_plots_clustering"]
    if not os.path.isdir(nombre_dir_plots):
        os.makedirs(nombre_dir_plots)
    config = load_config.config()

    #variable categorica que se va a ver si coincide con los grupos
    feature="PositionGrouped"
    df = pd.read_csv("data/preprocesed/dataFIFA.csv", index_col=0)


    #obtenemos las columnas que queremos usar
    columnas = obtener_variables_predictoras.obtenerVariablesPredictoras("todas")
    position = df[feature].values.reshape(-1)
    df=df.loc[:,columnas]

    #si se baja la dimension con pca
    dimension_reduction=True
    #numero de componentes si se usa pca
    n_comps = 2

    #si se hace kmeans con varias k y se visualizan usando el metodo del codo
    buscar_k=True

    #parametros para k_means
    max_iter = 30
    n_init= 100
    k_max = 20
    k_min=2



    # k elegido usando el metodo del codo
    k_elegido = 4


    if dimension_reduction:
        pca=PCA(n_components=n_comps)
        columnas=["comp"+str(e) for e in range(1,n_comps+1)]
        df=pd.DataFrame(pca.fit_transform(df))
        df.columns=columnas
        print("Varianza explicada {} ".format(np.cumsum(pca.explained_variance_ratio_)))

    if buscar_k:
        costes_finales = []
        centroides_finales = []
        for k in range(k_min,k_max+1):
            print(k)
            model,coste,centroides=functions_clustering.train_k_means(k, df, inicio="random", n_init= n_init, max_iter =max_iter)
            costes_finales.append(coste)
            centroides_finales.append(centroides)



        functions_clustering.elbow_method(k_min, k_max, costes_finales, nombre_archivo=nombre_dir_plots  + "elbow_method_"+str(n_comps)+"comps_.jpg")


    #clustering con el k elegido

    model, coste, centroides = functions_clustering.train_k_means(k_elegido, df, inicio="random", n_init=n_init,
                                                                      max_iter=max_iter)

    if n_comps==2:
        funcion_plot_scatter=functions_clustering.plot_scatter_2d_with_classes

    elif n_comps == 3:
        funcion_plot_scatter = functions_clustering.plot_scatter_3d_with_classes


    nombre_archivo1 = nombre_dir_plots + feature + "_" + str(n_comps) + "comps_.jpg"
    nombre_archivo2 = nombre_dir_plots +"clusters_k_" + str(k_elegido) + "_" + str(n_comps) + "comps_.jpg"

    df["position"]=position
    funcion_plot_scatter(df, centroides,
                                                          nombre_archivo=nombre_archivo1)

    df=df.drop("position",axis=1)

    grupos = model.labels_
    df["labels"]=grupos
    funcion_plot_scatter(df, centroides,
                                                          nombre_archivo=nombre_archivo2)

