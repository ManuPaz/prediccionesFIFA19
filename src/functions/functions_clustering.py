import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans
import numpy as np
import seaborn as sns
from scipy.spatial import distance
import matplotlib.pyplot as plt

def elbow_method(k_minimo: int,k_maximo:int,costes:list, nombre_archivo:str=None):
    """
    Dibuja el plot del para seleccionar k usando el método del codo

    :param nombre_archivo: nombre del archivo para guardar el plot
    :param k_minimo: valor minimo de k con el que se implemeto kmeans
    :param k_maximo: valor maximo de k
    :param costes: array de costes para cada k
    """
    fig = plt.figure()
    ax=fig.add_subplot(111, projection='rectilinear')
    ax.set_xticks(np.arange(len(costes)), labels=range(k_minimo,k_maximo+1))
    plt.plot(range(k_minimo, k_maximo + 1), costes, '--', color="green")
    plt.title("Disminución  del coste")
    plt.xlabel("k")
    plt.ylabel("custo")
    plt.show()
    if nombre_archivo is not None:
        fig.savefig(nombre_archivo)

def optimizar_k(k_min: int, k_max:int, df_transformed:pd.DataFrame, n_init:int, max_iter:int):
    """
    Entrena kmeans con multiples k y devuelve los costes y centroides.

    :param k_min: minimo k para kmeans
    :param k_max: maximo k para kmeans
    :param df_transformed: dataframe
    :param n_init: numero de inicios en k_mean
    :param max_iter: maximo numero de iteraciones en kmeans
    :return: lista de costes para cada k y lista de centroides para cada k
    :rtype list and list
    """
    costes_finales = []
    centroides_finales = []
    for k in range(k_min, k_max + 1):
        print(k)
        model, coste, centroides = train_k_means(k, df_transformed, inicio="random", n_init=n_init,
                                                                      max_iter=max_iter)
        costes_finales.append(coste)
        centroides_finales.append(centroides)

    return costes_finales,centroides_finales

def train_k_means(num_clusters:int,df:pd.DataFrame, inicio:str="random", n_init:int=100, max_iter=100 ):
    """
    Entrena kmeans para un k y unos parametros concretos.

    :param num_clusters: numero de clusters=k
    :param df:  dataframe
    :param inicio: random o k-means++
    :param n_init: numero de inicios aleatorios (numero de veces que se ejecuta el algorimto)
    :param max_iter: numero maximo de iteraciones del algoritmo
    :return: modelo de kmeans, coste final y centroides finales
    """
    model = KMeans(n_clusters=num_clusters, init=inicio, n_init=n_init,
                   max_iter=max_iter)

    agrupamento = model.fit(df)
    coste=(agrupamento.inertia_ / (df.shape[0]))
    centroides = agrupamento.cluster_centers_

    return model,coste,centroides






def asignar_centroides_a_clusters(df: pd.DataFrame, centroides: list, uno_a_uno:bool =True):
    """
    Asigna cada posicion al cluster donde hay mas muestras de esa posicion

    :param uno_a_uno: asignar cada posicion a un cluster o puede haber varias posiciones en un cluster
    :return:
    :param df:
    :param centroides: centroides obteneidos con kmean
    :return: diccionario con el cluster para cada posicion
    :rtype: dict
    """
    df_grouped = df.groupby(["position","cluster"]).count()
    posiciones_clusters={}
    for posicion in df_grouped.index.get_level_values(0).unique():

        minimo=df_grouped.loc[posicion].groupby("cluster").max().idxmax().comp1
        if not uno_a_uno:
            posiciones_clusters[posicion]=minimo
        else:
            df_aux=df_grouped.loc[posicion].groupby("cluster").max()
            while(minimo in posiciones_clusters.values()):

                df_aux.drop(minimo,inplace=True)
                minimo =  df_aux.idxmax().comp1
            posiciones_clusters[posicion] = minimo
    return posiciones_clusters




