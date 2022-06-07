import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler


def metrics_clustering(labels_true, labels):
    homogeneity_score = metrics.homogeneity_score(labels_true, labels)
    completeness = metrics.completeness_score(labels_true, labels)
    v_measure = metrics.v_measure_score(labels_true, labels)
    adjusted_rand_index = metrics.adjusted_rand_score(labels_true, labels)
    adjusuted_mutual_info = metrics.adjusted_mutual_info_score(labels_true, labels)
    print("Homogeneity: %0.3f" % homogeneity_score)
    print("Completeness: %0.3f" % completeness)
    print("V-measure: %0.3f" % v_measure)
    print("Adjusted Rand Index: %0.3f" % adjusted_rand_index)
    print(
        "Adjusted Mutual Information: %0.3f"
        % adjusuted_mutual_info
    )
    return {"Homogeneity": homogeneity_score,
            "Completeness": completeness,
            "V-measure": v_measure,
            "Adjusted Rand Index": adjusted_rand_index,
            "Adjusted Mutual Information": adjusuted_mutual_info,
            }


def k_vs_metric(k_minimo: int, k_maximo: int, costes: list, nombre_archivo: str = None,
                title: str = "Disminucion del coste"):
    """
    Dibuja el plot del para seleccionar k usando el método del codo

    :param nombre_archivo: nombre del archivo para guardar el plot
    :param k_minimo: valor minimo de k con el que se implemeto kmeans
    :param k_maximo: valor maximo de k
    :param costes: array de costes para cada k
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='rectilinear')
    ax.set_xticks(np.arange(len(costes)), labels=range(k_minimo, k_maximo + 1))
    plt.plot(range(k_minimo, k_maximo + 1), costes, '--', color="green")
    plt.title(title)
    plt.xlabel("k")
    plt.ylabel("custo")
    plt.show()
    if nombre_archivo is not None:
        fig.savefig(nombre_archivo)


def optimize_dbscan(df: pd.DataFrame, posicion: np.ndarray or pd.Series, eps: list, min_samples: list):
    """

    :param df: dataframe
    :param posicion: array de labels reales
    :param eps:  lista para optimizar eps de DBSCAN
    :param min_samples:  lista para optimizar min_samples de DBSCAN
    :return : diccionario con V-measure para cada valor
    """
    resultados = {}
    for k in eps:
        for i in min_samples:
            db = train_dbscan(df, k, i)
            labels = db.labels_
            resultados[(k, i)] = metrics_clustering(labels, posicion)["V-measure"]
    return resultados


def train_dbscan(df: pd.DataFrame, eps: float = 0.3, min_samples: int = 10):
    """

    :param df: dataframe
    :param eps: maxima distancia entre puntos para conseiderarlos vecinos
    :param min_samples: menor numero de puntos para tener un centro de densidad
    :return: DBSCAN fitted
    :rtype: DBSCAN
    """
    # transformamos X para aplicar el parametros esp de manera uniforme para cualquier dataframe
    X = StandardScaler().fit_transform(df)
    db = DBSCAN(eps=eps, min_samples=min_samples).fit(X)
    core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
    core_samples_mask[db.core_sample_indices_] = True
    labels = db.labels_

    # Number of clusters in labels, ignoring noise if present.
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise_ = list(labels).count(-1)

    print("Estimated number of clusters: %d" % n_clusters_)
    print("Estimated number of noise points: %d" % n_noise_)

    return db


def optimizar_k(k_min: int, k_max: int, df_transformed: pd.DataFrame, n_init: int, max_iter: int, posicion: np.array):
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
    metricas = {"Homogeneity": [],
                "Completeness": [],
                "V-measure": [],
                "Adjusted Rand Index": [],
                "Adjusted Mutual Information": []
                }
    for k in range(k_min, k_max + 1):
        print(k)
        model, coste, centroides = train_k_means(k, df_transformed, inicio="random", n_init=n_init,
                                                 max_iter=max_iter)
        costes_finales.append(coste)
        centroides_finales.append(centroides)

        labels = model.labels_
        metrics_k = metrics_clustering(df_transformed, labels, posicion)
        for e in metricas.keys():
            metricas[e].append(metrics_k[e])

    return costes_finales, centroides_finales, metricas


def train_k_means(num_clusters: int, df: pd.DataFrame, inicio: str = "random", n_init: int = 100, max_iter=100):
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
    coste = (agrupamento.inertia_ / (df.shape[0]))
    centroides = agrupamento.cluster_centers_

    return model, coste, centroides


def asignar_centroides_a_clusters(df: pd.DataFrame, centroides: list, uno_a_uno: bool = True):
    """
    Asigna cada posicion al cluster donde hay mas muestras de esa posicion

    :param uno_a_uno: asignar cada posicion a un cluster o puede haber varias posiciones en un cluster
    :return:
    :param df:
    :param centroides: centroides obteneidos con kmean
    :return: diccionario con el cluster para cada posicion
    :rtype: dict
    """
    df_grouped = df.groupby(["position", "cluster"]).count()
    posiciones_clusters = {}
    for posicion in df_grouped.index.get_level_values(0).unique():

        minimo = df_grouped.loc[posicion].groupby("cluster").max().idxmax().comp1
        if not uno_a_uno:
            posiciones_clusters[posicion] = minimo
        else:
            df_aux = df_grouped.loc[posicion].groupby("cluster").max()
            while (minimo in posiciones_clusters.values()):
                df_aux.drop(minimo, inplace=True)
                minimo = df_aux.idxmax().comp1
            posiciones_clusters[posicion] = minimo
    return posiciones_clusters
