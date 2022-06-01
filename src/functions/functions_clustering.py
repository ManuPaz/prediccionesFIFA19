import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans
import numpy as np
import seaborn as sns
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
    plt.title("Diminución do custo")
    plt.xlabel("k")
    plt.ylabel("custo")
    plt.show()
    if nombre_archivo is not None:
        fig.savefig(nombre_archivo)


def train_k_means(num_clusters:int,df:pd.DataFrame, inicio:str="random", n_init:int=100, max_iter=100 ):
    """

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


def plot_scatter_2d_with_classes(dataframe:pd.DataFrame,centroides:list,nombre_archivo:str=None):
    """
    Para hacer scatter plot 2D con colores que sean las categorias y puntos que en este caso son los centroides

    :param dataframe:
    :param centroides:  puntos a dibujar
    :param nombre_archivo: archivo para guardar el plot (opcional)
    """
    fig = plt.figure()


    # Engadimos os scatterplots
    cols=dataframe.columns
    rel=sns.relplot(x = cols[0], y = cols[1], data = dataframe, height=8, hue =cols[2] )
    ax1=rel.fig.axes[0]

    ax1.scatter(centroides[:, 0], centroides[:, 1], s=200, marker='x', color='black')  # Centroides

    ax1.set_title("Variables %s y %s " % (dataframe.columns[0], dataframe.columns[1]))  # Poñemos un título
    ax1.set_xlabel("%s" % dataframe.columns[0])  # Nombramos os eixos
    ax1.set_ylabel("%s" % dataframe.columns[1])

    plt.show()  # Mostramos a figura por pantalla
    if nombre_archivo is not None:
        rel.savefig(nombre_archivo)

def plot_scatter_3d_with_classes(dataframe:pd.DataFrame,centroides:list,nombre_archivo:str=None):
    """


    :param dataframe:
    :param centroides: puntos a dibujar
    :param nombre_archivo: archivo para guardar el plot (opcional)
    """
    fig = plt.figure()
    threedee = fig.add_subplot(projection='3d')
    dict_colors={e:i for i,e in enumerate(np.unique(dataframe.iloc[:,3]))}
    dict_colors_inv = {i: e for i, e in enumerate(np.unique(dataframe.iloc[:, 3]))}
    dataframe.loc[:,dataframe.columns[3]]=dataframe.iloc[:,3].transform(lambda x:dict_colors[x])
    ax=threedee.scatter(dataframe.iloc[:, 0], dataframe.iloc[:, 1], dataframe.iloc[:, 2], c=dataframe.iloc[:, 3])
    legend=ax.legend_elements()
    for i,e in dict_colors_inv.items():
        legend [1][i]=e

    threedee.legend(*legend)
    threedee.scatter(centroides[:, 0], centroides[:, 1],centroides[:, 2], s=200, marker='x', color='black')  # Centroides
    threedee.set_xlabel(dataframe.columns[0])
    threedee.set_ylabel(dataframe.columns[1])
    threedee.set_zlabel(dataframe.columns[2])

    if nombre_archivo is not None:
       fig.savefig(nombre_archivo)