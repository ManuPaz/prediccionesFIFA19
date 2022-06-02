
import matplotlib.pyplot as plt
import numpy as np
from pandas import DataFrame
import pandas as pd
from sklearn.decomposition import PCA

def plot_heat_map(cm,nombre_dir:str, title, xlabel, ylabel):
    figure, ax = plt.subplots()

    im = ax.imshow(cm, cmap=plt.get_cmap("cool"))
    ax.set_xticks(np.arange(len(cm.columns)), labels=cm.columns)
    plt.setp(ax.get_xticklabels(), rotation=90, ha="right", rotation_mode="anchor")
    ax.set_yticks(np.arange(len(cm.index)), labels=cm.index)
    plt.title(title)

    plt.xlabel(xlabel, labelpad=0.2)
    plt.ylabel(ylabel)

    for i, l in zip(range(len(cm.index)), cm.index):
        for j, k in zip(range(len(cm.columns)), cm.columns):
            if cm.loc[l, k] != 0:
                ax.text(j, i, cm.loc[l, k], ha="center", va="center", color="black", fontsize=6)

    figure.savefig(nombre_dir + "_heat_map.jpg")

    plt.show()

def pca_con_plots(df:pd.DataFrame,n_comps:int, nombre_dir:str):
    """
    PCA y plots de los componentes para cada eigenvec.

    :param df: dataframe original
    :param n_comps:  numero de componentes para pca
    :param nombre_archivo: nombre del directorio para guardar los componentes.
    """
    pca = PCA(n_components=n_comps)
    columnas = ["comp" + str(e) for e in range(1, n_comps + 1)]
    df_transformed = pd.DataFrame(pca.fit_transform(df))
    df_transformed.columns = columnas
    print("Varianza explicada {} ".format(np.cumsum(pca.explained_variance_ratio_)))
    data = pd.DataFrame(pca.components_).transpose()
    data.columns = ["comp_" + str(e) for e in data.columns]
    data.index = df.columns
    data = data.round(3)
    data.to_csv(nombre_dir + "pca_" + str(n_comps) + "comps.csv", index=True)
    for e in data.columns[0:6]:
        fig = plt.figure(figsize=(15, 12))
        x=[e for e in data.index]
        ax=plt.bar(x, data.loc[:, e].values,color="green")
        plt.title(e)
        plt.xticks(range(len(x)), x, rotation=90)

        fig.savefig(nombre_dir+"pca_"+str(n_comps)+"comps_"+e)

    return df_transformed




def plot_scatter_2d_with_classes(dataframe:pd.DataFrame,centroides:list,nombre_archivo:str=None,position_clusters:dict =None,girar=False):
    """
    Para hacer scatter plot 2D con colores que sean las categorias y puntos que en este caso son los centroides

    :param girar: no se usa. Para que sea coherente con  plot_scatter_3d_with_classes
    :param position_clusters:
    :param dataframe:
    :param centroides:  puntos a dibujar
    :param nombre_archivo: archivo para guardar el plot (opcional)
    """
    fig = plt.figure()

    if  position_clusters is not None:
        position_clusters_inv = {e: i for i, e in position_clusters.items()}
        dataframe.loc[:,dataframe.columns[2]]=dataframe.iloc[:,2].transform(lambda x:position_clusters[x])

    ax = plt.scatter(dataframe.iloc[:, 0], dataframe.iloc[:, 1], c=dataframe.iloc[:, 2])

    legend = ax.legend_elements()
    if    position_clusters is not None:
        for i,e in position_clusters_inv.items():
            legend [1][i]=e

    plt.legend(*legend)
    plt.scatter(centroides[:, 0], centroides[:, 1], s=200, marker='x',
                     color='black')  # Centroides
    plt.xlabel(dataframe.columns[0])
    plt.ylabel(dataframe.columns[1])


    plt.show()  # Mostramos a figura por pantalla
    if nombre_archivo is not None:
        fig.savefig(nombre_archivo)

def plot_scatter_3d_with_classes(dataframe:pd.DataFrame,centroides:list,nombre_archivo:str=None,
                                 position_clusters:dict =None,girar=False):
    """


    :param position_clusters: diccionario de cluster asignado a cada posicion
    :param dataframe:
    :param centroides: puntos a dibujar
    :param nombre_archivo: archivo para guardar el plot (opcional)
    """


    fig = plt.figure()
    threedee = fig.add_subplot(projection='3d')
    if  position_clusters is not None:
        position_clusters_inv = {e: i for i, e in position_clusters.items()}
        dataframe.loc[:,dataframe.columns[3]]=dataframe.iloc[:,3].transform(lambda x:position_clusters[x])
    if girar is False:
        x=0
        y=1
        z=2
    else:
        x=2
        y=1
        z=0
    ax=threedee.scatter(dataframe.iloc[:, x], dataframe.iloc[:, y], dataframe.iloc[:, z], c=dataframe.iloc[:, 3])

    legend=ax.legend_elements()
    if    position_clusters is not None:
        for i,e in position_clusters_inv.items():
            legend [1][i]=e

    threedee.legend(*legend)
    threedee.scatter(centroides[:, x], centroides[:, y],centroides[:, z],  marker='x', color='black')  # Centroides
    threedee.set_xlabel(dataframe.columns[x])
    threedee.set_ylabel(dataframe.columns[y])
    threedee.set_zlabel(dataframe.columns[z])

    if nombre_archivo is not None:
        if girar==False:
            fig.savefig(nombre_archivo)
        else:
            fig.savefig(nombre_archivo.split(".")[0]+"_girado.jpg")
