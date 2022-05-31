import pandas as pd
import os
os.chdir("../../")
from functions import obtener_variables_predictoras, entrenamiento
import logging
import warnings
from utils import  load_config
from sklearn.cluster import KMeans
warnings.filterwarnings('ignore')
logging.config.fileConfig('logs/logging.conf')
logger = logging.getLogger('training')
from numpy import np
if __name__ == '__main__':
    config = load_config.config()
    nombre_dir_modelos = config["nombre_dir_modelos"]
    df = pd.read_csv("data/preprocesed/dataFIFA.csv", index_col=0)
    nombre_modelo = "ridge"

    #obtenemos las columnas que queremos usar
    columnas = obtener_variables_predictoras.obtenerVariablesPredictoras("todas")
    df=df.loc[:,columnas]

    n_iter = 100
    k_max = 20
    SEED_VALUE = 190463
    dataframe_mat = df.values

    costes_finales = []
    centroides_finais = []
    clusters_finais = []
    for k in range(2, k_max + 1):
        model = KMeans(n_clusters=k, init='random', n_init=100,
                       max_iter=n_iter, algorithm='full', random_state=SEED_VALUE)

        # Axustamos o modelo aos datos
        np.random.seed(SEED_VALUE)
        agrupamento = model.fit(df)
        costes_finales.append(agrupamento.inertia_ / (df.shape[0]))
        centroides = agrupamento.cluster_centers_
        centroides_finais.append(centroides)

