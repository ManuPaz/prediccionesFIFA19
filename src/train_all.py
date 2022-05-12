import pandas as pd
from config import load_config
from functions import obtener_variables_predictoras, entrenamiento
import numpy as np
from functions import machine_learning
import logging
import pickle

logging.config.fileConfig('../logs/logging.conf')
logger = logging.getLogger('training')



if __name__ == '__main__':
    config = load_config.config()
    df = pd.read_csv("../data/preprocesed/dataFIFA.csv")

    features_tipo = {
        "Wage": machine_learning.Regresor,
        "Value": machine_learning.Regresor,
        "PositionGrouped": machine_learning.Clasificador
    }



    #entrenamiento para cada variable de las 3 con todos sus modelos (clasificacin o regresion)
    for feature, tipo_modelo in features_tipo.items():
        transformacion = None
        if tipo_modelo == machine_learning.Clasificador:
            array_nombres_modelo = config["entrenamiento"]["classification"]["modelos"]

        else:
            array_nombres_modelo = config["entrenamiento"]["regression"]["modelos"]
            if config["entrenamiento"]["regression"]["transformacion_logaritmica"]:
                transformacion = np.log

        for nombre_modelo in array_nombres_modelo:

            grupos_variables = [obtener_variables_predictoras.obtenerVariablesPredictoras(feature),
                                obtener_variables_predictoras.obtenerVariablesPredictoras("todas")]
            #entrenamiento con todas las variables o solo las seleccionadas
            for variables, info_variables in zip(grupos_variables, ["variables_seleccionadas", "todas_las_variables"]):

                # obtenemos el objeto encapsulando el modelo entrenado, las metricas y las predicciones
                if tipo_modelo == machine_learning.Clasificador:
                    modelo_encapsulado, metricas, predicciones_clase, predicciones_probabilidades = entrenamiento.ejecutarModelo(
                        nombre_modelo, feature, tipo_modelo, df,
                        variables,
                        transformacion=transformacion)
                else:
                    modelo_encapsulado, metricas, predicciones,_ = entrenamiento.ejecutarModelo(nombre_modelo, feature,
                                                                                              tipo_modelo, df.loc[df[feature]>0],
                                                                                              variables,
                                                                                              transformacion=transformacion)

                #log de las metricas
                logger.info("Prediccion para {} con modelo {} con {}: {} ".format(feature, nombre_modelo,
                                                                                  info_variables, metricas))
                #guardamos el modelo

                with open("../assets/pruebasModelos/" +feature+"_"+ nombre_modelo, 'wb') as handle:
                    pickle.dump(modelo_encapsulado, handle, protocol=pickle.HIGHEST_PROTOCOL)

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
