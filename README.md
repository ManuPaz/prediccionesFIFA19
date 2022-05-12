# Machine learning sobre datos del FIFA 19

Comparación de  diferentes modelos de regresión y clasificación para predecir diferentes variables de los jugadores del FIFA 19.
## VARIABLES A PREDECIR
* **Coste del jugador. (regresión)**
* **Sueldo del jugador (regresión).**
* **Posición del jugador (clasificación).**
##

## ESTRUCTURA DEL PROYECTO

En <b>src </b> están los scripts a ejecutar.
* **train_all**. Se entrenan todos los modelos de regresión (sobre Wage y Value ) y clasificación (sobre Position y PositionGrouped, que es la posicón agrupada).
* **forecast**. Se introduce por consola  parte de un nombre (jugador o equipo) y una variable a predecir y 
se muestran las predicciones ( con el mejor modelo o todos) y el valor real.
* **train_one_model_regression**. Para entrenar un solo modelo de regresión sobre una variable.
* **train_one_model_clasificacion**. Para entrenar un solo modelo de clasificación sobre una variable.
* **plots** Para visualizar los datos.

Todos los parámetros se configuran en <b>config.yaml</b>, los de entrenamiento y otros (hiperparámetros, hacer optimización de hiperparámetros en el entrenamiento o no,
modelos a utilizar, variables a predecir etc).

* **
Los modelos  se encapsulan en un objeto definodo en <b>functions/machine_learninge</b> que permite calcular las metricas, optimizar parametros,
,guarda en sus atributos los ultimos conjutos de  train con los que entrenó y de  test que predijo. Se guardan con pickle yse utilizan estos objetos para predecir.
