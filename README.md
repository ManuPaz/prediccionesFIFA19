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

## RESULTADOS
(Todas las métricas sobre validation test usando train_test_split de 0.7)

Regresión. 
* Value. Se consigue un SMAPE de 0.23 usando SVR con kernel rbf y haciendo una transformacion logaritmica sobre la variable (pero las metricas siempre calculadas sobre la variable
original, en caso de transformar se hace la transformacion inversa antes de calcular las metricas). Utilizando algoritmos lineales se consigue un SMAPE de 0.29.
* Wage. Usando todos los datos se consiguen malos resultados, los mejores usando SVR con kernel rbf y haciendo la tranformación de antes, se consigue un SMAPE de 0.45.
Eliminando los 4873 que tienen salario exacto de 1000 (en estos casos quizás no tiene mucha relación con las características) el SMAPE mejora y pasa a ser 0.39.
Si se divide el dataset por sueldos se pueden mejorar significativamente los resultados. 
Por ejemplo, cogiendo los jugadores con sueldos superiores o iguales a 25000 euros (1619 jugadores) se consigue un SMAPE de 0.28, con jugadores de entre 5000 y 25000 (5506 jugadores )se consigue un SMAPE de 0.30 y
con jugadores de sueldos inferiores o iguales a 5000 euros   (11889 jugadores)el SMAPE es de 0.36. Para estos últimos si además se eliminan los jugadores con salario de 1000 euros, quedando 6787, el SMAPE pasa a ser 0.25, y se puede conseguir incluso con un modelo lineal como lasso. 
Esto sin embargo no se podrá aplicar en la práctica porque puedes no saber en qué rango de sueldo está el jugador, habría que 
encontrar una variable que lo indique y ya se podría incluir en el modelo.

Se utiliza el dataset completo expceto los jugadores que tienen la variable a predecir igual a 0 (Value o Wage).
Estos son  jugadores sin equipo (todos los que tienen Wage=0 y solo hay 10 de 239 con Value=0 que tienen equipo), entonces este es el motivo del valor de la variable y es imposible predecirlo usando las careaterísticas. En el modelo seleccionado (SVR con kernel rbf)
incluyendo los jugadores sin equipo empeora el modelo, y no se consigue mejorar ingluyendo una nueva variable binaria que indique si el jugador tiene equipo o no.


Clasificación.
* Clasificación por grupos. Agrupando las posiciones en Delantero,Medio, Defensa y Portero se consiguen buenas métricas con muchos modelos.
La mejor es con random forest, con Accuracy de 0.86.
* Si utilizamos las 27 posiciones originales del FIFA, incluyendo solo las características de juego para predecir la accuracy es de 0.51 (usando también random forest), pero si incluimos el pie bueno asciende a 0.57.
Esto pasa porque muchos fallos en este modelo suceden entre la misma posición pero en el lado derecho, centro o izquierdo que se predicen casi siempre como centro.
* Si utilizamos las posiciones originales pero eliminando derecha e izquierda y asignando la posición central (16 clases) se consigue una accuracy ed 0.71 (también con random forest).
