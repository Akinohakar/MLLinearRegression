# Momento de Retroalimentación: Módulo 2 Implementación de una técnica de aprendizaje máquina sin el uso de un framework.
Alan Eduardo Aquino Rosas <br/>
A01366912 <br/>


## Arquitectura del modelo
Para esta actividad se ocupo linear regression para predecir un valor continuo.<br/>

La hipotesis del modelo consta de 5 caracteristicas
<br/>

## Dataset

El data set utilizado fue obtenido de la pagina de UCI Machine Learning Dataset llamada Auto-mpg que reporta el consumo de gasolina de distintos coches que tienen diferentes caracteristicas.En base a las caractaristicas se debe predecir la variable dependiente en este caso es el consumo de galones de gasolina por milla.<br/>

Link de dataset:<br/>
https://www.kaggle.com/datasets/uciml/autompg-dataset<br/>

### Variables independientes

1. cylinders: multi-valued discrete
2. displacement: continuous
3. horsepower: continuous
4. weight: continuous
5. acceleration: continuous
6. model year: multi-valued discrete
7. origin: multi-valued discrete
8. car name: string (unique for each instance)

### Variables dependientes

1. mpg: continuous


## Preprocesamiento

Para el preprocesamiento se eliminaron las caracteristicas que no se ocnsideran importantes:
1. model year: multi-valued discrete
2. origin: multi-valued discrete
3. car name: string (unique for each instance)


## Resultados

Se dividio el dataset en 2,el training set y el validation set,se avaluarion los resultadods mediante la funcion de costo ded Error cuadratico medio.<br/>



En el training set se obtubo un costo de: 2135.35352970866 <br/>
En el validation set se obtuvo un costo de: :2433.690951839321 <br/>

### Predicionon prueba

Predecir el consumo por milla de un coche con:
1. cylinders 8
2. displacement 307
3. horsepower 130
4. weight 3504
5. displacement 12

Valor de prediccion de millas por galon 87.0676774982622 valor real 15 

