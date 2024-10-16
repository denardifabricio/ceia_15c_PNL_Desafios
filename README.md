![img1](https://github.com/hernancontigiani/ceia_memorias_especializacion/raw/master/Figures/logoFIUBA.jpg)


## Alumno
Denardi, Fabricio

## Cohorte
15-2024

# Procesamiento de lenguaje natural
Les presento mi portfolio de desafíos superados durante el cursado de la asignatura **Procesamiento del Lenguaje Natural** del *Curso de Especialización en Inteligencia Artificial* dictado por la *Universidad de Buenos Aires (UBA)*

Les iré contando un resumen de cada uno de ellos y un link a la Jupiter Notebook principal en donde podrán ver en detalle todos los trabajos realizados.

## Comentario importante
Para terminar de comprender las conclusiones, ejemplos brindados y detalle, les recomiendo visitar cada una de las notebooks de los diferentes desafíos, cuyo link dejo más abajo con la leyenda 'Colab' y el número de desafío

# Desafío 1
## Vectorización de texto y modelo de clasificación Naïve Bayes con el dataset 20 newsgroups
Colab: [desafio_1](https://github.com/denardifabricio/ceia_15c_PNL_Desafios/blob/main/Desafio%201/Desafio_1.ipynb)

### Breve Resumen
La idea fue lograr vectorizar documentos de diferentes categorías y lograr identificar dicha categoría. Luego obtener los documentos más similares (usando similitud del coseno) y chequear  que correspondan a la misma categoría:


Por ejemplo:
```
==============================================================================================
==============================================================================================
El documento analizado es el índice 7030.
La categoría del mismo es: "sci.electronics". 
-------------------------------------------------------------------------------------------
»» documento con similaridad - Top 1
Tiene el índice 3668.
La categoría del mismo es: "sci.electronics".
-------------------------------------------------------------------------------------------
»» documento con similaridad - Top 2
Tiene el índice 652.
La categoría del mismo es: "sci.electronics".
-------------------------------------------------------------------------------------------
»» documento con similaridad - Top 3
Tiene el índice 2236.
La categoría del mismo es: "sci.electronics".
-------------------------------------------------------------------------------------------

```
#### Conclusiones más importantes
- Algunos documentos son más propensos a que, aplicando el algoritmo de similaridad por coseno, tengan otros documentos similares, de la misma categoría.

- A pesar de que algunos matches no pertenecen a la misma categoría, si nos ponemos a leer, podemos encontrar tópicos o temas parecidos. Lo cual le encuentro sentido. Por ejemplo, en el caso analizado de comp.os.ms-windows.misc, habla de un magazine de tecnlogía, y discute sobre temas de hardware, gráficos, y la similitud del coseno encuentra justamente documentos que se encuadran en esas categorías.

- Si bien la bibliografía de scikit-learn advierte del uso del parámetro stop_word, dado que los textos están en inglés y el único valor válido justamente es dicho idioma, probé parametrizarlo y considero que obtuve una pequeña mejora. 

- Aplicar n-gramas también considero que fue satisfactorio aunque no noté demasiada mejoría, además luego en el ejercicio de la matriz trampuesta (para ver la relación de palabras, no tenía el efecto deseado)

- Intenté "jugar" con los valores de min_df y max_df, estableciendo diferentes rangos de threshold, pero en este caso no tuve éxito, ya que rangos muy restrictivos me eliminaban por completo la salida, dando el siguiente error: "After pruning, no terms remain. Try a lower min_df or a higher max_df.". En otros casos me eliminaba textos de categorías o palabras del corpus, como por ej "car" que justo era la que estaba de ejemplo.

- Este algoritmo no captura sinónimos, es decir frases o documentos que son semánticamente similares pero que no comparten términos, la similaridad del coseno nos va a dar cero. Es por eso que necesitaremos modelos más complejos, como embeddings, que desarrollaremos en el próximo desafío.



#### Naive Bayes
Luego se procedió a entrenar modelos de clasificación Naïve Bayes para maximizar el desempeño de clasificación (f1-score macro) en el conjunto de datos de test. 

```
naive_classifier = MultinomialNB()
print("Naive Bayes - Parámetros por defecto")
analize_naive_bayes(naive_classifier)
```
Naive Bayes - Parámetros por defecto
El F1 score con average macro para el modelo MultinomialNB es: 0.6468

```
naive_classifier = MultinomialNB(alpha=0.01, force_alpha=True, fit_prior=False, class_prior=None)
print("Naive Bayes - Fine tuning")
analize_naive_bayes(naive_classifier)
```
Naive Bayes - Fine tuning
El F1 score con average macro para el modelo MultinomialNB es: 0.6877

```
naive_classifier = ComplementNB()
print("Complement Naive Bayes - Parámetros por defecto")
analize_naive_bayes(naive_classifier)
```
Complement Naive Bayes - Parámetros por defecto
El F1 score con average macro para el modelo ComplementNB es: 0.6936

```
naive_classifier = ComplementNB(alpha=0.1, force_alpha=True, fit_prior=False, class_prior=None, norm=False)
print("Complement Naive Bayes - Fine tuning")
analize_naive_bayes(naive_classifier)
```
Complement Naive Bayes - Fine tuning
El F1 score con average macro para el modelo ComplementNB es: 0.6919


#### Conclusiones más importantes:
- Realizar un fine tuning de los parámetros de los modelos resultó una buena elección. Proporcionalmente, tuvo mayor efecto en el Naive Bayes que en el Complement Naive Bayes.

- El mejor modelo obtenido fue el ComplementNB ajustando los valores de los parámetros.

- Como comentario personal, creo que el mejor F1 score obtenido, no resulta suficiente. Por ejemplo, yo trabajo para un Broker de seguros, y tenemos diferentes tipos de documentos que usualmente necesitamos clasificar. Este score no resulta prometedor, ya que puede genera problemas regulatorios o inclusos legales, ya que el rubro de seguros es muy incumbente y normado, que se rige por muchímas regulaciones.


#### Estudio de similaridad de palabras
El objetivo fue transponer la matriz documento-término. De esa manera se obtiene una matriz término-documento que puede ser interpretada como una colección de vectorización de palabras. 

Adicionalmente debí estudiar ahora similaridad entre palabras tomando X palabras y estudiando sus X más similares. 


**El resultado fue bastante bueno:**
```


==============================================================================================
==============================================================================================
El término analizado es el índice 25717.
Longitud: 3
El término es el siguiente:
car
-------------------------------------------------------------------------------------------
»» término con similaridad - Top 1
Tiene el índice 25863.
Longitud: 4
El término es el siguiente:
cars
-------------------------------------------------------------------------------------------
»» término con similaridad - Top 2
Tiene el índice 30471.
Longitud: 9
El término es el siguiente:
criterium
-------------------------------------------------------------------------------------------
»» término con similaridad - Top 3
Tiene el índice 32086.
Longitud: 6
El término es el siguiente:
dealer
```

#### Conclusiones más importantes
- Esta similaridad depende exclusivamente del contexto y la semántica de los documentos del corpus que poseemos. Por ejemplo, para la palabra *star*, el término más similar encontrado es *trek* que forman *star trek*, una reconocida serie de televisión. O para *car*, obtenemos *dealer* como uno de sus términos similares, que se refiere a los concesionarios de autos.

- En otros caso, como *car* también obtenemos su plural como término, lo cuál también tiene sentido.

- Lo que a mí me resultó contraintuitivo, es que el algoritmo, encontró pocos sinónimos o palabras dentro de una misma "especie", por ejemplo para "green", en el top-5 recién apareció otro color, *yellow*. Tratando de entender el porque, llegué a la conclusión que está correcto, ya que este algoritmo no comprende de sinónimos, antónimos, grupo de palabras, sino como se relacionan los términos en los documento del corpus y en los contextos de estos, como por ejemplo *green urbina*, que puede ser un ex jugador de beseball, cuya camiseta del equipo que lo hizo estrella era verde o el nombre de una planta.

- El punto anterior considero que se debe a que estos modelos no interpretan el contexto, es decir, en dónde esta la palabra y sus término vecinos.

Para entender mejor, ir a la notebook y revisar los ejemplos.

# Desafío 2
## Custom embedddings con Gensim
Colab: [desafio_2](https://github.com/denardifabricio/ceia_15c_PNL_Desafios/blob/main/Desafio%202/desafio_2.ipynb)

### Breve Resumen
### Objetivo
El objetivo es utilizar documentos / corpus para crear embeddings de palabras basado en ese contexto. Se utilizará recetas de cocinas para generar los embeddings, es decir, que los vectores tendrán la forma en función de como esa banda haya utilizado las palabras en sus canciones.

Las recetas pueden encontrarlas en:
[Recetario 1](https://github.com/denardifabricio/ceia_15c_PNL_Desafios/tree/main/Desafio%202/recipes_1)
[Recetario 2](https://github.com/denardifabricio/ceia_15c_PNL_Desafios/tree/main/Desafio%202/recipes_2)


### Busqueda de similaridad de palabras:
Mediante el uso  de Gessim logramos identificar palabras similares en el contexto de recetas:
```
# Palabras que MÁS se relacionan con...:
w2v_model.wv.most_similar(positive=["chocolate"], topn=10)
```

```
[('chips', 0.9587956666946411),
 ('caramel', 0.8662093877792358),
 ('unsweetened', 0.8509345054626465),
 ('piece', 0.8420158624649048),
 ('substitute', 0.8306741118431091),
 ('sweetened', 0.8303201198577881),
 ('heavy', 0.8193446397781372),
 ('topping', 0.8093628883361816),
 ('bitter', 0.8066312074661255),
 ('pre', 0.7897059917449951)]
```

A su vez palabras menos similares:
```
# Palabras que MENOS se relacionan con...:
w2v_model.wv.most_similar(negative=["pepper"], topn=10)
```

```
[('refrigerated', 0.25176477432250977),
 ('16', 0.23496967554092407),
 ('yield', 0.20719610154628754),
 ('cutting', 0.2057875692844391),
 ('makes', 0.19414596259593964),
 ('note', 0.1866300106048584),
 ('approx', 0.1845967024564743),
 ('24', 0.1758279651403427),
 ('brownies', 0.1497783660888672),
 ('these', 0.13892851769924164)]
```


También podemos ver un fragmento del gráfico de baja dimensionalidad:
![img1](https://github.com/denardifabricio/ceia_15c_PNL_Desafios/blob/main/Desafio%202/plot.png)




### Conclusiones
- La performance del modelo entrenado es muy buena. A pesar de las limitaciones del mismo, pude encontrar relaciones muy interesantes, la mayoría de esas esperadas, aunque otras que dependen exclusivamente del contexto de las recetas elegidas, que requeriran de empliar el dataset para evaluar si siguen o no siendo relevamentes, en cuyo caso, habría que ver el origen de esta relación.

- Para el análisis de este y cualquier problema, es necesario contar con conocimientos del tema elegido, al menos para evaluar la efectividad del mismo. Mi hobbie en gastronomía, me permitió realizar un buen análisis del mismo.

- Vemos en el gráfico del punto anterior que la reducción de dimensiones, tuvo bastante éxito. Veo, sin embargo, que palabras genéricas o no asociadas a la gastronomía, tienen cercanía con otros término vinculados a dicha temática, dificultando, a mi criterio, un análisis más limpio. No quise seguir sumando al hiperparámetro de cantidad mínima de apariciones, porque me eliminaba términos muy interesante y que, si bien no aparecen en todas las recetas, están estrechamente relacionadas con el tema de interés, como por ejemplo la canela.


# Desafío 3
## Modelo de lenguaje con tokenización por palabras y caracteres
Colab: [desafio_3](https://github.com/denardifabricio/ceia_15c_PNL_Desafios/blob/main/Desafio%203/Desafio_3.ipynb)

El TP fue dividido en 2 sub notebooks:
[desafio_3_modelo_lenguaje_natural_char.ipynb](https://github.com/denardifabricio/ceia_15c_PNL_Desafios/blob/main/Desafio%203/desafio_3_modelo_lenguaje_natural_char.ipynb) se trabajó con un modelo de lenguaje natural basado en caracteres y en [desafio_3_modelo_lenguaje_natural_word.ipynb](https://github.com/denardifabricio/ceia_15c_PNL_Desafios/blob/main/Desafio%203/desafio_3_modelo_lenguaje_natural_word.ipynb) con un modelo de lenguaje natural basado en palabras.

### Dataset
Se trabajó con los mismos recetarios del desafío anterior

### Objetivos
- Seleccionar un corpus de texto sobre el cual entrenar el modelo de lenguaje.
- Realizar el pre-procesamiento adecuado para tokenizar el corpus, estructurar el dataset y separar entre datos de entrenamiento y validación.
- Proponer arquitecturas de redes neuronales basadas en unidades recurrentes para implementar un modelo de lenguaje.
- Con el o los modelos que consideren adecuados, generar nuevas secuencias a partir de secuencias de contexto con las estrategias de greedy search y beam search determístico y estocástico. En este último caso observar el efecto de la temperatura en la generación de secuencias.



### Modelo basado en caracteres
La arquitectura del mejor modelo logrado fue:
```
Model: "sequential"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 time_distributed (TimeDistr  (None, None, 139)        0         
 ibuted)                                                         
                                                                 
 simple_rnn (SimpleRNN)      (None, None, 200)         68000     
                                                                 
 dense (Dense)               (None, None, 139)         27939     
                                                                 
=================================================================
Total params: 95,939
Trainable params: 95,939
Non-trainable params: 0
_________________________________________________________________
```

Y como resultados de validación obtuve:
```
Input: add salt to the
Ouput:add salt to the oven 
----------------------------------
Input: put the chicken in the
Ouput:put the chicken in the oven 
----------------------------------
Input: heat the oil in a
Ouput:heat the oil in a large
----------------------------------
Input: add the flour to the
Ouput:add the flour to the sauce
----------------------------------
Input: mix the eggs with the
Ouput:mix the eggs with the sauce
----------------------------------
```

#### Conclusiones
- Encontré que el costo computacional del entrenamiento del modelo es muy alto. Considerando que limité bastante el número de recetas, en un ambiente productivo, no lo veo óptimo para su uso. Esto se repite tanto como en el modelo de word como este, aunque aquí, al procesar por caracteres, se refuerza esta problemática.

- En cuanto a la predicción de secuencias, noto que las primeras palabras tienen sentido, no así si intentamos predecir una cadena más larga, en donde, a pesar de encontrar palabras del idioma inglés (las recetas están escritas en dicho idioma), se va perdiendo el sentido de la oración, por ej: "add salt to the pasta into the pasta into the".

- Utilizando beam search y muestreo aleatorio, y a pesar de la simplesa del algoritmo y que limité el número de recetas, encuentro que las secuencia sugerencias tienen un léxico coherente, en cuanto a la semántica, tal vez falla en algunas, por ejemplo, uno no agrega sal dentro del horno, pero sí se lo hace con un pollo (predicción correcta encontrada por el modelo también)

- Comparando el modelo por palabras y por caracter, considero que, aunque no con grandes diferencias, el modelo por caracter tienen mejor accuracy, analizándolo semánticamente, es decir revisando si la salida de las predicciones tienen sentido en el contexto de las recetas de cocina.


### Modelo basado en palabras

La longitud de las secuencias es la siguiente y me permitió ajustar los parámetros del modelo:

![img1](https://github.com/denardifabricio/ceia_15c_PNL_Desafios/blob/main/Desafio%203/longitud_word.png)

```
Model: "sequential_2"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 embedding_2 (Embedding)     (None, None, 50)          197450    
                                                                 
 lstm_4 (LSTM)               (None, None, 100)         60400     
                                                                 
 dropout_1 (Dropout)         (None, None, 100)         0         
                                                                 
 lstm_5 (LSTM)               (None, None, 100)         80400     
                                                                 
 dense_2 (Dense)             (None, None, 3949)        398849    
                                                                 
=================================================================
Total params: 737,099
Trainable params: 737,099
Non-trainable params: 0
_________________________________________________________________
```

Y como resultados de validación obtuve:
```
Input: add salt to the
['add salt to the sides and cook for a wooden']
Input: put the chicken in the
['put the chicken in the heat and cook for 1 5']
Input: heat the oil in a
['heat the oil in a large saucepan add the potatoes and']
Input: add the flour to the
['add the flour to the top of the cheese and took']
Input: mix the eggs with the
['mix the eggs with the butter and pepper to another her']
```
#### Conclusiones
- Encontré que el costo computacional del entrenamiento del modelo es muy alto. Considerando que limité bastante el número de recetas, en un ambiente productivo, no lo veo óptimo para su uso.

- La predicción de secuencia si bien respeta el léxico de recetas, semanticamente no lo veo tan bueno conforme se intenta predecir mayor cantidad de palabras.

- En cuanto a los hiperparámetros del modelo, intenté con diferentes números de embeddings, pero el mejor fue el inicial, con 50 de ellos. 

- Tuve que agregar un dropout porque a partir de la época 8/10 comenzaba a tener un gran overfitting, esto no solo hizo que el overfitting disminuyera notablemente sino que mejoró la calida de la respuesta (semánticamente, es decir el el sentido de las frases en el contexto de las recetas).

- Adicionalmente, probé con diferentes criterios del tamaño de secuencia, siendo el del percentile, el que me funcionó mejor.

# Desafío 4
## LSTM Bot QA
Colab: [desafio_4](https://github.com/denardifabricio/ceia_15c_PNL_Desafios/blob/main/Desafio%204/Desafio_4.ipynb)

## Objetivo
El objecto es utilizar datos disponibles del challenge ConvAI2 (Conversational Intelligence Challenge 2) de conversaciones en inglés. Se construirá un BOT para responder a preguntas del usuario (QA).\


### Modelo entrenado
Se entrenó un modelo con la siguiente arquitectura:

![img1](https://github.com/denardifabricio/ceia_15c_PNL_Desafios/blob/main/Desafio%204/modelo.png)


Y el resultado logrado fue:
```
-----------------------------
Ejemplo 0/6 - HOW ARE YOU?
-----------------------------
Representacion en vector de tokens de ids [10, 7, 2]
Padding del vector: [[ 0  0  0  0  0  0 10  7  2]]
Input: How Are you?
>>> Response: I AM DOING WELL HOW ARE YOU

-----------------------------
Ejemplo 1/6 - DO YOU READ?
-----------------------------
Representacion en vector de tokens de ids [3, 2, 23]
Padding del vector: [[ 0  0  0  0  0  0  3  2 23]]
Input: Do you read?
>>> Response: I DO I LOVE TO READ

-----------------------------
Ejemplo 2/6 - DO YOU HAVE ANY PET?
-----------------------------
Representacion en vector de tokens de ids [3, 2, 16, 31, 252]
Padding del vector: [[  0   0   0   0   3   2  16  31 252]]
Input: Do you have any pet?
>>> Response: YES I HAVE A TIGER

-----------------------------
Ejemplo 3/6 - WHERE ARE YOU FROM?
-----------------------------
Representacion en vector de tokens de ids [52, 7, 2, 39]
Padding del vector: [[ 0  0  0  0  0 52  7  2 39]]
Input: Where are you from?
>>> Response: I AM FROM THE UNITED STATES


-----------------------------
Ejemplo 4/6 - WHAT IS YOUR NAME?
-----------------------------
Representacion en vector de tokens de ids [4, 15, 21, 51]
Padding del vector: [[ 0  0  0  0  0  4 15 21 51]]
Input: What is your name?
>>> Response: I AM NOT SURE WHAT YOU MEAN


-----------------------------
Ejemplo 5/6 - DO YOU LIKE CHOCHOLATE?
-----------------------------
Representacion en vector de tokens de ids [3, 2, 12]
Padding del vector: [[ 0  0  0  0  0  0  3  2 12]]
Input: Do you like chocholate?
>>> Response: YES
```

#### Conclusiones
- Muchas preguntas fueron respondidas con cocherencia semántica, es decir que guardan relación con lo preguntado. En cambio, otras no, o al menos son ambiguas, lo que motiva a futuro a:
    - Utilizar y aprovechar más recursos computacionales.
    - Realizar una reingeniería de la red para hacerla lo sufucientemente compleja.
    - Utilizar Transfer Learning  y luego fine tuning a partir de modelos ya probados.

- Contrariamente a lo que esperaba, apilar una capa LSTM extra no mejoró la performance del modelo. Esto puede deberse a un sobreajuste, a un vanish del gradiente, a que no tenemos datos suficientes, entre otros.

- Un punto interesante es que a pesar de no tener una valid loss demasiado buena y con tendencia al sobreajuste, la performance resultó moderadamente buena.


# Desafío 5
## Bert Sentiment Analysis
Colab: [desafio_5](https://github.com/denardifabricio/ceia_15c_PNL_Desafios/blob/main/Desafio%205/Desafio_5.ipynb)

## Objetivo
Desarrollar un modelo de análisis de sentimiento.

Se trabajó con un dataset de críticas, cuyas clases, como podemos ver están desbalanceadas:
![img1](https://github.com/denardifabricio/ceia_15c_PNL_Desafios/blob/main/Desafio%205/clases2.png)

### Modelo entrenado
```
Model: "model"
__________________________________________________________________________________________________
 Layer (type)                Output Shape                 Param #   Connected to                  
==================================================================================================
 input_ids (InputLayer)      [(None, 280)]                0         []                            
                                                                                                  
 attention_mask (InputLayer  [(None, 280)]                0         []                            
 )                                                                                                
                                                                                                  
 tf_bert_model (TFBertModel  TFBaseModelOutputWithPooli   1094822   ['input_ids[0][0]',           
 )                           ngAndCrossAttentions(last_   40         'attention_mask[0][0]']      
                             hidden_state=(None, 280, 7                                           
                             68),                                                                 
                              pooler_output=(None, 768)                                           
                             , past_key_values=None, hi                                           
                             dden_states=None, attentio                                           
                             ns=None, cross_attentions=                                           
                             None)                                                                
                                                                                                  
 dropout_37 (Dropout)        (None, 768)                  0         ['tf_bert_model[0][1]']       
                                                                                                  
 dense (Dense)               (None, 128)                  98432     ['dropout_37[0][0]']          
                                                                                                  
 dropout_38 (Dropout)        (None, 128)                  0         ['dense[0][0]']               
                                                                                                  
 dense_1 (Dense)             (None, 5)                    645       ['dropout_38[0][0]']          
                                                                                                  
==================================================================================================
Total params: 109581317 (418.02 MB)
Trainable params: 99077 (387.02 KB)
Non-trainable params: 109482240 (417.64 MB)
__________________________________________________________________________________________________
```

PAra ver la performance, podemos ver la matriz de confusión:
Se trabajó con un dataset de críticas, cuyas clases, como podemos ver están desbalanceadas:

![img1](https://github.com/denardifabricio/ceia_15c_PNL_Desafios/blob/main/Desafio%205/matrix.png)

### Conclusiones
- Respecto a la heurística (azar del 20%, dado que son 5 clases) hay una mejora  pero resta mucho por mejorar.

- Quizás también podamos aumentar la cantidad de épocas, según vemos el gráfico, hay espacio para que el modelo siga aprendiendo. Por una cuestión de disponibilidad de recursos, lo dejé aquí.

- Como siempre se propone a futuro y para entornos productivos, realizar fine tuning con Cross Validation, Optuna o algún otro framework para tal fin.

# ¡Gracias por leer cada uno de los desafíos en los que participé!
Si querés conocer más de mí o tenés alguna consulta, podés escribirme a  _denardifabricio@gmail.com_ 

