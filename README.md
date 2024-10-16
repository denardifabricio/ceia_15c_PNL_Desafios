![img1](https://github.com/hernancontigiani/ceia_memorias_especializacion/raw/master/Figures/logoFIUBA.jpg)


# Esta portada se encuentra en Desarrollo

## Alumno
Denardi, Fabricio

## Cohorte
15-2024

# Procesamiento de lenguaje natural
Les presento mi portfolio de desafíos superados durante el cursado de la asignatura **Procesamiento del Lenguaje Natural** del *Curso de Especialización en Inteligencia Artificial* dictado por la *Universidad de Buenos Aires (UBA)*

Les iré contando un resumen de cada uno de ellos y un link a la Jupiter Notebook principal en donde podrán ver en detalle todos los trabajos realizados.

# Desafío 1
## Vectorización de texto y modelo de clasificación Naïve Bayes con el dataset 20 newsgroups
Colab: [desafio_1](https://github.com/denardifabricio/ceia_15c_PNL_Desafios/blob/main/Desafio%201/Desafio_1.ipynb)

### Breve Resumen
La idea fue lograr vectorizar documentos de diferentes categorías y lograr categorizarlos. Luego obtener los documentos más similares (usando similitud del coseno) y chequear  que correspondan a la misma categoría:


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

El modelo ganador  fue:
```
naive_classifier = MultinomialNB(alpha=0.01, force_alpha=True, fit_prior=False, class_prior=None)
print("Naive Bayes - Fine tuning")
analize_naive_bayes(naive_classifier)
```

#### Conclusiones más importantes:
- Realizar un fine tuning de los parámetros de los modelos resultó una buena elección. Proporcionalmente, tuvo mayor efecto en el Naive Bayes que en el Complement Naive Bayes.

- El mejor modelo obtenido fue el ComplementNB ajustando los valores de los parámetros.

- Como comentario personal, creo que el mejor F1 score obtenido, no resulta suficiente. Por ejemplo, yo trabajo para un Broker de seguros, y tenemos diferentes tipos de documentos que usualmente necesitamos clasificar. Este score no resulta prometedor, ya que puede genera problemas regulatorios o inclusos legales, ya que el rubro de seguros es muy incumbente y normado, que se rige por muchímas regulaciones.


#### Estudio de similaridad de palabras
El objetivo fue transponer la matriz documento-término. De esa manera se obtiene una matriz término-documento que puede ser interpretada como una colección de vectorización de palabras. 

Adicionalmente debí estudiar ahora similaridad entre palabras tomando 5 palabras y estudiando sus 5 más similares. 


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
![BajaDim](/Desafio 2/plot.png)




### Conclusiones
- La performance del modelo entrenado es muy buena. A pesar de las limitaciones del mismo, pude encontrar relaciones muy interesantes, la mayoría de esas esperadas, aunque otras que dependen exclusivamente del contexto de las recetas elegidas, que requeriran de empliar el dataset para evaluar si siguen o no siendo relevamentes, en cuyo caso, habría que ver el origen de esta relación.

- Para el análisis de este y cualquier problema, es necesario contar con conocimientos del tema elegido, al menos para evaluar la efectividad del mismo. Mi hobbie en gastronomía, me permitió realizar un buen análisis del mismo.

- Vemos en el gráfico del punto anterior que la reducción de dimensiones, tuvo bastante éxito. Veo, sin embargo, que palabras genéricas o no asociadas a la gastronomía, tienen cercanía con otros término vinculados a dicha temática, dificultando, a mi criterio, un análisis más limpio. No quise seguir sumando al hiperparámetro de cantidad mínima de apariciones, porque me eliminaba términos muy interesante y que, si bien no aparecen en todas las recetas, están estrechamente relacionadas con el tema de interés, como por ejemplo la canela.


# Desafío 3
## Modelo de lenguaje con tokenización por palabras y caracteres
Colab: [desafio_3](https://github.com/denardifabricio/ceia_15c_PNL_Desafios/blob/main/Desafio%203/Desafio_3.ipynb)

# Desafío 4
## LSTM Bot QA
Colab: [desafio_4](https://github.com/denardifabricio/ceia_15c_PNL_Desafios/blob/main/Desafio%204/Desafio_4.ipynb)

# Desafío 5
## Bert Sentiment Analysis
Colab: [desafio_5](https://github.com/denardifabricio/ceia_15c_PNL_Desafios/blob/main/Desafio%205/Desafio_5.ipynb)

# ¡Gracias por leer cada uno de los desafíos en los que participé!
Si querés conocer más de mí o tenés alguna consulta, podés escribirme a  _denardifabricio@gmail.com_ 

