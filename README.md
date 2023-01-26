## A plain Comparison: UNet and Transformer

![imagen](images/output-nvidia-mit-b3.png)

### Resumen

La presente investigación muestra un marco base para hacer comparación de modelos orientados a resolver problemas de segmentación. Está motivado por una intuición que propone a un modelo de __Transformer__ más rápido que una __UNet__. Esto resulta  ser contra intuitivo, por la cantidad de poder computacional que se requiere al entrenar un __Transformer__ y es una diferencia fundamental en ambos modelos. 
Los resultados se inscriben en un marco de decisiones que se argumentan a lo largo del documento y  que no pretenden ser una regla general; pero que sí se ajustan a este tipo de problemas de segmentación.

Si bien en el transcurso de la argumentación para la investigación uso la generalidad de Transformer y Unet; Estas tienen nombre propio. El __Transformer__ usado es [SegFormer](https://github.com/NVlabs/SegFormer) lanzado en octubre del 2021. Y una arquitectura específica de [UNet](https://github.com/qubvel/segmentation_models.pytorch), constituida  por un Encoder  __Resnet__ y  pre-entrenado con __Imagenet__.

![imagen](images/segformer_architecture.png)

![imagen](images/unet.png)

### Antecedentes

Es común hacer este tipo de comparaciones, y lo más frecuente es comparar modelos de un mismo tipo. Las condiciones para este tipo de investigaciones es mucho más fácil, ya que se enfrentan características similares. Para el caso que les presento, los modelos son diferentes, así que requerirán de muchas iteraciones con diferentes parámetros y datos en diferentes calidades. Todo esto es necesario para determinar un nivel base en el que podamos establecer condiciones justas para ambos modelos.

En la presente investigación no se explaya en la definición de estos conceptos básicos para establecer una comparación equilibrada; pero se mencionan algunos criterios a considerar que no pretenden ser la regla general, pero establece el marco que define los límites de la investigación.

En principio el tiempo es un limitante. Habría que considerar, muy cuidadosamente, cuáles serían las variables relevantes que debía mantener para ser riguroso en los resultados y cuales podría omitir o manejar, con un grado de libertad, para reducir el tiempo de entrenamiento de los modelos. 

El entorno de trabajo también, es un factor importante. Para la investigación he usado __Colab__ y __Kaggle__. La idea general es que los resultados sean fácilmente replicables, por cualquier persona, sin necesidad de un hardware privativo económicamente.  Ambos entornos tienen pequeñas diferencias; pero en general arrojan resultados similares.

### Warning!

Es importante señalar que para poder usar el código de la presente investigación habrá que tener en cuenta los procedimientos elegidos y modificarlo para obtener los mejores resultados en orden a uno de los modelos. Nuevamente recalco que el objetivo actual es la comparación más que el buen rendimiento. 

### Los datos utilizados

Para emprender esta experiencia he tomado prestado los datos de la competición: [Multi-Disease, Multi-View & Multi-Center Right Ventricular Segmentation in Cardiac MRI](https://competitions.codalab.org/competitions/31559#learn_the_details). Todas las imágenes requeridas se pueden descargar en este [link](https://www.ub.edu/mnms-2/).
Si bien la competición ha terminado, se puede acceder a ellos completando un formulario. Los datos tienen un tamaño total de 4.33GB. Esto se convierte en un inconveniente, considerando la cantidad de veces que se requiere pre-procesar para introducirlos a los modelos que usaremos. Habría que tomar una decisión con relación a los tiempos necesarios para desarrollar estos procesos además de considerar también los términos de la competición. 

En ese sentido he procesado las imágenes respetando la proporción entre el ancho y el alto además de la cantidad de píxeles. Para las máscaras he utilizado una codificación con los siguientes valores:

|  descripción   |  valor  |
|:--------:|:---:|
|  Fondo   |  0  |
|  Tipo a  |  1  |
|  Tipo b  |  2  |
|  Tipo c  |  3  |

Todas las imágenes las he guardado en formato PNG. Así que no debe causar asombro ver las máscaras totalmente oscuras a simple percepción. Además, estoy usando solamente las imágenes que contienen el objeto de estudio, el corazón, y he descartado aquellas en donde la máscara NO tiene información alguna. Esto ha reducido el set de datos en un 30%. En este punto, es importante señalar que, en un entorno de producción es inviable. No pretendo mostrar el modelo mejor entrenado para este tipo de segmentación, mi objetivo es hacer una comparación y para ello solo requiero que ambos modelos reciban la misma calidad de datos. El lector podrá adaptar el modelo fácilmente, si accede a los datos completos.  


### Los modelos

Tenemos dos modelos diferentes: __UNet__ y __Transformer__. Lo primero a destacar es que se llevan casi 5 años de diferencia. Las arquitecturas  de tipo Unet han sido diseñadas especialmente para desarrollar segmentación, mientras que, originalmente los __Transformer__ estaban orientadas al desarrollo de NLP(Natural Language Processing).

La diferencia más significativa está orientada en la cantidad de cálculos y necesidad de poder computacional. Obviamente esto dependerá de la cantidad de parámetros que contenga el modelo; pero en general, a una similar cantidad de parámetros, los modelos basados en __Transformers__ requieren mayor capacidad de cálculo. Esto es un problema que debería ser mencionado en el apartado antecedentes; pero requería una explicación más profunda. En este punto me remito a una explicación más adelante.

Los estudios recientes muestran que las potencialidades de los __Transformers__ se orientan, también en otras tareas: _clasificación_, _detección_, _segmentación_. Las capacidades de los mecanismos de atención están proponiendo nuevos campos de uso que no fueron definidos en su diseño y que están siendo explotados por los desarrolladores.

|   Model   | Parámetros  |   Val_iou   |     Time (min)    |
|:-------------:|:---------------:|:---------------:|:---------------:|
| nvidia/mit-b1 | 13.7 M          | 0.827           | 24:52        |
| nvidia/mit-b2 | 27.3 M          | 0.856           | 43:15        |
| nvidia/mit-b3 | 47.2 M          | 0.890           | 61:00        |
| resnet18      | 14.3 M          | 0.831           | 19:49        |
| resnet34      | 24.4 M          | 0.830           | 21:35        |
| resnet101     | 51.5 M          | 0.843           | 32:53        |

### Configuración de parámetros

Un buen indicador del rendimiento de los modelos es el tiempo. Esto se puede resumir en la cantidad de tiempo que tarda cada modelo en completar una __época__ (epoch). Completar una época implica alimentar con todos los datos al modelo. Había que definir la cantidad de datos que usaría para alimentar los modelos en cada __paso__ (step). Esta cantidad de datos se denomina __lote__ (batch). Como mencionamos, anteriormente, los modelos difieren en la cantidad de cálculos requeridos para procesar los datos, esto implica que la cantidad de datos se verá limitada por la capacidad de __GPU__ a usar. Para este menester dispusimos usar __Colab__, que nos provee de un _Tesla T4_ de 16MB, ese sería nuestro límite de memoria. Los modelos de Transformers más pequeños no tendrían problema; pero si los más grandes. Si bien, no usaríamos el modelo más elaborado para hacer las comparaciones, sería muy elocuente mostrar el potencial general del modelo, así que evaluar los datos con el modelo más grande nos daría el límite de tamaño de lote (en este caso 2). Este se mantendría para evaluar los modelos más pequeños. En el caso de la __UNet__, no tendríamos este problema. Así que usamos el máximo tamaño de lote posible (este sería 64).

|   Model   | Parámetros  |   Val_iou   |  Epoch  |  Batch |  Time (min) |
|:---------:|:-----------:|:-----------:|:-------:|:------:|:------:|
| resnet101 | 14.3 M      | 0.831       |    50   |   64   | 32:53  |
| nvidia/mit-b1 | 13.7 M  | 0.827       |    9    |   2    | 32:53  |

En este punto, el argumento estaba orientado a establecer el máximo rendimiento en cada modelo. No sería justo usar un mismo lote para ambos modelos; pero sí mantener el mismo tamaño del lote para la misma arquitectura. Definir el lote en la __UNet__, no fue muy complicado. no era crítico para el _GPU_ manejar esa cantidad de datos. Este criterio puede ser diferente si se requiere pre-procesar los datos originales. En nuestro caso, el pre-procesado se realizó con anterioridad e implicó un buen ahorro de cálculos, o dicho de otro modo, un inicio más ligero.

Para definir la cantidad de épocas a entrenar, estaba claro que no serían iguales para ambos modelos; pero habría que definir una cantidad fija para cada uno. Decidí usar los modelos más pequeños y establecer un tiempo base para ambos. Después de varios entrenamientos pude establecer un tiempo mínimo de 20 a 22 minutos que correspondía a 50 épocas para la __UNet__ y 9 para el __Transformer__. Esta cantidad de épocas se mantendría para el resto de entrenamientos. Este parámetro sería muy revelador; pero de ninguna manera define las mejores métricas, para el caso del __Transformer__ es mucho menor y de la __UNet__ podría ser hasta tres veces mayor en combinación de otros _hiperparámetros_. El lector debería considerar que el objetivo NO es encontrar los mejores valores de rendimiento. Si ese fuera el objetivo, debería prestar atención a estas aclaraciones y realizar las modificaciones respectivas.

Otro de los aspectos definido, fue la omisión de las __transformaciones__. Esto debería acelerar los resultados, desde el punto de vista de optimizar el proceso de comparación, esto quiere decir que los procesos de entrenamiento tomen los tiempos más reducidos. En este punto es importante indicar que para llegar a estos criterios se ha tenido que realizar varios entrenamientos que demuestren que, el uso de __transformaciones__ adecuadas para este tipo de problema, no alteran los resultados con relación a las comparaciones. En todo caso, la presente investigación quiere definir un marco base para realizar comparaciones mas no un modelo óptimamente entrenado.

Y en contra posición a lo dicho anterior mente, he añadido un __transformador__, uno no especialmente importante para este tipo de problemas. Pero después de contrastarlo con los habituales __transformadores__ como son: _flip_, _rotate_, _transpose_, _gridDistortion_, _gaussianBlur_. No influían negativamente en los resultados; pero si en los tiempos de entrenamientos y en la cantidad de épocas necesarias para encontrar mejores resultados en competencia con la misma arquitectura (Unet), mas no, en términos comparativos, con la otra arquitectura (transformers).

Además, es también importante la omisión del uso de __normalización__, que parece ser una omisión contra intuitiva. Pero luego del entrenamiento, resultó que implicaba un mayor tiempo de entrenamiento para obtener mejores métricas, para el caso de la __UNet__. En este punto debo dejar una interrogación que explique la razón de este comportamiento, el lector deberá investigar más sobre ello.

En resumen, el uso de __transformadores__, deberán ser usados con el fin de encontrar mejores métricas desvinculadas de esta investigación. Son irrelevantes con relación a objetivos comparativos, que es nuestro caso.

La __métrica__ para evaluar nuestros modelos será __intersection over union__ (iou), que es el cociente entre la intercepción de los resultados y el ground truth con la unión de los mismos. Esta __métrica__ es muy habitual para este tipo de problemas de segmentación. 

![imagen](images/iou.png)

Para el desarrollo de esta investigación he usado el código del algoritmo prestado de __Sensio__; pero hay librerías que permiten implementarlas con solo la evocación. La idea general era entender cómo funcionaba el algoritmo en detrimento del tiempo, obviamente. 

```python
def IoU(pr, gt, th=0.5, eps=1e-7):
    pr = torch.sigmoid(pr) > th
    gt = gt > th
    intersection = torch.sum(gt * pr, axis=(-2,-1))
    union = torch.sum(gt, axis=(-2,-1)) + torch.sum(pr, axis=(-2,-1)) - intersection + eps
    ious = (intersection + eps) / union
    return torch.mean(ious).item()
```

Otra de las decisiones que involucró agilizar el proceso de entrenamiento fue el uso de __pytorch-lightning__. Esto permitió acelerar el proceso en casi un __60%__. Ciertamente esto depende de los modelos y datos; pero se ha convertido en el estándar en entornos de producción.

### Los resultados

En una primera mirada, es muy evidente que, los __Transformers__ son más eficientes que la __UNet__; pero hay que considerar que No tienen la misma cantidad de parámetros, si bien no es una diferencia notoria se debe considerar que en los modelos más pequeños la UNet tiene una pequeña ventaja. 

![imagen](images/comparative.png)

En los modelos restantes, los transformers rebasan y mantienen una diferencia considerable.  Además, también hay que resaltar que los modelos basados en __Transformers__ solo requieren un tercio de las épocas para estar por encima de los modelos basados en __UNet__. Este detalle es muy elocuente, podemos determinar que los Transformers obtienen los mismos resultados en un periodo similar de tiempo comparado a los modelos basados en __UNet__. Y luego requieren unas pocas épocas para encontrar mejores métricas.

### Conclusiones
La conclusión más relevante es que estos procedimientos deben reevaluarse para otros modelos y conjunto de datos. El estudio no pretende establecer una regla general. El siguiente paso sería usar otros conjuntos de datos.

El modelo basado en __Transformers__ es el que mejores resultados ha mostrado; pero eso no quiere decir que el modelo de UNet esté descartado para _segmentación_. Es un hecho que es mucho más rápido que el Transformer y esto lo hace particularmente efectivo en la búsqueda de mejores _hiperparámetros_ para mejorar el rendimiento.

No es claro que el modelo basado en __Transformers__ rinda de igual manera con otros conjuntos de datos u otros problemas de segmentación. 

###
###
###
###
