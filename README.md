# Clasificación de imágenes con redes neuronales convolucionales

Proyecto de clasificación de imágenes mediante redes neuronales convolucionales (CNN) y Transfer Learning. El trabajo se desarrolla en el notebook [`neural-network.ipynb`](./neural-network.ipynb), donde se prepara el dataset, se entrenan varios modelos, se evalúan sus resultados y se comparan distintas decisiones de arquitectura e hiperparámetros.

## Autores

- [Pablo Llopis Parrilla](https://github.com/Putrici0)

## Descripción del proyecto

El objetivo del proyecto es resolver un problema de clasificación multiclase de imágenes de escenas naturales. Para ello se trabaja con el dataset **Intel Image Classification**, descargado automáticamente desde Kaggle mediante `kagglehub`.

El notebook compara diferentes aproximaciones:

- Una CNN base construida desde cero.
- La misma CNN con un learning rate menor.
- CNN con regularización mediante Dropout.
- CNN con menor capacidad para estudiar underfitting.
- CNN más profunda para estudiar el efecto de aumentar la complejidad.
- Transfer Learning con ResNet18 preentrenada en ImageNet.

El trabajo no se limita a entrenar un único modelo: se realiza una comparación experimental entre arquitecturas, se analizan las curvas de pérdida y precisión, se interpretan matrices de confusión y se extraen conclusiones sobre overfitting, underfitting, regularización y transferencia de aprendizaje.

## Dataset

Se utiliza el dataset:

```text
puneet6060/intel-image-classification
```

El conjunto contiene imágenes de escenas naturales divididas en seis clases:

- `buildings`
- `forest`
- `glacier`
- `mountain`
- `sea`
- `street`

El notebook descarga el dataset con `kagglehub` y construye un subconjunto de trabajo en `/content/intel_dataset`.

### División de datos

Para cada clase se seleccionan imágenes del conjunto de entrenamiento original y se reparten así:

| Split | Imágenes por clase | Total aproximado |
| --- | ---: | ---: |
| Train | 1000 | 6000 |
| Validation | 200 | 1200 |
| Test | Conjunto de test original | Depende del dataset original |

El conjunto de test se copia completo desde la partición `seg_test` del dataset original.

## Tecnologías utilizadas

El proyecto está implementado en Python y usa las siguientes librerías principales:

- `torch`
- `torchvision`
- `numpy`
- `pandas`
- `matplotlib`
- `scikit-learn`
- `kagglehub`

También se utilizan componentes de PyTorch como:

- `torch.nn`
- `torch.optim`
- `torch.utils.data.DataLoader`
- `torchvision.datasets.ImageFolder`
- `torchvision.transforms`
- `torchvision.models`

## Configuración del entrenamiento

El notebook fija una semilla para mejorar la reproducibilidad:

```python
SEED = 42
```

Parámetros generales:

| Parámetro | Valor |
| --- | ---: |
| Épocas | 10 |
| Batch size | 32 |
| Tamaño de imagen CNN propias | 128 x 128 |
| Tamaño de imagen Transfer Learning | 224 x 224 |
| Optimizador | Adam |
| Función de pérdida | CrossEntropyLoss |
| Dispositivo | CUDA si está disponible, si no CPU |

## Preprocesamiento y data augmentation

Para los modelos CNN propios se redimensionan las imágenes a `128 x 128`.

Transformaciones usadas en entrenamiento:

- Redimensionado.
- Volteo horizontal aleatorio.
- Rotación aleatoria de hasta 10 grados.
- Conversión a tensor.
- Normalización con medias y desviaciones típicas de ImageNet.

Transformaciones usadas en validación y test:

- Redimensionado.
- Conversión a tensor.
- Normalización.

Para el modelo con Transfer Learning se usa `224 x 224`, ya que ResNet18 espera imágenes de ese tamaño en su configuración habitual.

## Flujo de trabajo del notebook

El notebook sigue esta estructura:

1. Instalación e importación de dependencias.
2. Configuración de semilla, dispositivo, épocas, batch size y tamaño de imagen.
3. Descarga del dataset desde Kaggle.
4. Filtrado y organización del dataset en `train`, `validation` y `test`.
5. Comprobación del número de imágenes por clase.
6. Definición de transformaciones.
7. Creación de datasets y dataloaders.
8. Visualización de un batch de imágenes.
9. Definición de funciones reutilizables de entrenamiento, graficado y evaluación.
10. Entrenamiento y evaluación de siete modelos.
11. Comparación final de resultados.
12. Conclusiones generales.

## Funciones principales

### `train_model`

Entrena un modelo durante un número determinado de épocas. En cada época calcula:

- Pérdida de entrenamiento.
- Accuracy de entrenamiento.
- Pérdida de validación.
- Accuracy de validación.

Devuelve el modelo entrenado y un diccionario `history` con las métricas registradas.

### `plot_history`

Genera dos gráficas:

- Evolución de la pérdida en entrenamiento y validación.
- Evolución de la accuracy en entrenamiento y validación.

Estas gráficas permiten detectar comportamientos como overfitting, underfitting o convergencia inestable.

### `evaluate_model`

Evalúa el modelo sobre un `DataLoader` y muestra:

- Accuracy final.
- Matriz de confusión.
- Classification report de `scikit-learn`, con precisión, recall y F1-score por clase.

## Modelos entrenados

### Modelo 1 - CNN Base

Arquitectura inicial construida desde cero:

- 3 capas convolucionales:
  - 32 filtros.
  - 64 filtros.
  - 128 filtros.
- MaxPooling después de cada convolución.
- Capa fully connected de 512 neuronas.
- Capa de salida con 6 clases.
- Learning rate: `0.001`.
- Sin Dropout ni Batch Normalization.

Resultado:

| Métrica | Valor |
| --- | ---: |
| Accuracy de validación | 80.25% |

Análisis:

El modelo aprende correctamente durante las primeras épocas, pero a partir de aproximadamente la quinta época aparece overfitting. La pérdida de entrenamiento sigue bajando mientras que la de validación empieza a subir. La accuracy de entrenamiento alcanza valores cercanos al 90%, pero la de validación se estabiliza alrededor del 80%.

Las principales confusiones aparecen entre `mountain` y `glacier`, y entre `street` y `buildings`.

### Modelo 2 - CNN Base con learning rate menor

Se mantiene la arquitectura del Modelo 1, pero se reduce el learning rate:

```python
lr = 0.0005
```

Resultado:

| Métrica | Valor |
| --- | ---: |
| Accuracy de validación | 82.17% |

Análisis:

La reducción del learning rate produce una convergencia más estable. La pérdida disminuye de forma más suave y la accuracy mejora progresivamente con menos oscilaciones. Este modelo mejora en torno a dos puntos porcentuales respecto al modelo base.

### Modelo 3 - CNN con Dropout 0.2

Se añade una capa Dropout después de la capa fully connected:

```python
nn.Dropout(0.2)
```

Resultado:

| Métrica | Valor |
| --- | ---: |
| Accuracy de validación | 82.08% |

Análisis:

El Dropout moderado reduce parcialmente el overfitting. La diferencia entre entrenamiento y validación es menor que en el modelo base y la pérdida de validación aumenta más lentamente. El rendimiento es similar al Modelo 2.

### Modelo 4 - CNN con Dropout 0.5

Se incrementa la regularización:

```python
nn.Dropout(0.5)
```

Resultado:

| Métrica | Valor |
| --- | ---: |
| Accuracy de validación | 80.08% |

Análisis:

El Dropout de 0.5 resulta demasiado agresivo para esta arquitectura. Aunque reduce el overfitting, también dificulta el aprendizaje. La accuracy de entrenamiento crece más lentamente y el rendimiento final baja respecto a los modelos con mejor equilibrio.

### Modelo 5 - CNN Sencilla

Se reduce la capacidad del modelo:

- Convoluciones con 16, 32 y 64 filtros.
- Capa fully connected de 256 neuronas.

Resultado:

| Métrica | Valor |
| --- | ---: |
| Accuracy de validación | 78.58% |

Análisis:

La red tiene menos parámetros y menor capacidad de representación. Esto provoca una pérdida de rendimiento y síntomas de underfitting. Aumentan los errores especialmente en `buildings`, `mountain` y `sea`.

### Modelo 6 - CNN Profunda

Se aumenta la profundidad de la arquitectura:

- 4 capas convolucionales:
  - 32 filtros.
  - 64 filtros.
  - 128 filtros.
  - 256 filtros.
- Capa fully connected de 512 neuronas.
- Capa fully connected adicional de 256 neuronas.
- Capa de salida con 6 clases.

Resultado:

| Métrica | Valor |
| --- | ---: |
| Accuracy de validación | 80.92% |

Análisis:

El aumento de profundidad no aporta una mejora significativa. La precisión de entrenamiento crece, pero la validación se estabiliza alrededor del 81%. La complejidad adicional no mejora claramente la generalización.

### Modelo 7 - Transfer Learning con ResNet18

Se utiliza una ResNet18 preentrenada en ImageNet:

```python
model7 = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
```

Todas las capas se congelan:

```python
for param in model7.parameters():
    param.requires_grad = False
```

Después se sustituye la capa final para adaptarla a las seis clases del problema:

```python
model7.fc = nn.Linear(model7.fc.in_features, num_classes)
```

Solo se entrena la capa final:

```python
optimizer = optim.Adam(model7.fc.parameters(), lr=0.001)
```

Resultados:

| Métrica | Valor |
| --- | ---: |
| Accuracy de validación | 88.92% |
| Accuracy de test | 89.00% |

Análisis:

Transfer Learning ofrece el mejor rendimiento del proyecto. Las curvas de entrenamiento y validación se mantienen cercanas, la pérdida disminuye rápidamente y no aparecen síntomas importantes de overfitting. El modelo aprovecha características visuales aprendidas previamente en ImageNet, lo que mejora claramente la clasificación frente a las CNN entrenadas desde cero.

## Comparación de resultados

| Posición | Modelo | Accuracy de validación |
| ---: | --- | ---: |
| 1 | Modelo 7 - Transfer Learning ResNet18 | 88.92% |
| 2 | Modelo 2 - CNN Base con LR 0.0005 | 82.17% |
| 3 | Modelo 3 - CNN con Dropout 0.2 | 82.08% |
| 4 | Modelo 6 - CNN Profunda | 80.92% |
| 5 | Modelo 1 - CNN Base | 80.25% |
| 6 | Modelo 4 - CNN con Dropout 0.5 | 80.08% |
| 7 | Modelo 5 - CNN Sencilla | 78.58% |

## Interpretación de los resultados

Las CNN diseñadas desde cero consiguen resultados razonables, pero presentan limitaciones. El modelo base muestra overfitting a partir de la mitad del entrenamiento. Reducir el learning rate mejora la estabilidad y produce la mejor CNN propia. Añadir Dropout moderado también ayuda a controlar el overfitting, aunque un Dropout demasiado alto limita la capacidad de aprendizaje.

La CNN sencilla tiene menos parámetros y no alcanza suficiente capacidad representativa, por lo que obtiene el peor rendimiento. La CNN más profunda no mejora de forma clara, lo que indica que aumentar capas no garantiza mejores resultados si no se acompaña de una arquitectura más adecuada o de más datos.

El mejor enfoque es Transfer Learning con ResNet18, que supera ampliamente a los modelos entrenados desde cero. Esto confirma que reutilizar una red preentrenada es especialmente eficaz en problemas de visión artificial con datasets de tamaño medio.

## Errores más frecuentes

En las matrices de confusión se observa que los errores principales se concentran en clases visualmente parecidas:

- `glacier` y `mountain`: ambas pueden contener nieve, roca, relieve y paisajes naturales similares.
- `buildings` y `street`: las escenas urbanas comparten elementos como carreteras, edificios, aceras y estructuras artificiales.

Las clases `forest`, `sea` y `street` suelen obtener buenos resultados, especialmente con ResNet18.

## Cómo ejecutar el proyecto

### Opción recomendada: Google Colab

El notebook está preparado para ejecutarse fácilmente en Google Colab, ya que utiliza rutas como `/content/intel_dataset`.

Pasos:

1. Abrir `neural-network.ipynb` en Google Colab.
2. Ejecutar las celdas en orden.
3. El notebook instalará `kagglehub` si es necesario.
4. Se descargará automáticamente el dataset.
5. Se prepararán los splits de datos.
6. Se entrenarán y evaluarán los modelos.

### Ejecución local

También puede ejecutarse localmente si se dispone de Python y Jupyter.

Instalación de dependencias:

```bash
pip install kagglehub numpy pandas matplotlib scikit-learn torch torchvision
```

Ejecución:

```bash
jupyter notebook neural-network.ipynb
```

Si se ejecuta fuera de Colab, puede ser necesario modificar la ruta:

```python
filtered_dataset_dir = "/content/intel_dataset"
```

por una ruta local, por ejemplo:

```python
filtered_dataset_dir = "./intel_dataset"
```

## Requisitos recomendados

- Python 3.10 o superior.
- Jupyter Notebook o Google Colab.
- GPU compatible con CUDA para acelerar el entrenamiento.
- Conexión a internet para descargar el dataset y los pesos preentrenados de ResNet18.

El proyecto puede ejecutarse en CPU, pero el entrenamiento será más lento.

## Estructura del repositorio

```text
fsi-neural-network/
|-- neural-network.ipynb
`-- README.md
```

## Conclusiones generales

El proyecto demuestra de forma experimental varios conceptos importantes en deep learning aplicado a visión artificial:

- Una CNN base puede aprender patrones útiles, pero tiende a sobreajustar.
- Reducir el learning rate puede mejorar la estabilidad del entrenamiento.
- El Dropout ayuda a regularizar, pero un valor excesivo puede provocar underfitting.
- Reducir demasiado la arquitectura limita la capacidad de representación.
- Aumentar profundidad no garantiza mejores resultados.
- Transfer Learning con una red preentrenada ofrece el mejor rendimiento del trabajo.

El mejor modelo final es **ResNet18 con Transfer Learning**, con una accuracy de validación del **88.92%** y una accuracy de test del **89.00%**.
