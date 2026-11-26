# Actividad 4.2 · Empaquetado y despliegue de Faster R-CNN

Este proyecto ejecuta en Docker el pipeline de detección de la Actividad 4.1. La prueba local en CPU simula el despliegue en un dispositivo que ejecuta inferencia cerca de la fuente de datos.

## 1. Instala y abre Docker

En Windows o macOS, instala [Docker Desktop](https://www.docker.com/products/docker-desktop/) y ábrelo. En Windows, activa WSL 2 si el instalador te lo solicita. En Linux puedes usar [Docker Engine](https://docs.docker.com/engine/install/).

Descomprime el ZIP. Abre una terminal en la carpeta que contiene el archivo `Dockerfile`. En Windows puedes abrir la carpeta en el Explorador y escribir `powershell` en la barra de direcciones. Comprueba que Docker está listo con:

```sh
docker info
```

Si aparece un error de conexión con el servidor, abre Docker Desktop y espera a que termine de iniciar.

## 2. Construye la imagen

```sh
docker build -t faster-rcnn-4-2 .
```

El punto indica que Docker debe usar la carpeta actual. La primera construcción necesita internet para descargar Python, las dependencias y los pesos COCO de Faster R-CNN ResNet50 FPN. El archivo `modelo/descargar.py` verifica el hash de los pesos y los guarda dentro de la imagen. Las ejecuciones posteriores pueden usar esos pesos sin internet.

## 3. Ejecuta y guarda la evidencia

El comando abre un contenedor temporal con las dos fotos incluidas. Puedes sustituirlas por tus imágenes en `datos/` y volver a construir la imagen antes de ejecutar la prueba.

**Windows PowerShell**

```powershell
docker run --rm --network none faster-rcnn-4-2 2>&1 | Tee-Object -FilePath evidencia/mi_ejecucion.txt
```

**macOS o Linux**

```sh
docker run --rm --network none faster-rcnn-4-2 2>&1 | tee evidencia/mi_ejecucion.txt
```

Toma una captura donde se vean el comando y los resultados. Registra en tu entrega el equipo, el tiempo medio y las imágenes por segundo. Si cambiaste las fotos o el equipo, identifica esos cambios al comparar mediciones.

## Qué se mide

El programa informa por separado el tiempo de carga del modelo con los pesos ya almacenados. Realiza un calentamiento y tres repeticiones cronometradas. Los tiempos de cada repetición incluyen la conversión a tensor, la predicción y el filtrado de resultados. La lectura de los archivos se hace antes de cronometrar. El rendimiento global divide la cantidad total de imágenes procesadas entre la suma de los tiempos medidos. Se utiliza un umbral de confianza del 70 % y lotes de hasta dos imágenes.

## Pesos del proyecto

La prueba incluida utiliza COCO, los mismos pesos que se usaron en la ejecución guardada de la Actividad 4.1. Si para tu entrega necesitas los pesos ajustados con tus datos, coloca en `modelo/` el `state_dict` del Faster R-CNN ResNet50 FPN y un JSON con las clases en el orden del entrenamiento, con `__background__` primero. Puedes pasar ambos archivos montando la carpeta en el contenedor. En PowerShell reemplaza `$(pwd)` por `${PWD}` en el siguiente comando.

```sh
docker run --rm --network none -v "$(pwd)/modelo:/modelo:ro" faster-rcnn-4-2 --checkpoint /modelo/pesos.pth --clases /modelo/clases.json
```

## Resultados de la prueba en Docker

Ejecuté el contenedor sin conexión de red en Docker Desktop para macOS (Docker Engine 29.9.3), con CPU ARM64 y dos hilos de PyTorch. Usé Faster R-CNN con pesos COCO y un umbral de confianza del 70 %. Después de una pasada de calentamiento, medí tres repeticiones del mismo lote de dos imágenes.

| Medición | Resultado |
| --- | ---: |
| Carga del modelo | 0.467 s |
| Tiempos de las tres repeticiones | 6.299 s, 6.071 s y 6.055 s |
| Tiempo medio por lote | 6.142 s |
| Mediana por lote | 6.071 s |
| Rendimiento global | 0.33 imágenes/s |

El modelo detectó una persona y una corbata en `grace_hopper.jpg` y un perro en `perro.jpg`. En esta prueba, procesar el lote de dos imágenes tardó más que cargar el modelo. Los tiempos corresponden al equipo y a las imágenes de esta prueba.

[Evidencia de mi ejecución en macOS](evidencia/mi_ejecucion.txt)

### Prueba de referencia en Codex Cloud

La ejecución de referencia se hizo en Codex Cloud, con Docker Engine 29.9.2, CPU x86_64 y dos hilos de PyTorch. El contenedor se ejecutó con la red deshabilitada. El [registro de terminal](evidencia/ejecucion_docker.txt) muestra estos resultados.

| Medición | Resultado |
| --- | ---: |
| Carga del modelo | 1.563 s |
| Tiempos de las tres repeticiones | 8.513 s, 8.344 s y 9.146 s |
| Tiempo medio por lote de dos imágenes | 8.668 s |
| Rendimiento global | 0.23 imágenes/s |

El procesamiento de cada lote demoró más que la carga inicial de los pesos. El modelo detectó una persona y una corbata en la primera imagen, y un perro en la segunda. Los tiempos describen esta prueba con dos imágenes y dependen del equipo y del tamaño de las entradas.

## Entrega

Comparte el proyecto con el Dockerfile, la evidencia y un breve análisis de los tiempos. Si la entrega se realiza por GitHub o GitLab, sube estos archivos al repositorio y comparte su enlace. Los pesos COCO se obtienen automáticamente durante la construcción.

Imágenes de ejemplo disponibles en [PyTorch Hub](https://github.com/pytorch/hub/blob/master/images/dog.jpg) y [Matplotlib](https://github.com/matplotlib/matplotlib/blob/v3.10.8/lib/matplotlib/mpl-data/sample_data/grace_hopper.jpg).
