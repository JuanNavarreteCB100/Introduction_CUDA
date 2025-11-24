## ECU03

Autor: Juan Navarrete Guzmán

## Introducción

En el ECU03 se explora el uso de CUDA para acelerar operaciones de procesamiento de imágenes mediante el filtro de bordes Sobel, comparando el rendimiento entre GPU y CPU.
Esta práctica permite comprender cómo aprovechar el paralelismo masivo de la GPU para tareas intensivas en cómputo, como el cálculo de gradientes en imágenes de alta resolución.
A diferencia de los ejercicios previos, aquí se trabaja con una imagen real, arreglos 2D y un kernel más complejo que opera sobre vecinos de cada píxel. 

Este ECU introduce el uso de grids y bloques bidimensionales, acceso a vecinos, y la comparación entre modelos de ejecución CPU–GPU.

## Descripción del ejercicio

El ejercicio consistio en implementar un kernel CUDA que aplique el operador Sobel a una imagen en escala de grises, calculando los gradientes horizontales (Gx) y verticales (Gy) para obtener un mapa de bordes.

El programa ejecuta los siguientes pasos:

1. Carga de una imagen 4K desde internet y conversión a matriz NumPy.
2. Transferencia de datos a la GPU mediante cuda.to_device.
3. Ejecución del kernel Sobel, donde cada hilo procesa un píxel usando cuda.grid(2) para indexar matrices 2D.
4. Sincronización y copia de resultados de vuelta al CPU.
5. Comparación contra la implementación Sobel de OpenCV en CPU, verificando exactitud numérica.
6. Medición de tiempos y cálculo del speedup, demostrando la eficiencia del paralelismo en GPU.
7. Visualización de imagen original, bordes en GPU y bordes en CPU.

