<img width="1536" height="1024" alt="image" src="https://github.com/user-attachments/assets/3c6f6ca8-5a92-4538-900f-9ace952ae61c" />

## INTRODUCCIÓN A CUDA
Autor: Juan Navarrete Guzmán

Instructor: Dr. German Pinedo

 ## Descripción general
Este repositorio contiene una colección de pruebas, ejemplos y experimentos utilizando Numba CUDA para ejecutar cómputo paralelo en GPUs mediante kernels escritos en Python.
Incluye ejemplos fundamentales como:

## Lanzamiento de kernels 1D y 2D 
## Identificación de hilos y bloques
## Sobel Edge Detection en GPU
## Multiplicación de matrices (naive kernel)
## Escalado de matrices a gran tamaño
## Comparaciones de rendimiento CPU vs GPU
## Uso del simulador CUDA (NUMBA_ENABLE_CUDASIM) cuando no hay GPU disponible
## Estos programas demuestran cómo aprovechar la GPU para acelerar tareas intensivas.


OBSERVACION: 

** Numba CUDA permite escribir kernels en Python manteniendo una sintaxis cercana a CUDA C.
** Las operaciones paralelas muestran aceleraciones notables frente al CPU, especialmente Sobel y operaciones matriciales.
** El simulador CUDA facilita el aprendizaje sin requerir una GPU física.
** Estos ejemplos representan una base sólida para avanzar hacia kernels optimizados con memoria compartida, tiling y warp-level programming.
