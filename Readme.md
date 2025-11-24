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

   ## configuración requerida

<img width="392" height="128" alt="image" src="https://github.com/user-attachments/assets/ede9694f-c43d-4966-9eeb-5d109af9b9f8" />

# ex1_vector_add.py
## Código 1: vector_add_kernel
```python
import numpy as np
from numba import cuda
import math
import time

@cuda.jit
def vector_add_kernel(a, b, c):
    """
    Each thread computes one element: c[i] = a[i] + b[i]
    """
    # Compute global thread index
    idx = cuda.grid(1)

    # Boundary check
    if idx < c.size:
        c[idx] = a[idx] + b[idx]

def main():
    N_large = 10_000_000
    a = np.random.randn(N_large).astype(np.float32)
    b = np.random.randn(N_large).astype(np.float32)
    c = np.zeros(N_large, dtype=np.float32)

    d_a = cuda.to_device(a)
    d_b = cuda.to_device(b)
    d_c = cuda.to_device(c)

    threads_per_block = 256
    blocks_per_grid = math.ceil(N_large / threads_per_block)

    # Warmup
    vector_add_kernel[blocks_per_grid, threads_per_block](d_a, d_b, d_c)
    cuda.synchronize()

    # GPU timing
    start = time.time()
    vector_add_kernel[blocks_per_grid, threads_per_block](d_a, d_b, d_c)
    cuda.synchronize()
    gpu_time = (time.time() - start) * 1000

    result = d_c.copy_to_host()

    # CPU timing
    cpu_start = time.time()
    expected = a + b
    cpu_time = (time.time() - cpu_start) * 1000

    print(f"GPU kernel time: {gpu_time:.3f} ms")
    print(f"CPU NumPy time: {cpu_time:.3f} ms")
    print(f"Speedup: {cpu_time / gpu_time:.2f}x")
    print("Correct:", np.allclose(result, expected))

if __name__ == "__main__":
    main()
```
## pantalla del resultado

<img width="135" height="39" alt="image" src="https://github.com/user-attachments/assets/8f5023bc-5ad0-4053-b812-c5afc16f37f2" />


## Código 2: dummy_compute_kernel
```python
import numpy as np
from numba import cuda
import math
import time

@numba.cuda.jit
def dummy_compute_kernel(a, b, c):
    """
    Simple compute to measure timing: c[i] = sqrt(a[i]^2 + b[i]^2)
    """
    idx = cuda.grid(1)
    if idx < c.size:
        c[idx] = math.sqrt(a[idx]**2 + b[idx]**2)

def main():
    N = 10_000_000   # 1M elements
    a = np.random.randn(N).astype(np.float32)
    b = np.random.randn(N).astype(np.float32)
    c = np.zeros(N, dtype=np.float32)

    # Device arrays
    d_a = cuda.to_device(a)
    d_b = cuda.to_device(b)
    d_c = cuda.to_device(c)

    threads_per_block = 256
    blocks_per_grid = math.ceil(N / threads_per_block)
    # Warmup (first launch can be slower due to JIT compilation)
    dummy_compute_kernel[blocks_per_grid, threads_per_block](d_a, d_b, d_c)
    cuda.synchronize()

    # Timed run
    start = time.time()
    dummy_compute_kernel[blocks_per_grid, threads_per_block](d_a, d_b, d_c)
    cuda.synchronize()      # IMPORTANT: wait for kernel to finish
    end = time.time()

    gpu_time = (end - start) * 1000   # convert to ms

    result = d_c.copy_to_host()

    # CPU reference
    cpu_start = time.time()
    expected = np.sqrt(a**2 + b**2)
    cpu_end = time.time()
    cpu_time = (cpu_end - cpu_start) * 1000

    print(f"GPU time: {gpu_time:.3f} ms")
    print(f"CPU time: {cpu_time:.3f} ms")
    print(f"Speedup: {cpu_time / gpu_time:.2f}x")
    print("Correct:", np.allclose(result, expected))

if __name__ == "__main__":
    main()
````
## Pantalla del resultado

<img width="136" height="35" alt="image" src="https://github.com/user-attachments/assets/cf1a85ec-2ed1-4f61-bbb0-472bffc712d9" />

## Código 3 matrix_scale

```python
import numpy as np
from numba import cuda
import math
import time

@numba.cuda.jit
def matrix_scale_kernel(mat, scalar, out):
    """
    Scale every element: out[row, col] = mat[row, col] * scalar
    """
    row, col = cuda.grid(2)

    if row < out.shape[0] and col < out.shape[1]:
        out[row, col] = mat[row, col] * scalar


def main():
    rows_large, cols_large = 4096, 4096
    mat = np.random.randn(rows_large, cols_large).astype(np.float32)
    out = np.zeros_like(mat)
    scalar = 2.5
    d_mat = cuda.to_device(mat)
    d_out = cuda.to_device(out)

    threads_per_block = (16, 16)
    blocks_per_grid_x = math.ceil(rows_large / threads_per_block[0])
    blocks_per_grid_y = math.ceil(cols_large / threads_per_block[1])
    blocks_per_grid = (blocks_per_grid_x, blocks_per_grid_y)

    # Warmup
    matrix_scale_kernel[blocks_per_grid, threads_per_block](d_mat, scalar, d_out)
    cuda.synchronize()

    # GPU timing
    start = time.time()
    matrix_scale_kernel[blocks_per_grid, threads_per_block](d_mat, scalar, d_out)
    cuda.synchronize()
    gpu_time = (time.time() - start) * 1000

    result = d_out.copy_to_host()

    # CPU timing
    cpu_start = time.time()
    expected = mat * scalar
    cpu_time = (time.time() - cpu_start) * 1000

    print(f"GPU kernel time: {gpu_time:.3f} ms")
    print(f"CPU NumPy time: {cpu_time:.3f} ms")
    print(f"Speedup: {cpu_time / gpu_time:.2f}x")
    print("Correct:", np.allclose(result, expected))

if __name__ == "__main__":
    main()
```
#Resultado en pantalla

<img width="133" height="37" alt="image" src="https://github.com/user-attachments/assets/b465acf2-b422-4bdd-a0ce-a5180599a747" />

## Código 4 matmul_naive_kernel

```python

import numpy as np
from numba import cuda
import math
import time

@numba.cuda.jit
def matmul_naive_kernel(A, B, C):
    """
    Naive matrix multiply: C = A @ B
    Each thread computes one element of C.
    """

    row, col = cuda.grid(2)

    M, K = A.shape
    K2, N = B.shape

    if row < M and col < N:
        total = 0.0
        for k in range(K):
            total += A[row, k] * B[k, col]
        C[row, col] = total

def main():
    M, K, N = 1000, 1000, 1000
    A = np.random.randn(M, K).astype(np.float32)
    B = np.random.randn(K, N).astype(np.float32)
    C = np.zeros((M, N), dtype=np.float32)

    threads_per_block = (16, 16)
    d_A = cuda.to_device(A)
    d_B = cuda.to_device(B)
    d_C = cuda.to_device(C)

    blocks_per_grid_x = (M + threads_per_block[0] - 1) // threads_per_block[0]
    blocks_per_grid_y = (N + threads_per_block[1] - 1) // threads_per_block[1]
    blocks_per_grid = (blocks_per_grid_x, blocks_per_grid_y)

    # Warmup
    matmul_naive_kernel[blocks_per_grid, threads_per_block](d_A, d_B, d_C)
    cuda.synchronize()

    # GPU timing
    start = time.time()
    matmul_naive_kernel[blocks_per_grid, threads_per_block](d_A, d_B, d_C)
    cuda.synchronize()
    gpu_time = (time.time() - start) * 1000

    C_gpu = d_C.copy_to_host()

    # CPU timing
    cpu_start = time.time()
    C_cpu = A @ B
    cpu_time = (time.time() - cpu_start) * 1000

    print(f"GPU kernel time: {gpu_time:.4f} ms")
    print(f"CPU NumPy time: {cpu_time:.4f} ms")
    print(f"Speedup: {cpu_time / gpu_time:.2f}x")
    print(f"Correct: {np.allclose(C_gpu, C_cpu, atol=1e-3)}")

main()
````

##Pantalla del resultado
<img width="131" height="41" alt="image" src="https://github.com/user-attachments/assets/958cf130-5a8e-4c52-a0a1-84c7a3a85f46" />

## Código 5:sobel_kernel

```python
import numpy as np
import numba.cuda as cuda
import time
import urllib.request
from PIL import Image
from matplotlib import pyplot as plt
import cv2

import numpy as np
import numba.cuda as cuda
import time
import urllib.request
from PIL import Image
import cv2
from matplotlib import pyplot as plt


@numba.cuda.jit
def sobel_kernel(img, out):
    """Apply Sobel edge detection - each thread processes one pixel"""
    row, col = cuda.grid(2)
    H, W = img.shape

    if 0 < row < H-1 and 0 < col < W-1:

        # Horizontal gradient (Gx)
        gx = ( -img[row-1, col-1] + img[row-1, col+1]
               -2*img[row, col-1] + 2*img[row, col+1]
               -img[row+1, col-1] + img[row+1, col+1] )

        # Vertical gradient (Gy)
        gy = ( -img[row-1, col-1] - 2*img[row-1, col] - img[row-1, col+1]
               + img[row+1, col-1] + 2*img[row+1, col] + img[row+1, col+1] )

        # Edge magnitude
        out[row, col] = (gx*gx + gy*gy)**0.5


def sobel_opencv(img):
    """OpenCV CPU version using Sobel"""
    gx = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=3)
    return np.sqrt(gx**2 + gy**2)


# Load 4K image from internet
urllib.request.urlretrieve("https://picsum.photos/3840/2160", "image.jpg")
img = Image.open("image.jpg").convert('L')   # Convert to grayscale
img = np.array(img, dtype=np.float32)

H, W = img.shape
print(f"Image: {W}×{H} ({W*H:,} pixels)")


d_img = cuda.to_device(img)
d_out = cuda.to_device(np.zeros_like(img))

threads = (32, 32)
blocks = ((W + 15) // 16, (H + 15) // 16)

print(f"Grid: {blocks} blocks × {threads} threads")


# Warmup
sobel_kernel[blocks, threads](d_img, d_out)
cuda.synchronize()


# Timed run (GPU)
start = time.time()
sobel_kernel[blocks, threads](d_img, d_out)
cuda.synchronize()
gpu_time = (time.time() - start) * 1000

out_gpu = d_out.copy_to_host()


# CPU Sobel timing
start = time.time()
out_cpu = sobel_opencv(img)
cpu_time = (time.time() - start) * 1000


# Results
print("\n" + "="*60)
print("Results")
print("="*60)
print(f"GPU: {gpu_time:.2f} ms")
print(f"CPU: {cpu_time:.2f} ms")
print(f"Speedup: {cpu_time/gpu_time:.1f}x")
print(f"Correct: {np.allclose(out_gpu, out_cpu, atol=1e-3)}")


# Resize for display
H, W = img.shape
target_w = 256
target_h = int(target_w * H / W)


def resize_for_plot(array):
    normalized = (array / array.max() * 255).astype(np.uint8)
    return np.array(Image.fromarray(normalized).resize((target_w, target_h), Image.LANCZOS))


plt.figure(figsize=(20, 10))

plt.subplot(1, 3, 1)
plt.imshow(resize_for_plot(img), cmap='gray')
plt.title('Original Image')
plt.axis('off')

plt.subplot(1, 3, 2)
plt.imshow(resize_for_plot(out_gpu), cmap='gray')
plt.title('GPU Sobel Edges')
plt.axis('off')

plt.subplot(1, 3, 3)
plt.imshow(resize_for_plot(out_cpu), cmap='gray')
plt.title('OpenCV CPU Sobel Edges')
plt.axis('off')

plt.tight_layout()
plt.show()
````

## Resultado de la pantalla
<img width="755" height="159" alt="image" src="https://github.com/user-attachments/assets/3b52beea-7494-483c-97c7-ecad75a20569" />

## Conclusiones

El conjunto de los cinco ejercicios constituye una secuencia de aprendizaje progresivo que permite comprender de manera estructurada el modelo de cómputo paralelo de CUDA utilizando Numba. 
En el primer ejercicio, se introduce el flujo esencial de ejecución en GPU mediante una suma de vectores, mostrando cómo se transfieren datos, cómo se configuran los hilos y cómo se ejecuta un kernel simple. 
El segundo ejercicio incrementa la complejidad al incorporar operaciones matemáticas más complejas, evidenciando cómo la GPU ofrece beneficios significativos en tareas de cómputo intensivo. 
Posteriormente, el ejercicio de escalado de matrices presenta el uso de grids y bloques en dos dimensiones, un paso fundamental para trabajar con imágenes o matrices de gran tamaño. 
El cuarto ejercicio, correspondiente a la multiplicación de matrices naive, pone de manifiesto las limitaciones de los kernels no optimizados y la importancia del acceso eficiente a la memoria para mejorar el rendimiento.
Finalmente, el ejercicio del filtro Sobel representa el caso más completo: procesa una imagen real de gran resolución, usa vecinos dentro de la matriz, trabaja con un kernel 2D y permite comparar directamente el desempeño entre CPU y GPU. 

En conjunto, estas prácticas brindan una base sólida para avanzar hacia técnicas más avanzadas como tiling y memoria compartida.






Resopopo
