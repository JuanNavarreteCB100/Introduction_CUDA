## ECU01
Autor: Juan Navarrete Guzmán
## introducción a CUDA

```` Python

#CUDA Steps:
#Initializin data from CPU
#Transfer from CPU to GPU
# Run Kernel with defined Grid/Block size (Threads)
#Transfer results from GPU to CPU
#Clear memory

# CUDA kernel Device
@cuda.jit
def first_kernel(a, result):
    idx = cuda.grid(1)
    if idx < a.size:
        result[idx] = a[idx]
#Host
def main():
    # 1. Initialize data on CPU
    N = 10_000_000
    a_cpu = np.arange(N, dtype=np.float32)

    # -------------------------------
    # CPU computation
    # -------------------------------
    # Transfer to GPU
    start = time.time()
    result_cpu = a_cpu
    cpu_time = time.time() -start
    print(f"CPU time: {cpu_time * 1e3:.2f} ms")

    #---------------------------------------
    # GPU computation
    #--------------------------------------
    # 2. Transfer from CPU to GPU
    start = time.time()
    a_gpu = cuda.to_device(a_cpu)
    result_gpu = cuda.device_array_like(a_cpu) #Reserva Memoria
    transfer_in_time = time.time() - start

    # Kernel launch
    threads_per_block = 128
    blocks_per_grid = (N + threads_per_block - 1) // threads_per_block
    start = time.time()
    first_kernel[blocks_per_grid, threads_per_block](a_gpu, result_gpu) #launch Kernel
    cuda.synchronize()
    kernel_time = time.time() - start

    #Copy back
    start = time.time()
    result_from_gpu = result_gpu.copy_to_host()
    cuda.synchronize()
    transfer_out_time = time.time() -start

    # Report
    print(f"GPU transfer to device: {transfer_in_time * 1e3:.2f} ms")
    print(f"GPU kernel execution: {kernel_time * 1e3:.2f} ms")
    print(f"GPU transfer to host: {transfer_out_time * 1e3:.2f} ms")
    print(f"Total GPU time: {(transfer_in_time + kernel_time + transfer_out_time) * 1e3:.2f} ms")

    # Cleanup
    del a_gpu, result_gpu
    cuda.close()

````

## Imagen de la terminal de Colab

<img width="362" height="226" alt="image" src="https://github.com/user-attachments/assets/66a27329-946b-494f-9c71-a152ff4aa0a9" />

## Conclusión del ECU01

El ejercicio demuestra el ciclo básico de cómputo en GPU usando Numba CUDA:

Las transferencias de memoria suelen ser más costosas que el kernel.

Para tareas grandes (millones de elementos), el uso de GPU representa una aceleración significativa.

Este ECU establece la base para kernels más complejos como Sobel, MatMul o Grid 2D.


  

if __name__ == "__main__":
    main()
