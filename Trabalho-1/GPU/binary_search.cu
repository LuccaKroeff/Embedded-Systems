#include <stdio.h>
#include <cuda_runtime.h>

__device__ int binary_search_device(int arr[], int low, int high, int key) {
    while (low <= high) {
        int mid = (low + high) / 2;

        if (arr[mid] == key)
            return mid;
        else if (key < arr[mid])
            high = mid - 1;
        else
            low = mid + 1;
    }
    return -1;
}

// Kernel que realiza busca binária para uma única chave
__global__ void binary_search_kernel(int* d_arr, int size, int key, int* d_result) {
    // Só um thread executa
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        d_result[0] = binary_search_device(d_arr, 0, size - 1, key);
    }
}

int main() {
    int h_arr[] = {1, 3, 5, 7, 9, 11, 13, 15, 17, 19};
    int size = sizeof(h_arr) / sizeof(h_arr[0]);
    int key = 7;  // valor a buscar
    int h_result = -1;

    int *d_arr, *d_result;
    cudaMalloc(&d_arr, size * sizeof(int));
    cudaMalloc(&d_result, sizeof(int));

    cudaMemcpy(d_arr, h_arr, size * sizeof(int), cudaMemcpyHostToDevice);

    binary_search_kernel<<<1, 1>>>(d_arr, size, key, d_result);

    cudaMemcpy(&h_result, d_result, sizeof(int), cudaMemcpyDeviceToHost);

    if (h_result != -1) {
        printf("Elemento %d encontrado no índice %d\n", key, h_result);
    } else {
        printf("Elemento %d não encontrado\n", key);
    }

    cudaFree(d_arr);
    cudaFree(d_result);

    return 0;
}
