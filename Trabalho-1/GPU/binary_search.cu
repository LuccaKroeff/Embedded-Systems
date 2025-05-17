#include <stdio.h>
#include <cuda_runtime.h>

#define FREQ_GPU_HZ 2.5e9  // Frequência da GPU em Hz

__device__ int binary_search_device(int arr[], int low, int high, int key) {
    while (low <= high) {
        int mid = (low + high) / 2;
        if (arr[mid] == key) return mid;
        else if (key < arr[mid]) high = mid - 1;
        else low = mid + 1;
    }
    return -1;
}

__global__ void binary_search_kernel(int* d_arr, int size, int key, int* d_result, unsigned long long* d_cycles) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        unsigned long long start = clock64();
        d_result[0] = binary_search_device(d_arr, 0, size - 1, key);
        unsigned long long end = clock64();
        d_cycles[0] = end - start;
    }
}

int main() {
    int h_arr[] = {1, 3, 5, 7, 9, 11, 13, 15, 17, 19};
    int size = sizeof(h_arr) / sizeof(h_arr[0]);
    int key = 7;
    int h_result = -1;
    unsigned long long h_cycles = 0;

    int *d_arr, *d_result;
    unsigned long long* d_cycles;

    cudaMalloc(&d_arr, size * sizeof(int));
    cudaMalloc(&d_result, sizeof(int));
    cudaMalloc(&d_cycles, sizeof(unsigned long long));

    cudaMemcpy(d_arr, h_arr, size * sizeof(int), cudaMemcpyHostToDevice);

    binary_search_kernel<<<1, 1>>>(d_arr, size, key, d_result, d_cycles);
    cudaDeviceSynchronize();

    cudaMemcpy(&h_result, d_result, sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(&h_cycles, d_cycles, sizeof(unsigned long long), cudaMemcpyDeviceToHost);

    if (h_result != -1)
        printf("Elemento %d encontrado no índice %d\n", key, h_result);
    else
        printf("Elemento %d não encontrado\n", key);

    double elapsed_seconds = (double)h_cycles / FREQ_GPU_HZ;
    printf("Ciclos de GPU: %llu\n", h_cycles);
    printf("Tempo estimado (ciclos / FREQ_GPU_HZ): %.9f segundos\n", elapsed_seconds);

    cudaFree(d_arr);
    cudaFree(d_result);
    cudaFree(d_cycles);

    return 0;
}
