#include <stdio.h>
#include <cuda_runtime.h>

#define WIDTH 5
#define HEIGHT 5
#define KERNEL_SIZE 3

// Substitua pelo valor real da sua GPU (em Hz). Ex: 1.5 GHz = 1.5e9
#define FREQ_GPU_HZ 2.5e9

// Kernel CUDA para convolução 2D com medição de tempo por thread (usando clock64)
__global__ void convolve2D_cuda(
    unsigned char* input,
    float* kernel,
    unsigned char* output,
    int width, int height, int kernel_size,
    unsigned long long* d_cycles
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int idx = y * width + x;
    int pad = kernel_size / 2;

    if (x < width && y < height) {
        unsigned long long start = clock64();

        float sum = 0.0f;
        for (int ky = 0; ky < kernel_size; ky++) {
            for (int kx = 0; kx < kernel_size; kx++) {
                int ix = x + kx - pad;
                int iy = y + ky - pad;
                if (ix >= 0 && ix < width && iy >= 0 && iy < height) {
                    sum += input[iy * width + ix] * kernel[ky * kernel_size + kx];
                }
            }
        }

        if (sum < 0) sum = 0;
        if (sum > 255) sum = 255;

        output[y * width + x] = (unsigned char)sum;

        unsigned long long end = clock64();
        d_cycles[idx] = end - start;
    }
}

int main() {
    unsigned char image[HEIGHT][WIDTH] = {
        { 10, 50, 80, 50, 10 },
        { 60, 120, 200, 120, 60 },
        { 90, 180, 255, 180, 90 },
        { 60, 120, 200, 120, 60 },
        { 10, 50, 80, 50, 10 }
    };

    float kernel[KERNEL_SIZE][KERNEL_SIZE] = {
        { -1, -1, -1 },
        { -1,  8, -1 },
        { -1, -1, -1 }
    };

    unsigned char output[HEIGHT][WIDTH];

    unsigned char *d_input, *d_output;
    float *d_kernel;
    unsigned long long* d_cycles;

    int num_pixels = WIDTH * HEIGHT;

    cudaMalloc(&d_input, num_pixels * sizeof(unsigned char));
    cudaMalloc(&d_kernel, KERNEL_SIZE * KERNEL_SIZE * sizeof(float));
    cudaMalloc(&d_output, num_pixels * sizeof(unsigned char));
    cudaMalloc(&d_cycles, num_pixels * sizeof(unsigned long long));

    cudaMemcpy(d_input, image, num_pixels * sizeof(unsigned char), cudaMemcpyHostToDevice);
    cudaMemcpy(d_kernel, kernel, KERNEL_SIZE * KERNEL_SIZE * sizeof(float), cudaMemcpyHostToDevice);

    dim3 blockDim(16, 16);
    dim3 gridDim((WIDTH + blockDim.x - 1) / blockDim.x, (HEIGHT + blockDim.y - 1) / blockDim.y);

    convolve2D_cuda<<<gridDim, blockDim>>>(
        d_input, d_kernel, d_output, WIDTH, HEIGHT, KERNEL_SIZE, d_cycles
    );
    cudaDeviceSynchronize();

    cudaMemcpy(output, d_output, num_pixels * sizeof(unsigned char), cudaMemcpyDeviceToHost);

    unsigned long long h_cycles[num_pixels];
    cudaMemcpy(h_cycles, d_cycles, num_pixels * sizeof(unsigned long long), cudaMemcpyDeviceToHost);

    // Calcular o total de ciclos e o tempo total
    unsigned long long total_cycles = 0;
    for (int i = 0; i < num_pixels; i++) {
        total_cycles += h_cycles[i];
    }
    double total_seconds = total_cycles / FREQ_GPU_HZ;

    // Exibir resultado
    printf("Resultado da convolução (CUDA):\n");
    for (int y = 0; y < HEIGHT; y++) {
        for (int x = 0; x < WIDTH; x++) {
            printf("%3d ", output[y][x]);
        }
        printf("\n");
    }

    printf("\nTempo total estimado (s): %.9f\n", total_seconds);
    printf("Ciclos totais acumulados: %llu\n", total_cycles);

    // Liberar memória
    cudaFree(d_input);
    cudaFree(d_kernel);
    cudaFree(d_output);
    cudaFree(d_cycles);

    return 0;
}
