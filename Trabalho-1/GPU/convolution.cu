#include <stdio.h>
#include <cuda_runtime.h>

#define WIDTH 5
#define HEIGHT 5
#define KERNEL_SIZE 3

// Kernel CUDA para convolução 2D
__global__ void convolve2D_cuda(
    unsigned char* input, 
    float* kernel, 
    unsigned char* output, 
    int width, int height, int kernel_size) 
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int pad = kernel_size / 2;

    if (x < width && y < height) {
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

        // Clipping do valor entre 0 e 255
        if (sum < 0) sum = 0;
        if (sum > 255) sum = 255;

        output[y * width + x] = (unsigned char)sum;
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

    // Alocar memória da GPU
    unsigned char* d_input;
    float* d_kernel;
    unsigned char* d_output;

    cudaMalloc(&d_input, WIDTH * HEIGHT * sizeof(unsigned char));
    cudaMalloc(&d_kernel, KERNEL_SIZE * KERNEL_SIZE * sizeof(float));
    cudaMalloc(&d_output, WIDTH * HEIGHT * sizeof(unsigned char));

    // Copiar dados para a GPU
    cudaMemcpy(d_input, image, WIDTH * HEIGHT * sizeof(unsigned char), cudaMemcpyHostToDevice);
    cudaMemcpy(d_kernel, kernel, KERNEL_SIZE * KERNEL_SIZE * sizeof(float), cudaMemcpyHostToDevice);

    // Definir dimensões da grade e blocos
    dim3 blockDim(16, 16);
    dim3 gridDim((WIDTH + blockDim.x - 1) / blockDim.x, (HEIGHT + blockDim.y - 1) / blockDim.y);

    // Executar kernel
    convolve2D_cuda<<<gridDim, blockDim>>>(d_input, d_kernel, d_output, WIDTH, HEIGHT, KERNEL_SIZE);

    // Copiar resultado de volta para CPU
    cudaMemcpy(output, d_output, WIDTH * HEIGHT * sizeof(unsigned char), cudaMemcpyDeviceToHost);

    // Mostrar saída
    printf("Resultado da convolução (CUDA):\n");
    for (int y = 0; y < HEIGHT; y++) {
        for (int x = 0; x < WIDTH; x++) {
            printf("%3d ", output[y][x]);
        }
        printf("\n");
    }

    // Liberar memória da GPU
    cudaFree(d_input);
    cudaFree(d_kernel);
    cudaFree(d_output);

    return 0;
}
