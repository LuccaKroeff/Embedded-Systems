#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cuda_runtime.h>

#define WIDTH 3840
#define HEIGHT 2160
#define CHANNELS 3  // RGB

#ifndef MAX
#define MAX(a, b) ((a) > (b) ? (a) : (b))
#endif

#ifndef MIN
#define MIN(a, b) ((a) < (b) ? (a) : (b))
#endif

__global__ void applyConvolutionCUDA(unsigned char* input, unsigned char* output, int width, int height, int channels, float* kernel) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    int offset = 1;
    if (x >= offset && x < (width - offset) && y >= offset && y < (height - offset)) {
        for (int c = 0; c < channels; c++) {
            float sum = 0.0f;
            for (int ky = 0; ky < 3; ky++) {
                for (int kx = 0; kx < 3; kx++) {
                    int px = x + kx - offset;
                    int py = y + ky - offset;
                    int idx = (py * width + px) * channels + c;
                    sum += input[idx] * kernel[ky * 3 + kx];
                }
            }
            int outIdx = (y * width + x) * channels + c;
            output[outIdx] = (unsigned char)MIN(MAX((int)sum, 0), 255);
        }
    }
}

int main() {
    size_t imgSize = WIDTH * HEIGHT * CHANNELS;
    unsigned char* h_image = (unsigned char*)malloc(imgSize);
    unsigned char* h_output = (unsigned char*)malloc(imgSize);

    if (!h_image || !h_output) {
        printf("Erro ao alocar memória.\n");
        return 1;
    }

    // Inicializa imagem com zeros
    for (int i = 0; i < imgSize; i++) h_image[i] = 0;

    // Mock: insere 100 pontos RGB brancos aleatórios
    srand((unsigned int)time(NULL));
    for (int i = 0; i < 100; i++) {
        int x = rand() % WIDTH;
        int y = rand() % HEIGHT;
        int idx = (y * WIDTH + x) * CHANNELS;
        h_image[idx + 0] = 255;
        h_image[idx + 1] = 255;
        h_image[idx + 2] = 255;
    }

    // Define kernel Laplaciano
    float h_kernel[9] = {
        -1, -1, -1,
        -1,  8, -1,
        -1, -1, -1
    };

    // Aloca memória na GPU
    unsigned char* d_image;
    unsigned char* d_output;
    float* d_kernel;

    cudaMalloc(&d_image, imgSize);
    cudaMalloc(&d_output, imgSize);
    cudaMalloc(&d_kernel, sizeof(float) * 9);

    // Copia dados para a GPU
    cudaMemcpy(d_image, h_image, imgSize, cudaMemcpyHostToDevice);
    cudaMemcpy(d_kernel, h_kernel, sizeof(float) * 9, cudaMemcpyHostToDevice);

    // Define grid e blocos
    dim3 blockSize(16, 16);
    dim3 gridSize((WIDTH + blockSize.x - 1) / blockSize.x,
                  (HEIGHT + blockSize.y - 1) / blockSize.y);

    const double gpu_frequency_hz = 2.5e9;  // 2.5 GHz

    clock_t start = clock();
    applyConvolutionCUDA<<<gridSize, blockSize>>>(d_image, d_output, WIDTH, HEIGHT, CHANNELS, d_kernel);
    cudaDeviceSynchronize();
    clock_t end = clock();

    // Copia resultado para a CPU
    cudaMemcpy(h_output, d_output, imgSize, cudaMemcpyDeviceToHost);

    clock_t ticks = end - start;
    double cycles = (double)ticks * (gpu_frequency_hz / CLOCKS_PER_SEC);
    double time_sec = cycles / gpu_frequency_hz;

    printf("Ticks do clock: %ld\n", (long)ticks);
    printf("Número estimado de ciclos: %.0f\n", cycles);
    printf("Tempo de execução (calculado via ciclos e frequência): %.8f segundos\n", time_sec);


    // Libera memória
    cudaFree(d_image);
    cudaFree(d_output);
    cudaFree(d_kernel);
    free(h_image);
    free(h_output);

    return 0;
}
