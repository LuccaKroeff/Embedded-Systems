// sobel_sequential.c
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <stdbool.h>
#include "energia.h"

// --- Kernels de Sobel ---
const float Gx[3][3] = {{-1.0f,0.0f,1.0f},{-2.0f,0.0f,2.0f},{-1.0f,0.0f,1.0f}};
const float Gy[3][3] = {{-1.0f,-2.0f,-1.0f},{0.0f,0.0f,0.0f},{1.0f,2.0f,1.0f}};
const int KERNEL_RADIUS = 1;

// --- Funções Auxiliares ---
inline int min(int a, int b) { return a < b ? a : b; }
inline int max(int a, int b) { return a > b ? a : b; }
inline int clamp(int value, int min_val, int max_val) {
    return max(min_val, min(value, max_val));
}

void initialize_matrix(float* matrix, int width, int height) {
    srand(12345);
    for (int i = 0; i < width * height; ++i) {
        matrix[i] = (float)rand() / (float)RAND_MAX * 255.0f;
    }
}

// --- Implementação Sequencial ---
void sobel_filter_sequential(
    const float* input, float* output, int width, int height) {
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            float px = 0.0f, py = 0.0f;
            for (int ky = -KERNEL_RADIUS; ky <= KERNEL_RADIUS; ++ky) {
                for (int kx = -KERNEL_RADIUS; kx <= KERNEL_RADIUS; ++kx) {
                    int pixel_y = clamp(y + ky, 0, height - 1);
                    int pixel_x = clamp(x + kx, 0, width - 1);
                    float pixel_value = input[pixel_y * width + pixel_x];
                    px += pixel_value * Gx[ky + KERNEL_RADIUS][kx + KERNEL_RADIUS];
                    py += pixel_value * Gy[ky + KERNEL_RADIUS][kx + KERNEL_RADIUS];
                }
            }
            output[y * width + x] = sqrtf(px * px + py * py);
        }
    }
}

int main(int argc, char* argv[]) {
    rapl_init();

    if (argc != 2) {
        fprintf(stderr, "Uso: %s <tamanho>\n", argv[0]);
        return 1;
    }
    int size = atoi(argv[1]);
    if (size <= 0) {
        fprintf(stderr, "Erro: Tamanho inválido.\n");
        return 1;
    }

    int width = size, height = size;
    size_t total_size = (size_t)width * height;
    printf("Executando filtro de Sobel (Sequencial) em uma matriz %dx%d\n", width, height);

    float *input = malloc(total_size * sizeof(float));
    float *output = malloc(total_size * sizeof(float));
    if (!input || !output) {
        fprintf(stderr, "Falha na alocação de memória!\n");
        free(input); free(output);
        return 1;
    }

    initialize_matrix(input, width, height);

    start_rapl_sysfs();

    struct timespec real_start, real_end;
    clock_gettime(CLOCK_MONOTONIC, &real_start);
    clock_t start = clock();


    sobel_filter_sequential(input, output, width, height);

    clock_t end = clock();
    clock_gettime(CLOCK_MONOTONIC, &real_end);
    
    double cpu_time = (double)(end - start) / CLOCKS_PER_SEC;
    double real_time = (real_end.tv_sec - real_start.tv_sec)
                     + (real_end.tv_nsec - real_start.tv_nsec) / 1e9;
    double energy = end_rapl_sysfs();

    printf("Real Time (simd): %.5f\n", real_time);
    printf("CPU Time (simd): %.5f\n", cpu_time);
    printf("Energy (J): %.5f\n", energy);

    free(input);
    free(output);
    return 0;
}