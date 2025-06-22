// sobel_openmp.c
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <stdbool.h>
#include <omp.h>
#include "energia.h"

// --- Kernels e Funções Auxiliares (idênticos ao anterior) ---
const float Gx[3][3] = {{-1.0f,0.0f,1.0f},{-2.0f,0.0f,2.0f},{-1.0f,0.0f,1.0f}};
const float Gy[3][3] = {{-1.0f,-2.0f,-1.0f},{0.0f,0.0f,0.0f},{1.0f,2.0f,1.0f}};
const int KERNEL_RADIUS = 1;
inline int min(int a, int b) { return a < b ? a : b; }
inline int max(int a, int b) { return a > b ? a : b; }
inline int clamp(int value, int min_val, int max_val) { return max(min_val, min(value, max_val)); }
void initialize_matrix(float* m, int w, int h) { srand(12345); for(int i=0;i<w*h;++i) m[i]=(float)rand()/(float)RAND_MAX*255.0f; }

// --- Implementação OpenMP ---
void sobel_filter_openmp(
    const float* input, float* output, int width, int height) {
    #pragma omp parallel for //collapse(2) schedule(guided)
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            float px = 0.0f, py = 0.0f;
            for (int ky = -KERNEL_RADIUS; ky <= KERNEL_RADIUS; ++ky) {
                for (int kx = -KERNEL_RADIUS; kx <= KERNEL_RADIUS; ++kx) {
                    int p_y = clamp(y + ky, 0, height - 1);
                    int p_x = clamp(x + kx, 0, width - 1);
                    float p_val = input[p_y * width + p_x];
                    px += p_val * Gx[ky + KERNEL_RADIUS][kx + KERNEL_RADIUS];
                    py += p_val * Gy[ky + KERNEL_RADIUS][kx + KERNEL_RADIUS];
                }
            }
            output[y * width + x] = sqrtf(px * px + py * py);
        }
    }
}

// --- Função de Verificação ---
void verify_results(const float* vec1, const float* vec2, size_t size) {
    // ... Implementação da verificação (opcional, mas recomendado) ...
    printf("Verificação não implementada neste exemplo para simplicidade.\n");
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

    sobel_filter_openmp(input, output, width, height);

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