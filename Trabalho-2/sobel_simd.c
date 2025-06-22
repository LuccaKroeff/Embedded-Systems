// sobel_simd.c
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <stdbool.h>
#include <immintrin.h>
#include "energia.h"

// --- Kernels e Funções Auxiliares (idênticos ao anterior) ---
const float Gx[3][3] = {{-1.0f,0.0f,1.0f},{-2.0f,0.0f,2.0f},{-1.0f,0.0f,1.0f}};
const float Gy[3][3] = {{-1.0f,-2.0f,-1.0f},{0.0f,0.0f,0.0f},{1.0f,2.0f,1.0f}};
const int KERNEL_RADIUS = 1;
inline int min(int a, int b) { return a < b ? a : b; }
inline int max(int a, int b) { return a > b ? a : b; }
inline int clamp(int value, int min_val, int max_val) { return max(min_val, min(value, max_val)); }
void initialize_matrix(float* m, int w, int h) { srand(12345); for(int i=0;i<w*h;++i) m[i]=(float)rand()/(float)RAND_MAX*255.0f; }

// --- Implementação SIMD ---
void sobel_filter_simd(
    const float* input, float* output, int width, int height) {
    const int VEC_WIDTH = 8;
    for (int y = 0; y < height; ++y) {
        if (y < KERNEL_RADIUS || y >= height - KERNEL_RADIUS) {
            for (int x = 0; x < width; ++x) {
                float px = 0.0f, py = 0.0f;
                for (int ky = -KERNEL_RADIUS; ky <= KERNEL_RADIUS; ++ky) {
                    for (int kx = -KERNEL_RADIUS; kx <= KERNEL_RADIUS; ++kx) {
                        int p_y = clamp(y + ky, 0, height - 1);
                        int p_x = clamp(x + kx, 0, width - 1);
                        px += input[p_y * width + p_x] * Gx[ky + KERNEL_RADIUS][kx + KERNEL_RADIUS];
                        py += input[p_y * width + p_x] * Gy[ky + KERNEL_RADIUS][kx + KERNEL_RADIUS];
                    }
                }
                output[y * width + x] = sqrtf(px * px + py * py);
            }
        } else {
            int x = 0;
            for (x = 0; x < KERNEL_RADIUS; ++x) { // Borda Esquerda
                float px = 0.0f, py = 0.0f;
                for (int ky = -KERNEL_RADIUS; ky <= KERNEL_RADIUS; ++ky) {
                    for (int kx = -KERNEL_RADIUS; kx <= KERNEL_RADIUS; ++kx) {
                        int p_y = y + ky;
                        int p_x = clamp(x + kx, 0, width - 1);
                        px += input[p_y * width + p_x] * Gx[ky + KERNEL_RADIUS][kx + KERNEL_RADIUS];
                        py += input[p_y * width + p_x] * Gy[ky + KERNEL_RADIUS][kx + KERNEL_RADIUS];
                    }
                }
                output[y * width + x] = sqrtf(px * px + py * py);
            }
            int simd_end_x = width - VEC_WIDTH - KERNEL_RADIUS;
            for (; x <= simd_end_x; x += VEC_WIDTH) { // Centro SIMD
                __m256 px_vec = _mm256_setzero_ps(); __m256 py_vec = _mm256_setzero_ps();
                for (int ky = -KERNEL_RADIUS; ky <= KERNEL_RADIUS; ++ky) {
                    for (int kx = -KERNEL_RADIUS; kx <= KERNEL_RADIUS; ++kx) {
                        float gv_x = Gx[ky+KERNEL_RADIUS][kx+KERNEL_RADIUS], gv_y = Gy[ky+KERNEL_RADIUS][kx+KERNEL_RADIUS];
                        if (gv_x==0.0f && gv_y==0.0f) continue;
                        __m256 p_vec = _mm256_loadu_ps(&input[(y+ky)*width+(x+kx)]);
                        if (gv_x!=0.0f) px_vec=_mm256_add_ps(px_vec,_mm256_mul_ps(p_vec,_mm256_set1_ps(gv_x)));
                        if (gv_y!=0.0f) py_vec=_mm256_add_ps(py_vec,_mm256_mul_ps(p_vec,_mm256_set1_ps(gv_y)));
                    }
                }
                __m256 mag = _mm256_sqrt_ps(_mm256_add_ps(_mm256_mul_ps(px_vec,px_vec),_mm256_mul_ps(py_vec,py_vec)));
                _mm256_storeu_ps(&output[y * width + x], mag);
            }
            for (; x < width; ++x) { // Borda Direita
                float px = 0.0f, py = 0.0f;
                for (int ky = -KERNEL_RADIUS; ky <= KERNEL_RADIUS; ++ky) {
                    for (int kx = -KERNEL_RADIUS; kx <= KERNEL_RADIUS; ++kx) {
                        int p_y = y + ky;
                        int p_x = clamp(x + kx, 0, width - 1);
                        px += input[p_y*width+p_x] * Gx[ky+KERNEL_RADIUS][kx+KERNEL_RADIUS];
                        py += input[p_y*width+p_x] * Gy[ky+KERNEL_RADIUS][kx+KERNEL_RADIUS];
                    }
                }
                output[y * width + x] = sqrtf(px * px + py * py);
            }
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


    sobel_filter_simd(input, output, width, height);

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