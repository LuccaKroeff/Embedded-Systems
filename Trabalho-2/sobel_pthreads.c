// sobel_pthreads.c
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <stdbool.h>
#include <pthread.h>
#include "energia.h"

// --- Kernels de Sobel e Funções Auxiliares ---
const float Gx[3][3] = {{-1.0f,0.0f,1.0f},{-2.0f,0.0f,2.0f},{-1.0f,0.0f,1.0f}};
const float Gy[3][3] = {{-1.0f,-2.0f,-1.0f},{0.0f,0.0f,0.0f},{1.0f,2.0f,1.0f}};
const int KERNEL_RADIUS = 1;

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

// --- Estrutura para passar dados para as threads ---
typedef struct {
    int start_row;
    int end_row;
    int width;
    int height;
    const float* input;
    float* output;
} thread_data_t;

// --- Função de trabalho que cada thread executará ---
void* sobel_worker(void* arg) {
    thread_data_t* data = (thread_data_t*)arg;

    // A lógica do filtro é aplicada apenas à fatia de linhas da thread
    for (int y = data->start_row; y < data->end_row; ++y) {
        for (int x = 0; x < data->width; ++x) {
            float px = 0.0f, py = 0.0f;
            for (int ky = -KERNEL_RADIUS; ky <= KERNEL_RADIUS; ++ky) {
                for (int kx = -KERNEL_RADIUS; kx <= KERNEL_RADIUS; ++kx) {
                    int pixel_y = clamp(y + ky, 0, data->height - 1);
                    int pixel_x = clamp(x + kx, 0, data->width - 1);
                    float pixel_value = data->input[pixel_y * data->width + pixel_x];
                    px += pixel_value * Gx[ky + KERNEL_RADIUS][kx + KERNEL_RADIUS];
                    py += pixel_value * Gy[ky + KERNEL_RADIUS][kx + KERNEL_RADIUS];
                }
            }
            data->output[y * data->width + x] = sqrtf(px * px + py * py);
        }
    }
    return NULL;
}

// --- Função principal que orquestra a criação das threads ---
void sobel_filter_pthreads(
    const float* input, float* output, int width, int height, int num_threads) {

    pthread_t* threads = malloc(num_threads * sizeof(pthread_t));
    thread_data_t* thread_data = malloc(num_threads * sizeof(thread_data_t));

    int rows_per_thread = height / num_threads;

    for (int i = 0; i < num_threads; ++i) {
        thread_data[i].width = width;
        thread_data[i].height = height;
        thread_data[i].input = input;
        thread_data[i].output = output;
        thread_data[i].start_row = i * rows_per_thread;
        // Garante que a última thread processe até o fim, cobrindo o resto da divisão.
        thread_data[i].end_row = (i == num_threads - 1) ? height : (i + 1) * rows_per_thread;

        pthread_create(&threads[i], NULL, sobel_worker, &thread_data[i]);
    }

    // Aguarda todas as threads terminarem (sincronização)
    for (int i = 0; i < num_threads; ++i) {
        pthread_join(threads[i], NULL);
    }

    free(threads);
    free(thread_data);
}

int main(int argc, char* argv[]) {
    rapl_init();
    if (argc != 3) {
        fprintf(stderr, "Uso: %s <tamanho> <num_threads>\n", argv[0]);
        return 1;
    }
    int size = atoi(argv[1]);
    int num_threads = atoi(argv[2]);

    if (size <= 0 || num_threads <= 0) {
        fprintf(stderr, "Erro: Tamanho e número de threads devem ser positivos.\n");
        return 1;
    }

    int width = size, height = size;
    size_t total_size = (size_t)width * height;
    printf("Executando filtro de Sobel (Pthreads) em uma matriz %dx%d com %d threads\n", width, height, num_threads);

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

    sobel_filter_pthreads(input, output, width, height, num_threads);

    clock_t end = clock();
    clock_gettime(CLOCK_MONOTONIC, &real_end);
    
    double cpu_time = (double)(end - start) / CLOCKS_PER_SEC;
    double real_time = (real_end.tv_sec - real_start.tv_sec)
                     + (real_end.tv_nsec - real_start.tv_nsec) / 1e9;
    double energy = end_rapl_sysfs();

    printf("Real Time (pthreads): %.5f\n", real_time);
    printf("CPU Time (pthreads): %.5f\n", cpu_time);
    printf("Energy (J): %.5f\n", energy);

    free(input);
    free(output);
    return 0;
}