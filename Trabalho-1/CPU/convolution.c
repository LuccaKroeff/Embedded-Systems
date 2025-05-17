#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define WIDTH 5
#define HEIGHT 5
#define KERNEL_SIZE 3
#define FREQ_CPU_HZ 4.2e9

double get_elapsed_time(struct timespec start, struct timespec end) {
    return (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;
}

void convolve2D(unsigned char input[HEIGHT][WIDTH], float kernel[KERNEL_SIZE][KERNEL_SIZE], unsigned char output[HEIGHT][WIDTH]) {
    int pad = KERNEL_SIZE / 2;

    for (int y = 0; y < HEIGHT; y++) {
        for (int x = 0; x < WIDTH; x++) {
            float sum = 0.0;

            for (int ky = 0; ky < KERNEL_SIZE; ky++) {
                for (int kx = 0; kx < KERNEL_SIZE; kx++) {
                    int ix = x + kx - pad;
                    int iy = y + ky - pad;

                    if (ix >= 0 && ix < WIDTH && iy >= 0 && iy < HEIGHT) {
                        sum += input[iy][ix] * kernel[ky][kx];
                    }
                }
            }

            if (sum < 0) sum = 0;
            if (sum > 255) sum = 255;
            output[y][x] = (unsigned char)sum;
        }
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

    struct timespec start, end;
    clock_gettime(CLOCK_MONOTONIC, &start);

    convolve2D(image, kernel, output);

    clock_gettime(CLOCK_MONOTONIC, &end);

    double elapsed = get_elapsed_time(start, end);
    long long cycles = (long long)(elapsed * FREQ_CPU_HZ);

    printf("Resultado da convolução:\n");
    for (int y = 0; y < HEIGHT; y++) {
        for (int x = 0; x < WIDTH; x++) {
            printf("%3d ", output[y][x]);
        }
        printf("\n");
    }

    printf("Tempo de execução: %.9f segundos\n", elapsed);
    printf("Estimativa de ciclos: %lld\n", cycles);

    return 0;
}
