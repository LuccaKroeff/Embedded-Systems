#include <stdio.h>
#include <stdlib.h>

#define WIDTH 5
#define HEIGHT 5
#define KERNEL_SIZE 3

// Função para aplicar convolução
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

            // Clipping do valor resultante entre 0 e 255
            if (sum < 0) sum = 0;
            if (sum > 255) sum = 255;

            output[y][x] = (unsigned char)sum;
        }
    }
}

int main() {
    // Exemplo de imagem 5x5 (tons de cinza)
    unsigned char image[HEIGHT][WIDTH] = {
        { 10, 50, 80, 50, 10 },
        { 60, 120, 200, 120, 60 },
        { 90, 180, 255, 180, 90 },
        { 60, 120, 200, 120, 60 },
        { 10, 50, 80, 50, 10 }
    };

    // Kernel exemplo: Filtro de detecção de borda (Sobel simplificado)
    float kernel[KERNEL_SIZE][KERNEL_SIZE] = {
        { -1, -1, -1 },
        { -1,  8, -1 },
        { -1, -1, -1 }
    };

    unsigned char output[HEIGHT][WIDTH];

    convolve2D(image, kernel, output);

    // Mostrar saída
    printf("Resultado da convolução:\n");
    for (int y = 0; y < HEIGHT; y++) {
        for (int x = 0; x < WIDTH; x++) {
            printf("%3d ", output[y][x]);
        }
        printf("\n");
    }

    return 0;
}
