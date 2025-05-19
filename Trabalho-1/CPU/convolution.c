#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#define WIDTH 3840
#define HEIGHT 2160
#define CHANNELS 3  // RGB

#ifndef MAX
#define MAX(a, b) ((a) > (b) ? (a) : (b))
#endif

#ifndef MIN
#define MIN(a, b) ((a) < (b) ? (a) : (b))
#endif

// Aplica convolução com um kernel 3x3
void applyConvolution(unsigned char* input, unsigned char* output, int width, int height, int channels, float kernel[3][3]) {
    int offset = 1;
    for (int y = offset; y < height - offset; y++) {
        for (int x = offset; x < width - offset; x++) {
            for (int c = 0; c < channels; c++) {
                float sum = 0.0f;
                for (int ky = 0; ky < 3; ky++) {
                    for (int kx = 0; kx < 3; kx++) {
                        int px = x + kx - offset;
                        int py = y + ky - offset;
                        int idx = (py * width + px) * channels + c;
                        sum += input[idx] * kernel[ky][kx];
                    }
                }
                int outIdx = (y * width + x) * channels + c;
                output[outIdx] = (unsigned char)MIN(MAX((int)sum, 0), 255);
            }
        }
    }
}

int main() {
    unsigned char* image = malloc(WIDTH * HEIGHT * CHANNELS);
    unsigned char* output = malloc(WIDTH * HEIGHT * CHANNELS);

    if (!image || !output) {
        printf("Erro ao alocar memória.\n");
        return 1;
    }

    // Zera toda a imagem
    for (int i = 0; i < WIDTH * HEIGHT * CHANNELS; i++) {
        image[i] = 0;
    }

    // Mock: insere 100 pontos RGB brancos em posições aleatórias
    srand((unsigned int)time(NULL));
    for (int i = 0; i < 100; i++) {
        int x = rand() % WIDTH;
        int y = rand() % HEIGHT;
        int idx = (y * WIDTH + x) * CHANNELS;
        image[idx + 0] = 255;  // R
        image[idx + 1] = 255;  // G
        image[idx + 2] = 255;  // B
    }

    // Kernel Laplaciano
    float kernel[3][3] = {
        { -1, -1, -1 },
        { -1,  8, -1 },
        { -1, -1, -1 }
    };

    clock_t start = clock();
    applyConvolution(image, output, WIDTH, HEIGHT, CHANNELS, kernel);
    clock_t end = clock();

    double time_spent = (double)(end - start) / CLOCKS_PER_SEC;
    printf("Convolução concluída em %.5f ms.\n", time_spent * 1000);

    free(image);
    free(output);
    return 0;
}
