CFLAGS := -Wall -O2 -Wno-unused-result -march=native
CC := gcc
BUILD_DIR := build

TARGETS := sobel_sequential sobel_openmp sobel_simd sobel_pthreads sobel_simd_openmp
BUILD_TARGETS := $(addprefix $(BUILD_DIR)/,$(TARGETS))


all: dirs $(BUILD_TARGETS)

build/sobel_sequential: sobel_sequential.c
	$(CC) $(CFLAGS) -o $@ sobel_sequential.c energia.c -lm

build/sobel_openmp: sobel_openmp.c
	$(CC) $(CFLAGS) -o $@ sobel_openmp.c energia.c -fopenmp -lm

build/sobel_simd: sobel_simd.c
	$(CC) $(CFLAGS) -o $@ sobel_simd.c energia.c -mavx2 -lm

build/sobel_simd_openmp: sobel_simd_openmp.c
	$(CC) $(CFLAGS) -o $@ sobel_simd_openmp.c energia.c -mavx2 -lm -fopenmp

build/sobel_pthreads: sobel_pthreads.c
	$(CC) $(CFLAGS) -o $@ sobel_pthreads.c energia.c -lm -pthread

dirs:
	mkdir -p $(BUILD_DIR)

clean:
	rm -rf build