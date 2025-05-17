#include <stdio.h>
#include <time.h>

#define FREQ_CPU_HZ 4.2e9

double get_elapsed_time(struct timespec start, struct timespec end) {
    return (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;
}

int binary_search(int arr[], int low, int high, int key) {
    if (low > high)
        return -1;

    int mid = (low + high) / 2;

    if (arr[mid] == key)
        return mid;
    else if (key < arr[mid])
        return binary_search(arr, low, mid - 1, key);
    else
        return binary_search(arr, mid + 1, high, key);
}

int main() {
    int arr[] = {1, 3, 5, 7, 9, 11, 13, 15, 17, 19};
    int size = sizeof(arr) / sizeof(arr[0]);
    int key = 7;

    struct timespec start, end;
    clock_gettime(CLOCK_MONOTONIC, &start);

    int result = binary_search(arr, 0, size - 1, key);

    clock_gettime(CLOCK_MONOTONIC, &end);

    double elapsed = get_elapsed_time(start, end);
    long long cycles = (long long)(elapsed * FREQ_CPU_HZ);

    if (result != -1)
        printf("Elemento %d encontrado no índice %d\n", key, result);
    else
        printf("Elemento %d não encontrado\n", key);

    printf("Tempo de execução: %.9f segundos\n", elapsed);
    printf("Estimativa de ciclos: %lld\n", cycles);

    return 0;
}
