#include <stdio.h>

// Função de busca binária recursiva
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

// Função main equivalente ao código CUDA
int main() {
    int arr[] = {1, 3, 5, 7, 9, 11, 13, 15, 17, 19};
    int size = sizeof(arr) / sizeof(arr[0]);
    int key = 7;
    int result = binary_search(arr, 0, size - 1, key);

    if (result != -1) {
        printf("Elemento %d encontrado no índice %d\n", key, result);
    } else {
        printf("Elemento %d não encontrado\n", key);
    }

    return 0;
}
