#include <stdio.h>
#include <cuda_runtime.h>
#include <time.h>

#define N 9
#define UNASSIGNED 0

// Verifica se é seguro colocar num em board[row][col]
__device__ int isSafeDevice(int board[N][N], int row, int col, int num) {
    for (int x = 0; x < N; x++) {
        if (board[row][x] == num) return 0;
        if (board[x][col] == num) return 0;
    }

    int boxStartRow = row - row % 3;
    int boxStartCol = col - col % 3;

    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++)
            if (board[boxStartRow + i][boxStartCol + j] == num)
                return 0;

    return 1;
}

// Backtracking recursivo dentro do device
__device__ int solveSudokuDevice(int board[N][N]) {
    int row = -1, col = -1;
    int emptyFound = 0;

    for (int i = 0; i < N && !emptyFound; i++) {
        for (int j = 0; j < N && !emptyFound; j++) {
            if (board[i][j] == UNASSIGNED) {
                row = i;
                col = j;
                emptyFound = 1;
            }
        }
    }

    if (!emptyFound) return 1;

    for (int num = 1; num <= N; num++) {
        if (isSafeDevice(board, row, col, num)) {
            board[row][col] = num;
            if (solveSudokuDevice(board))
                return 1;
            board[row][col] = UNASSIGNED;
        }
    }

    return 0;
}

// Kernel: cada thread tenta um número no primeiro espaço vazio
__global__ void sudokuKernel(int *d_board, int *d_solution, int row, int col, int *d_solved) {
    int num = threadIdx.x + 1; // números 1 a 9

    if (num > N) return;

    int board[N][N];

    // Copia o tabuleiro da memória global para a local (thread)
    for (int i = 0; i < N*N; i++) {
        board[i / N][i % N] = d_board[i];
    }

    if (isSafeDevice(board, row, col, num)) {
        board[row][col] = num;

        if (solveSudokuDevice(board)) {
            if (atomicCAS(d_solved, 0, 1) == 0) {
                for (int i = 0; i < N*N; i++) {
                    d_solution[i] = board[i / N][i % N];
                }
            }
        }
    }
}

// Função para imprimir o tabuleiro 9x9
void printBoard(int board[N][N]) {
    for (int row = 0; row < N; row++) {
        for (int col = 0; col < N; col++) {
            if (board[row][col] == 0)
                printf(". ");
            else
                printf("%d ", board[row][col]);
        }
        printf("\n");
    }
}

int main() {
    cudaDeviceSetLimit(cudaLimitStackSize, 32768);
    int board[N][N] = {
        {5,3,0, 0,7,0, 0,0,0},
        {6,0,0, 1,9,5, 0,0,0},
        {0,9,8, 0,0,0, 0,6,0},

        {8,0,0, 0,6,0, 0,0,3},
        {4,0,0, 8,0,3, 0,0,1},
        {7,0,0, 0,2,0, 0,0,6},

        {0,6,0, 0,0,0, 2,8,0},
        {0,0,0, 4,1,9, 0,0,5},
        {0,0,0, 0,8,0, 0,7,9}
    };

    // Encontrar primeiro espaço vazio
    int firstRow = -1, firstCol = -1;
    for (int i = 0; i < N && firstRow == -1; i++) {
        for (int j = 0; j < N; j++) {
            if (board[i][j] == UNASSIGNED) {
                firstRow = i;
                firstCol = j;
                break;
            }
        }
    }
    if (firstRow == -1) {
        printf("Tabuleiro já está completo!\n");
        return 0;
    }

    int *d_board, *d_solution, *d_solved;
    int h_solution[N*N] = {0};
    int h_solved = 0;

    cudaMalloc((void**)&d_board, N*N*sizeof(int));
    cudaMalloc((void**)&d_solution, N*N*sizeof(int));
    cudaMalloc((void**)&d_solved, sizeof(int));

    cudaMemcpy(d_board, board, N*N*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_solved, &h_solved, sizeof(int), cudaMemcpyHostToDevice);

    clock_t start = clock();

    sudokuKernel<<<1, N>>>(d_board, d_solution, firstRow, firstCol, d_solved);

    cudaDeviceSynchronize();

    clock_t end = clock();
    double time_spent = (double)(end - start) / CLOCKS_PER_SEC;

    cudaMemcpy(h_solution, d_solution, N*N*sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(&h_solved, d_solved, sizeof(int), cudaMemcpyDeviceToHost);

    if (h_solved) {
        printf("Sudoku 9x9 resolvido com CUDA:\n");
        int solvedBoard[N][N];
        for (int i = 0; i < N*N; i++) {
            solvedBoard[i / N][i % N] = h_solution[i];
        }
        printBoard(solvedBoard);
    } else {
        printf("Nenhuma solução encontrada.\n");
    }

    printf("\nTempo para resolver: %.5f ms\n", time_spent * 1000);

    cudaFree(d_board);
    cudaFree(d_solution);
    cudaFree(d_solved);

    return 0;
}
