#include <stdio.h>
#include <time.h>  // Incluído para medir o tempo

#define N 16

int isSafe(int board[N][N], int row, int col, int num) {
    for (int x = 0; x < N; x++) {
        if (board[row][x] == num) return 0;
    }
    for (int x = 0; x < N; x++) {
        if (board[x][col] == num) return 0;
    }
    int startRow = row - row % 4;  // corrigido de 3 para 4
    int startCol = col - col % 4;  // corrigido de 3 para 4
    for (int i = 0; i < 4; i++) {  // 4x4 bloco
        for (int j = 0; j < 4; j++) {
            if (board[i + startRow][j + startCol] == num) return 0;
        }
    }
    return 1;
}

int solveSudoku(int board[N][N]) {
    int row, col;
    int emptyFound = 0;
    for (row = 0; row < N; row++) {
        for (col = 0; col < N; col++) {
            if (board[row][col] == 0) {
                emptyFound = 1;
                break;
            }
        }
        if (emptyFound) break;
    }
    if (!emptyFound) return 1;

    for (int num = 1; num <= 16; num++) {  // corrigido de 9 para 16
        if (isSafe(board, row, col, num)) {
            board[row][col] = num;
            if (solveSudoku(board)) return 1;
            board[row][col] = 0;
        }
    }
    return 0;
}

void printBoard(int board[N][N]) {
    for (int row = 0; row < N; row++) {
        for (int col = 0; col < N; col++) {
            int val = board[row][col];
            if (val == 0) {
                printf(". ");
            } else if (val <= 9) {
                printf("%d ", val);
            } else {
                printf("%c ", 'A' + val - 10);
            }
        }
        printf("\n");
    }
}

int main() {
    int board[N][N] = {
        {0, 0, 0, 0,  0, 0, 3, 0,  0, 0, 0, 0,  0, 0, 0, 0},
        {0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 7, 0,  0, 0, 0, 0},
        {0, 0, 0, 6,  0, 0, 0, 0,  0, 9, 0, 0,  0, 0, 0, 0},
        {0, 0, 0, 0,  2, 0, 0, 0,  0, 0, 0, 0,  8, 0, 0, 0},

        {0, 0, 0, 0,  0, 0, 0, 0,  0, 3, 0, 0,  0, 7, 0, 0},
        {0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 4},
        {7, 0, 0, 0,  0, 0, 0, 5,  0, 0, 0, 0,  0, 0, 0, 0},
        {0, 0, 0, 0,  4, 0, 0, 0,  0, 0, 0, 1,  0, 0, 0, 0},

        {0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 5, 0,  0, 0, 0, 0},
        {0, 0, 0, 0,  0, 1, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0},
        {0, 0, 0, 0,  7, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0},
        {0, 0, 6, 0,  0, 0, 0, 0,  0, 0, 0, 3,  0, 0, 0, 0},

        {0, 0, 0, 0,  0, 0, 0, 0,  8, 0, 0, 0,  0, 0, 0, 0},
        {0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0,  6, 0, 0, 0},
        {0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0,  0, 5, 0, 0},
        {0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0}
    };

    clock_t start = clock();  // Início da medição
    int solved = solveSudoku(board);
    clock_t end = clock();    // Fim da medição

    double time_spent = (double)(end - start) / CLOCKS_PER_SEC;

    if (solved) {
        printf("Sudoku resolvido:\n");
        printBoard(board);
    } else {
        printf("Nenhuma solução existe.\n");
    }

    printf("Tempo para resolver: %.5f segundos\n", time_spent);

    return 0;
}
