#include <stdio.h>
#include <time.h>

#define N 9

int isSafe(int board[N][N], int row, int col, int num) {
    for (int x = 0; x < N; x++) {
        if (board[row][x] == num) return 0;
    }
    for (int x = 0; x < N; x++) {
        if (board[x][col] == num) return 0;
    }
    int startRow = row - row % 3;
    int startCol = col - col % 3;
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
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

    for (int num = 1; num <= 9; num++) {
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

    const double cpu_frequency_hz = 4.2e9;  // 4.2 GHz

    clock_t start = clock();
    int solved = solveSudoku(board);
    clock_t end = clock();

    clock_t ticks = end - start;
    double cycles = (double)ticks * (cpu_frequency_hz / CLOCKS_PER_SEC);
    double time_sec = cycles / cpu_frequency_hz;

    if (solved) {
        printf("Sudoku resolvido:\n");
        printBoard(board);
    } else {
        printf("Nenhuma solução existe.\n");
    }

    printf("Ticks do clock: %ld\n", (long)ticks);
    printf("Número estimado de ciclos: %.0f\n", cycles);
    printf("Tempo de execução (calculado via ciclos e frequência): %.8f segundos\n", time_sec);

    return 0;
}
