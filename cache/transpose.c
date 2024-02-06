/* 
 * trans.c - Matrix transpose B = A^T
 *
 * Each transpose function must have a prototype of the form:
 * void trans(int M, int N, int A[N][M], int B[M][N]);
 *
 * A transpose function is evaluated by counting the number of misses
 * on a 1KB direct mapped cache with a block size of 32 bytes.
 */ 
#include <stdio.h>
#include "cachelab.h"

int is_transpose(int M, int N, int A[N][M], int B[M][N]);

/* 
 * transpose - This is the solution transpose function that you
 *     will be graded on for Part B of the assignment. Do not change
 *     the description string "Transpose submission", as the driver
 *     searches for that string to identify the transpose function to
 *     be graded. 
 */
char transpose_desc[] = "Transpose submission";
void transpose(int M, int N, int A[N][M], int B[M][N])
{
  /* TODO: FILL IN HERE */
  int blockH = 8;
  int blockW = 8;
  int blockIterH = (M % blockH == 0) ? M / blockH : (M / blockH) + 1;
  int blockIterW = (N % blockW == 0) ? N / blockW : (N / blockW) + 1;
  for (int i = 0; i < blockIterH; i++) {
    for (int j = 0; j < blockIterW; j++) {
        for (int m = 0; m < blockH; m++) {
            for (int n = 0; n < blockW; n++) {
                int col = j * blockW + n;
                int row = i * blockH + m;
                if (col < M && row < N) {
                    B[col][row] = A[row][col];
                }
            }
        }
    }
  }
}


/*
 * registerFunctions - This function registers your transpose
 *     functions with the driver.  At runtime, the driver will
 *     evaluate each of the registered functions and summarize their
 *     performance. This is a handy way to experiment with different
 *     transpose strategies.
 */
void registerFunctions()
{
    /* Register your solution function */
    registerTransFunction(transpose, transpose_desc); 
}

/* 
 * is_transpose - This helper function checks if B is the transpose of
 *     A. You can check the correctness of your transpose by calling
 *     it before returning from the transpose function.
 */
int is_transpose(int M, int N, int A[N][M], int B[M][N])
{
    int i, j;

    for (i = 0; i < N; i++) {
        for (j = 0; j < M; ++j) {
            if (A[i][j] != B[j][i]) {
                return 0;
            }
        }
    }
    return 1;
}