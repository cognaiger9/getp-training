#include "mpi.h"
#include <stdio.h>
#include <stdlib.h>

int main() {
    int count = 10000;
    float *buf = (float *)malloc(count * sizeof(float));
    int rank;

    MPI_Init(NULL, NULL);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Status status;

    if (rank == 0) {
        for (int i = 0; i < count; i++) {
            buf[i] = i * 2;
        }
        MPI_Send(buf, count, MPI_FLOAT, 1, 1234, MPI_COMM_WORLD);
    }

    if (rank == 1) {
        MPI_Recv(buf, count, MPI_FLOAT, 0, 1234, MPI_COMM_WORLD, &status);
        printf("Last ele = %f\n", buf[count - 1]);
    }

    MPI_Finalize();
    return 0;
}