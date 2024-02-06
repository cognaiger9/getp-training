#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include "util.h"

/*
Time increase: Standard send -> Synchronous send -> 
*/
void communication(int count)
{
    int rank;
    MPI_Init(NULL, NULL);
    MPI_Status status;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    int packsize;
    MPI_Pack_size(count, MPI_FLOAT, MPI_COMM_WORLD, &packsize);

    int buffersize = packsize + MPI_BSEND_OVERHEAD;
    float *arr = (float *)malloc(buffersize);
    MPI_Buffer_attach(arr, buffersize);
    if (rank == 0)
    {
        for (int i = 0; i < count; i++)
        {
            arr[i] = i;
        }
        MPI_Bsend(arr, count, MPI_FLOAT, 1, 123, MPI_COMM_WORLD);
    }

    if (rank == 1)
    {
        MPI_Recv(arr, count, MPI_FLOAT, 0, 123, MPI_COMM_WORLD, &status);
        for (int i = 0; i < count; i++)
        {
            if (arr[i] != i)
            {
                printf("Invalid array in destination \n");
                exit(1);
            }
        }
    }
    MPI_Finalize();
}

int main()
{
    int count = 100000;
    double elapsed_time = 0;
    timer_start();
    communication(count);
    elapsed_time = timer_stop();
    printf("Time elapsed: %f\n", elapsed_time);
    printf("Network size (GB) = %f\n", sizeof(float) * count / elapsed_time / 1e9);
    return 0;
}