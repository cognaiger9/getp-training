#include "mpi.h"
#include <stdio.h>

#define CHECK_MPI(call)                                                           \
    do                                                                            \
    {                                                                             \
        int code = call;                                                          \
        if (code != MPI_SUCCESS)                                                  \
        {                                                                         \
            char estr[MPI_MAX_ERROR_STRING];                                      \
            int elen;                                                             \
            MPI_Error_string(code, estr, &elen);                                  \
            fprintf(stderr, "MPI error (%s:%d): %s\n", __FILE__, __LINE__, estr); \
            MPI_Abort(MPI_COMM_WORLD, code);                                      \
        }                                                                         \
    } while (0)

int main(int argc, char **argv)
{
    int my_rank, size;
    char hostname[MPI_MAX_PROCESSOR_NAME];
    int hostnamelen;

    MPI_Init(&argc, &argv);
    CHECK_MPI(MPI_Comm_rank(MPI_COMM_WORLD, &my_rank));
    CHECK_MPI(MPI_Comm_size(MPI_COMM_WORLD, &size));
    CHECK_MPI(MPI_Get_processor_name(hostname, &hostnamelen));
    printf("Hello, I am % d of % d, host name is %s\n",
           my_rank, size, hostname);
    MPI_Finalize();
    return 0;
}