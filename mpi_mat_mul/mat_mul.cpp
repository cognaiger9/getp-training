#include "mat_mul.h"

#include <cstdio>
#include <cstdlib>
#include <mpi.h>
#include <pthread.h>
#include <numa.h>
#include <numaif.h>

#define MTILESIZE (32)
#define NTILESIZE (1024)
#define KTILESIZE (384)


static float *A, *B, *C;
static int M, N, K;
static int num_threads;
static int mpi_rank, mpi_world_size, rows_per_node;

void mat_mul(float *_A, float *_B, float *_C, int _M, int _N, int _K,
             int _num_threads, int _mpi_rank, int _mpi_world_size)
{
  A = _A, B = _B, C = _C;
  M = _M, N = _N, K = _K;
  num_threads = _num_threads, mpi_rank = _mpi_rank,
  mpi_world_size = _mpi_world_size;

  rows_per_node = M / mpi_world_size;

  MPI_Request requests[3];

  // Scatter A
  MPI_Iscatter(A, rows_per_node * K, MPI_FLOAT, A, rows_per_node * K, MPI_FLOAT, 0, MPI_COMM_WORLD, &requests[0]);
  // Broadcast B
  MPI_Ibcast(B, K * N, MPI_FLOAT, 0, MPI_COMM_WORLD, &requests[1]);

  MPI_Wait(&requests[0], MPI_STATUS_IGNORE);
  MPI_Wait(&requests[1], MPI_STATUS_IGNORE);

  pthread_one_node();

  // Gather result in C
  MPI_Igather(C, rows_per_node * N, MPI_FLOAT, C, rows_per_node * N, MPI_FLOAT, 0, MPI_COMM_WORLD, &requests[2]);
  MPI_Wait(&requests[2], MPI_STATUS_IGNORE);
}

void pthread_one_node()
{

  pthread_t tid[num_threads];

  pthread_attr_t attr[num_threads];
  cpu_set_t cpus[num_threads];

  for (long i = 0; i < num_threads; i++)
  {
    pthread_attr_init(&attr[i]);
    CPU_ZERO(&cpus[i]);
    CPU_SET(i, &cpus[i]);

    // Limit thread to run only on 1 core
    pthread_attr_setaffinity_np(&attr[i], sizeof(cpu_set_t), &cpus[i]);
    pthread_create(&tid[i], &attr[i], mat_mul_thread, (void *)i);
  }

  // Sync threads
  for (int i = 0; i < num_threads; i++)
  {
    pthread_join(tid[i], NULL);
  }
}

void *mat_mul_thread(void *arg)
{
  int tid = (long)arg;

  int startM = rows_per_node / num_threads * tid + std::min(tid, rows_per_node % num_threads);
  int endM = rows_per_node / num_threads * (tid + 1) + std::min(tid + 1, rows_per_node % num_threads);

  int evenK = K - K % KTILESIZE;

  // Handle for even K
  for (int kk = 0; kk < evenK; kk += KTILESIZE)
  {
    int KTileBoundary = kk + KTILESIZE;
    for (int mm = startM; mm < endM; mm += MTILESIZE)
    {
      int MTileBoundary = std::min(mm + MTILESIZE, endM);
      for (int nn = 0; nn < N; nn += NTILESIZE)
      {
        int NTileBoundary = std::min(nn + NTILESIZE, N);
        // Mat mul
        for (int k = kk; k < KTileBoundary; k += 6)
        {
          for (int m = mm; m < MTileBoundary; m++)
          {

            float a0 = A[m * K + (k + 0)];
            float a1 = A[m * K + (k + 1)];
            float a2 = A[m * K + (k + 2)];
            float a3 = A[m * K + (k + 3)];
            float a4 = A[m * K + (k + 4)];
            float a5 = A[m * K + (k + 5)];

            for (int n = nn; n < NTileBoundary; n++)
            {
              float b0 = B[(k + 0) * N + n];
              float b1 = B[(k + 1) * N + n];
              float b2 = B[(k + 2) * N + n];
              float b3 = B[(k + 3) * N + n];
              float b4 = B[(k + 4) * N + n];
              float b5 = B[(k + 5) * N + n];

              float c0 = a0 * b0 + a1 * b1 + a2 * b2 + a3 * b3 + a4 * b4 + a5 * b5;

              C[m * N + n] += c0;
            }
          }
        }
      }
    }
  }

  // Handle the remainder
  for (int mm = startM; mm < endM; mm += MTILESIZE)
  {
    int MTileBoundary = std::min(mm + MTILESIZE, endM);
    for (int nn = 0; nn < N; nn += NTILESIZE)
    {
      int NTileBoundary = std::min(nn + NTILESIZE, N);

      for (int k = evenK; k < K; k += 4)
      {
        for (int m = mm; m < MTileBoundary; m++)
        {
          float a1 = A[m * K + k + 0];
          float a2 = (k + 1 >= K) ? 0 : A[m * K + k + 1];
          float a3 = (k + 2 >= K) ? 0 : A[m * K + k + 2];
          float a4 = (k + 3 >= K) ? 0 : A[m * K + k + 3];

          for (int n = nn; n < NTileBoundary; n++)
          {
            float b1 = B[(k + 0) * N + n];
            float b2 = (k + 1 >= K) ? 0 : B[(k + 1) * N + n];
            float b3 = (k + 2 >= K) ? 0 : B[(k + 2) * N + n];
            float b4 = (k + 3 >= K) ? 0 : B[(k + 3) * N + n];

            C[m * N + n] += a1 * b1 + a2 * b2 + a3 * b3 + a4 * b4;
          }
        }
      }
    }
  }

  pthread_exit(NULL);
}
