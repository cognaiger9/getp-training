for (int k = 0; k < K; k += KTILESIZE)
  {
    int KBoundary = std::min(K, k + KTILESIZE);
    #pragma omp parallel for num_threads(num_threads)
    for (int i = 0; i < row; i += MTILESIZE)
    {
      int MBoundary = std::min(M, i + MTILESIZE);
      for (int j = 0; j < N; j += NTILESIZE)
      {
        int NBoundary = std::min(N, j + NTILESIZE);
        
        for (int kk = k; kk < KBoundary; kk++) {
          for (int mm = i; mm < MBoundary; mm++) {
            for (int nn = j; nn < NBoundary; nn++) {
              C[kk * N + nn] += A[mm * K + kk] * B[kk * N + nn];
            }
          }
        }
      }
    }
  }




/*

MPI OpenMP fluc around 600 900

*/

#include "mat_mul.h"

#include <cstdio>
#include <cstdlib>
#include <mpi.h>
#include "util.h"

#define MTILESIZE 32
#define NTILESIZE 512
#define KTILESIZE 512

static float *A, *B, *C;
static int M, N, K;
static int num_threads;
static int mpi_rank, mpi_world_size;

void mat_mul(float *_A, float *_B, float *_C, int _M, int _N, int _K,
             int _num_threads, int _mpi_rank, int _mpi_world_size)
{
  A = _A, B = _B, C = _C;
  M = _M, N = _N, K = _K;
  num_threads = _num_threads, mpi_rank = _mpi_rank,
  mpi_world_size = _mpi_world_size;

  // distribute portion of A to nodes
  int startM = M / mpi_world_size * mpi_rank;
  int endM = M / mpi_world_size * (mpi_rank + 1);
  int row = endM - startM;

  // Scatter A
  MPI_Scatter(A, row * K, MPI_FLOAT, A, row * K, MPI_FLOAT, 0, MPI_COMM_WORLD);

  // Broadcast B
  MPI_Bcast(B, K * N, MPI_FLOAT, 0, MPI_COMM_WORLD);

  // core process on 1 node
  for (int k = 0; k < K; k += KTILESIZE)
  {
    int KBoundary = std::min(K, k + KTILESIZE);
    #pragma omp parallel for num_threads(num_threads)
    for (int m = 0; m < row; m += MTILESIZE)
    {
      int MBoundary = std::min(row, m + MTILESIZE);
      for (int n = 0; n < N; n += NTILESIZE)
      {
        int NBoundary = std::min(N, n + NTILESIZE);

        for (int kk = k; kk < KBoundary; kk += 4)
        {
          for (int mm = m; mm < MBoundary; mm++)
          {
            float a1 = A[mm * K + kk + 0];
            float a2 = A[mm * K + kk + 1];
            float a3 = A[mm * K + kk + 2];
            float a4 = A[mm * K + kk + 3];
            for (int nn = n; nn < NBoundary; nn++)
            {
              float b1 = B[(kk + 0) * N + nn];
              float b2 = B[(kk + 1) * N + nn];
              float b3 = B[(kk + 2) * N + nn];
              float b4 = B[(kk + 3) * N + nn];

              C[mm * N + nn] += a1 * b1 + a2 * b2 + a3 * b3 + a4 * b4;
            }
          }
        }
      }
    }
  }

  // Gather result C
  MPI_Gather(C, row * N, MPI_FLOAT, C, row * N, MPI_FLOAT, 0, MPI_COMM_WORLD);
}




/*
SIMD MPI ~ 1100
*/
#include "mat_mul.h"

#include <cstdio>
#include <cstdlib>
#include <mpi.h>
#include <immintrin.h>
#include "util.h"

#define MTILESIZE 96
#define NTILESIZE 512
#define KTILESIZE 1024

// static variables used in 1 process
static float *A, *B, *C;
static int M, N, K;
static int num_threads;
static int mpi_rank, mpi_world_size;
static int row_per_node;

void pthread_one_node();
void *mat_mul_thread(void *args);

void mat_mul(float *_A, float *_B, float *_C, int _M, int _N, int _K,
             int _num_threads, int _mpi_rank, int _mpi_world_size)
{
  // distribute portion of A to nodes
  row_per_node = _M / _mpi_world_size;

  A = _A, B = _B, C = _C;
  M = _M, N = _N, K = _K;
  num_threads = _num_threads, mpi_rank = _mpi_rank, mpi_world_size = _mpi_world_size;

  MPI_Request requests[3];

  // Scatter A
  // MPI_Request request;
  MPI_Iscatter(A, row_per_node * K, MPI_FLOAT, A, row_per_node * K, MPI_FLOAT, 0, MPI_COMM_WORLD, &requests[0]);
  // MPI_Wait(&request, MPI_STATUS_IGNORE);

  // Broadcast B
  // MPI_Request request2;
  MPI_Ibcast(B, K * N, MPI_FLOAT, 0, MPI_COMM_WORLD, &requests[1]);
  MPI_Wait(&requests[1], MPI_STATUS_IGNORE);

  pthread_one_node();

  // Gather result C
  // MPI_Request request3;
  MPI_Igather(C, row_per_node * N, MPI_FLOAT, C, row_per_node * N, MPI_FLOAT, 0, MPI_COMM_WORLD, &requests[2]);
  // MPI_Wait(&request3, MPI_STATUS_IGNORE);
}

// Function to use pthread on one node
void pthread_one_node()
{
  // Multi-thread on 1 node
  pthread_t threads[num_threads];
  pthread_attr_t attr[num_threads];
  cpu_set_t cpus[num_threads];
  for (long i = 0; i < num_threads; i++)
  {
    pthread_attr_init(&attr[i]);
    CPU_ZERO(&cpus[i]);
    CPU_SET(i, &cpus[i]);

    // Limit thread to run only on 1 core
    pthread_attr_setaffinity_np(&attr[i], sizeof(cpu_set_t), &cpus[i]);
    pthread_create(&threads[i], &attr[i], mat_mul_thread, (void *)i);
  }

  // Sync threads
  for (int i = 0; i < num_threads; i++)
  {
    pthread_join(threads[i], NULL);
  }
}

void *mat_mul_thread(void *args)
{
  long my_args = (long)args;
  int startM = row_per_node / num_threads * my_args;
  int endM = row_per_node / num_threads * (my_args + 1);
  // memset(C, 0, row_per_node * N * sizeof(float));

  int evenK = K / KTILESIZE * KTILESIZE;
  for (int kk = 0; kk < evenK; kk += KTILESIZE)
  {
    int KTileBoundary1 = std::min(kk + KTILESIZE, K);
    for (int mm = startM; mm < endM; mm += MTILESIZE)
    {
      int MTileBoundary = std::min(mm + MTILESIZE, endM);
      for (int nn = 0; nn < N; nn += NTILESIZE)
      {
        int NTileBoundary = std::min(nn + NTILESIZE, N);

        for (int k = kk; k < KTileBoundary1; k += 6)
        {
          for (int m = mm; m < MTileBoundary; m++)
          {
            __m256 a1, a2, a3, a4, a5, a6;
            __m256 c1;
            a1 = _mm256_set1_ps(A[m * K + k + 0]);
            a2 = _mm256_set1_ps(A[m * K + k + 1]);
            a3 = _mm256_set1_ps(A[m * K + k + 2]);
            a4 = _mm256_set1_ps(A[m * K + k + 3]);
            a5 = _mm256_set1_ps(A[m * K + k + 4]);
            a6 = _mm256_set1_ps(A[m * K + k + 5]);

            for (int n = nn; n < NTileBoundary; n += 8)
            {
              c1 = _mm256_load_ps(&(C[m * N + n]));

              __m256 b1, b2, b3, b4, b5, b6;
              b1 = _mm256_load_ps(&B[(k + 0) * N + n]);
              b2 = _mm256_load_ps(&B[(k + 1) * N + n]);
              b3 = _mm256_load_ps(&B[(k + 2) * N + n]);
              b4 = _mm256_load_ps(&B[(k + 3) * N + n]);
              b5 = _mm256_load_ps(&B[(k + 4) * N + n]);
              b6 = _mm256_load_ps(&B[(k + 5) * N + n]);

              c1 = _mm256_fmadd_ps(a1, b1, c1);
              c1 = _mm256_fmadd_ps(a2, b2, c1);
              c1 = _mm256_fmadd_ps(a3, b3, c1);
              c1 = _mm256_fmadd_ps(a4, b4, c1);
              c1 = _mm256_fmadd_ps(a5, b5, c1);
              c1 = _mm256_fmadd_ps(a6, b6, c1);

              _mm256_store_ps(&(C[m * N + n]), c1);
            }
          }
        }
      }
    }
  }

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
          __m256 a1, a2, a3, a4, a5, a6;
          __m256 c1;
          a1 = _mm256_set1_ps(A[m * K + k + 0]);
          a2 = _mm256_set1_ps(A[m * K + k + 1]);
          a3 = _mm256_set1_ps(A[m * K + k + 2]);
          a4 = _mm256_set1_ps(A[m * K + k + 3]);

          for (int n = nn; n < NTileBoundary; n += 8)
          {
            c1 = _mm256_load_ps(&(C[m * N + n]));

            __m256 b1, b2, b3, b4, b5, b6;
            b1 = _mm256_load_ps(&B[(k + 0) * N + n]);
            b2 = _mm256_load_ps(&B[(k + 1) * N + n]);
            b3 = _mm256_load_ps(&B[(k + 2) * N + n]);
            b4 = _mm256_load_ps(&B[(k + 3) * N + n]);

            c1 = _mm256_fmadd_ps(a1, b1, c1);
            c1 = _mm256_fmadd_ps(a2, b2, c1);
            c1 = _mm256_fmadd_ps(a3, b3, c1);
            c1 = _mm256_fmadd_ps(a4, b4, c1);

            _mm256_store_ps(&(C[m * N + n]), c1);
          }
        }
      }
    }
  }
  pthread_exit(NULL);
}




/*
MPI no SIMD 1700
*/
#include "mat_mul.h"

#include <cstdio>
#include <cstdlib>
#include <mpi.h>
#include <immintrin.h>
#include "util.h"

#define MTILESIZE 32
#define NTILESIZE 512
#define KTILESIZE 512

// static variables used in 1 process
static float *A, *B, *C;
static int M, N, K;
static int num_threads;
static int mpi_rank, mpi_world_size;
static int row_per_node;

void pthread_one_node();
void *mat_mul_thread(void *args);

void mat_mul(float *_A, float *_B, float *_C, int _M, int _N, int _K,
             int _num_threads, int _mpi_rank, int _mpi_world_size)
{
  // distribute portion of A to nodes
  row_per_node = _M / _mpi_world_size;

  A = _A, B = _B, C = _C;
  M = _M, N = _N, K = _K;
  num_threads = _num_threads, mpi_rank = _mpi_rank, mpi_world_size = _mpi_world_size;

  MPI_Request requests[3];

  // Scatter A
  // MPI_Request request;
  MPI_Iscatter(A, row_per_node * K, MPI_FLOAT, A, row_per_node * K, MPI_FLOAT, 0, MPI_COMM_WORLD, &requests[0]);
  // MPI_Wait(&request, MPI_STATUS_IGNORE);

  // Broadcast B
  // MPI_Request request2;
  MPI_Ibcast(B, K * N, MPI_FLOAT, 0, MPI_COMM_WORLD, &requests[1]);
  MPI_Wait(&requests[1], MPI_STATUS_IGNORE);

  pthread_one_node();

  // Gather result C
  // MPI_Request request3;
  MPI_Igather(C, row_per_node * N, MPI_FLOAT, C, row_per_node * N, MPI_FLOAT, 0, MPI_COMM_WORLD, &requests[2]);
  // MPI_Wait(&request3, MPI_STATUS_IGNORE);
}

// Function to use pthread on one node
void pthread_one_node()
{
  // Multi-thread on 1 node
  pthread_t threads[num_threads];
  pthread_attr_t attr[num_threads];
  cpu_set_t cpus[num_threads];
  for (long i = 0; i < num_threads; i++)
  {
    pthread_attr_init(&attr[i]);
    CPU_ZERO(&cpus[i]);
    CPU_SET(i, &cpus[i]);

    // Limit thread to run only on 1 core
    pthread_attr_setaffinity_np(&attr[i], sizeof(cpu_set_t), &cpus[i]);
    pthread_create(&threads[i], &attr[i], mat_mul_thread, (void *)i);
  }

  // Sync threads
  for (int i = 0; i < num_threads; i++)
  {
    pthread_join(threads[i], NULL);
  }
}

void *mat_mul_thread(void *args)
{
  long my_args = (long)args;
  int startM = row_per_node / num_threads * my_args;
  int endM = row_per_node / num_threads * (my_args + 1);
  // memset(C, 0, row_per_node * N * sizeof(float));

  for (int kk = 0; kk < K; kk += KTILESIZE)
  {
    int KTileBoundary1 = std::min(kk + KTILESIZE, K);
    for (int mm = startM; mm < endM; mm += MTILESIZE)
    {
      int MTileBoundary = std::min(mm + MTILESIZE, endM);
      for (int nn = 0; nn < N; nn += NTILESIZE)
      {
        int NTileBoundary = std::min(nn + NTILESIZE, N);

        for (int k = kk; k < KTileBoundary1; k += 4)
        {
          for (int m = mm; m < MTileBoundary; m++)
          {
            float a1 = A[m * K + k + 0];
            float a2 = A[m * K + k + 1];
            float a3 = A[m * K + k + 2];
            float a4 = A[m * K + k + 3];

            for (int n = nn; n < NTileBoundary; n++)
            {
              float b1 = B[(k + 0) * N + n];
              float b2 = B[(k + 1) * N + n];
              float b3 = B[(k + 2) * N + n];
              float b4 = B[(k + 3) * N + n];

              C[m * N + n] += a1 * b1 + a2 * b2 + a3 * b3 + a4 * b4;
            }
          }
        }
      }
    }
  }

  pthread_exit(NULL);
}


/*
MPI with pthread > 2000 GFLOPS

*/
#include "mat_mul.h"

#include <cstdio>
#include <cstdlib>
#include <mpi.h>
#include <immintrin.h>
#include "util.h"

#define MTILESIZE 48
#define NTILESIZE 512
#define KTILESIZE 512

// static variables used in 1 process
static float *A, *B, *C;
static int M, N, K;
static int num_threads;
static int mpi_rank, mpi_world_size;
static int row_per_node;

void pthread_one_node();
void *mat_mul_thread(void *args);

void mat_mul(float *_A, float *_B, float *_C, int _M, int _N, int _K,
             int _num_threads, int _mpi_rank, int _mpi_world_size)
{
  // distribute portion of A to nodes
  row_per_node = _M / _mpi_world_size;

  A = _A, B = _B, C = _C;
  M = _M, N = _N, K = _K;
  num_threads = _num_threads, mpi_rank = _mpi_rank, mpi_world_size = _mpi_world_size;

  MPI_Request requests[3];

  // Scatter A
  // MPI_Request request;
  MPI_Iscatter(A, row_per_node * K, MPI_FLOAT, A, row_per_node * K, MPI_FLOAT, 0, MPI_COMM_WORLD, &requests[0]);
  // MPI_Wait(&request, MPI_STATUS_IGNORE);

  // Broadcast B
  // MPI_Request request2;
  MPI_Ibcast(B, K * N, MPI_FLOAT, 0, MPI_COMM_WORLD, &requests[1]);
  MPI_Wait(&requests[1], MPI_STATUS_IGNORE);

  pthread_one_node();

  // Gather result C
  // MPI_Request request3;
  MPI_Igather(C, row_per_node * N, MPI_FLOAT, C, row_per_node * N, MPI_FLOAT, 0, MPI_COMM_WORLD, &requests[2]);
  // MPI_Wait(&request3, MPI_STATUS_IGNORE);
}

// Function to use pthread on one node
void pthread_one_node()
{
  // Multi-thread on 1 node
  pthread_t threads[num_threads];
  pthread_attr_t attr[num_threads];
  cpu_set_t cpus[num_threads];
  for (long i = 0; i < num_threads; i++)
  {
    pthread_attr_init(&attr[i]);
    CPU_ZERO(&cpus[i]);
    CPU_SET(i, &cpus[i]);

    // Limit thread to run only on 1 core
    pthread_attr_setaffinity_np(&attr[i], sizeof(cpu_set_t), &cpus[i]);
    pthread_create(&threads[i], &attr[i], mat_mul_thread, (void *)i);
  }

  // Sync threads
  for (int i = 0; i < num_threads; i++)
  {
    pthread_join(threads[i], NULL);
  }
}

void *mat_mul_thread(void *args)
{
  long my_args = (long)args;
  int startM = row_per_node / num_threads * my_args;
  int endM = row_per_node / num_threads * (my_args + 1);
  // memset(C, 0, row_per_node * N * sizeof(float));

  int evenK = K / KTILESIZE * KTILESIZE;
  for (int kk = 0; kk < evenK; kk += KTILESIZE)
  {
    int KTileBoundary1 = std::min(kk + KTILESIZE, K);
    for (int mm = startM; mm < endM; mm += MTILESIZE)
    {
      int MTileBoundary = std::min(mm + MTILESIZE, endM);
      for (int nn = 0; nn < N; nn += NTILESIZE)
      {
        int NTileBoundary = std::min(nn + NTILESIZE, N);

        for (int k = kk; k < KTileBoundary1; k += 6)
        {
          for (int m = mm; m < MTileBoundary; m++)
          {
            float a1 = A[m * K + k + 0];
            float a2 = A[m * K + k + 1];
            float a3 = A[m * K + k + 2];
            float a4 = A[m * K + k + 3];
            float a5 = A[m * K + k + 4];
            float a6 = A[m * K + k + 5];

            for (int n = nn; n < NTileBoundary; n++)
            {
              float b1 = B[(k + 0) * N + n];
              float b2 = B[(k + 1) * N + n];
              float b3 = B[(k + 2) * N + n];
              float b4 = B[(k + 3) * N + n];
              float b5 = B[(k + 4) * N + n];
              float b6 = B[(k + 5) * N + n];

              C[m * N + n] += a1 * b1 + a2 * b2 + a3 * b3 + a4 * b4 + a5 * b5 + a6 * b6;
            }
          }
        }
      }
    }
  }

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
          float a2 = A[m * K + k + 1];
          float a3 = A[m * K + k + 2];
          float a4 = A[m * K + k + 3];

          for (int n = nn; n < NTileBoundary; n++)
          {
            float b1 = B[(k + 0) * N + n];
            float b2 = B[(k + 1) * N + n];
            float b3 = B[(k + 2) * N + n];
            float b4 = B[(k + 3) * N + n];

            C[m * N + n] += a1 * b1 + a2 * b2 + a3 * b3 + a4 * b4;
          }
        }
      }
    }
  }

  pthread_exit(NULL);
}