#include "mat_mul.h"

#include <cstdio>
#include <cstdlib>
#include <mpi.h>
#include <pthread.h>
#include <numa.h>
#include <sched.h>
#include <numaif.h>

static float *A, *B, *C;
static int M, N, K;
static int num_threads;
static int mpi_rank, mpi_world_size, rows_per_node;
static MPI_Request request_s, request_b, request_g;

#define ITILESIZE (32)
#define JTILESIZE (1024)
#define KTILESIZE (512)

#define min(a,b) (((a)<(b))?(a):(b))

void *mat_mul_thread(void* arg);

void mat_mul_pthread() {

  pthread_t tid[num_threads];

  pthread_attr_t attr[num_threads];
  cpu_set_t cpus[num_threads];

  for (int i=0; i<num_threads; i++) {
    pthread_attr_init(&attr[i]);
    CPU_ZERO(&cpus[i]);
    CPU_SET(i, &cpus[i]);
    pthread_attr_setaffinity_np(&attr[i], sizeof(cpu_set_t), &cpus[i]);
    pthread_create(&tid[i], &attr[i], mat_mul_thread, (void*)i);
  }

  // Sync
  for (int i=0; i<num_threads; i++) {
    pthread_join(tid[i], NULL); 
  } 
}

void* mat_mul_thread(void* arg) {
  int tid = (long)arg;

  int is = rows_per_node / num_threads * tid + min(tid, rows_per_node % num_threads);
  int ie = rows_per_node / num_threads * (tid+1) + min(tid+1, rows_per_node % num_threads);

  int i, j, k;
  int ii, jj, kk;

  // MxK x KxN

  // Submatrices
  for (kk=0; kk<K; kk+=KTILESIZE) { 
    for (ii=is; ii<ie; ii+=ITILESIZE) {
      for (jj=0; jj<N; jj+=JTILESIZE) {
        // Mat mul
        for (k=kk; k<min(kk+KTILESIZE, K); k+=4){
          for (i=ii; i<min(ii+ITILESIZE, M); i++){

            float a0 = A[i * K + (k + 0)];
            float a1 = A[i * K + (k + 1)];
            float a2 = A[i * K + (k + 2)];
            float a3 = A[i * K + (k + 3)];
            
            for (j=jj; j<min(jj+JTILESIZE, N); j+=1){
              float b0 = B[(k + 0) * N + (j + 0)];
              float b1 = B[(k + 1) * N + (j + 0)];
              float b2 = B[(k + 2) * N + (j + 0)];
              float b3 = B[(k + 3) * N + (j + 0)];
              float c0 = a0*b0 + a1*b1 + a2*b2 + a3*b3;
    
              C[i * N + j] += c0;

            }
          }
        }
      }
    }
  }
  return NULL;
}

void mat_mul(float *_A, float *_B, float *_C, int _M, int _N, int _K,
             int _num_threads, int _mpi_rank, int _mpi_world_size) {
  A = _A, B = _B, C = _C;
  M = _M, N = _N, K = _K;
  num_threads = _num_threads, mpi_rank = _mpi_rank,
  mpi_world_size = _mpi_world_size;

  rows_per_node = M / mpi_world_size;

  // Scatter A and broadcast B
  MPI_Iscatter(A, rows_per_node * K, MPI_FLOAT, A, rows_per_node * K, MPI_FLOAT, 0, MPI_COMM_WORLD, &request_s);
  MPI_Ibcast(B, K * N, MPI_FLOAT, 0, MPI_COMM_WORLD, &request_b);

  MPI_Wait(&request_s, MPI_STATUS_IGNORE);
  MPI_Wait(&request_b, MPI_STATUS_IGNORE);

  mat_mul_pthread();

  MPI_Igather(C, rows_per_node * N, MPI_FLOAT, C, rows_per_node * N, MPI_FLOAT, 0, MPI_COMM_WORLD, &request_g);
  MPI_Wait(&request_g, MPI_STATUS_IGNORE);
}