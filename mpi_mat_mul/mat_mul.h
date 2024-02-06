#pragma once
void *mat_mul_thread(void *args);
void pthread_one_node();
void mat_mul(float *_A, float *_B, float *_C,
             int _M, int _N, int _K, int _num_threads, int _mpi_rank, int _mpi_world_size);
