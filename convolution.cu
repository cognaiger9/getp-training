#include <cstdio>
#include <cstdlib>

#include "convolution.cuh"

#include <omp.h>
#include <mpi.h>
#include <nccl.h>

#define CHECK_CUDA(call)                                                       \
    do {                                                                       \
        cudaError_t status_ = call;                                            \
        if (status_ != cudaSuccess) {                                          \
            fprintf(stderr, "CUDA error (%s:%d): %s:%s\n", __FILE__, __LINE__, \
                    cudaGetErrorName(status_), cudaGetErrorString(status_));   \
            exit(EXIT_FAILURE);                                                \
        }                                                                      \
    } while (0)

static int n_gpus;
static half *I_gpu[64], *F_gpu[64], *BUF1_gpu[64]; // miners, probably
static float *O_gpu[64];

static MPI_Datatype mpi_half;

__global__ void gpu_im2col(half *_I, half *workspace, int N, int C, int H, int W,
                      int R, int S, int pad_h, int pad_w, int stride_h,
                      int stride_w, int dilation_h, int dilation_w) {

    const int ON = N;
    const int OH = 1 + (H + 2 * pad_h - (((R - 1) * dilation_h) + 1)) / stride_h;
    const int OW = 1 + (W + 2 * pad_w - (((S - 1) * dilation_w) + 1)) / stride_w;

    const int tidx = blockIdx.x * blockDim.x + threadIdx.x;

    if (tidx >= (ON * OH * OW)) return;

    const int on = tidx / (OH * OW);
    const int oh = (tidx % (OH * OW)) / OW;
    const int ow = (tidx % (OH * OW)) % OW;

    // For each channel
    for (int c = 0; c < C; ++c) {
        // For each filter row
        for (int r = 0; r < R; ++r) {
            // For each filter column
            #pragma unroll
            for (int s = 0; s < S; ++s) {
                const int n = on;
                const int h = oh * stride_h - pad_h + r * dilation_h;
                const int w = ow * stride_w - pad_w + s * dilation_w;

                if (h < 0 || h >= H || w < 0 || w >= W)
                    continue;

                workspace[((c * R * S) + (r * S) + s) * (ON * OH * OW) +
                            (on * OH * OW + oh * OW + ow)] =
                    _I[n * C * H * W + c * H * W + h * W + w];
            }
        }
    }
}

template <int BLOCK_SIZE>
__global__ void tiling_gpu_matmul(half *input_A, half *input_B, float *output, int M, int N, int K, int OH, int OW, int K_) {
    int thrIdx = threadIdx.x; // 0 ~ BLOCK_SIZE
    int thrIdy = threadIdx.y; // 0 ~ BLOCK_SIZE

    int col = blockIdx.x * BLOCK_SIZE + thrIdx;
    int row = blockIdx.y * BLOCK_SIZE + thrIdy;

    float sum = 0.0f;

    __shared__ half A_tile[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ half B_tile[BLOCK_SIZE][BLOCK_SIZE+1];

    for (int phase = 0; phase < (K + BLOCK_SIZE - 1) / BLOCK_SIZE; phase++) {

        A_tile[thrIdy][thrIdx] = input_A[row * K + phase * BLOCK_SIZE + thrIdx];
        B_tile[thrIdy][thrIdx] = input_B[(phase * BLOCK_SIZE + thrIdy) * N + col];

        // Wait for all threads to finish copying
        __syncthreads();

        // Do the fma operation
        #pragma unroll BLOCK_SIZE
        for (int i = 0; i < BLOCK_SIZE; i++) {
            // sum += A_tile[thrIdy][i] * B_tile[i][thrIdx];
            sum += __half2float(A_tile[thrIdy][i]) * __half2float(B_tile[i][thrIdx]);
        }

        // Wait for all threads to finish computing
        __syncthreads();
    }

    // Now checked here
    if (row >= M || col >= N) {
        return;
    }
    
    // Reshape output
    int kIndex = row; // row index in original matrix
    int nIndex = col / (OH * OW); // col index in original matrix

    int new_index = nIndex * K_ * OH * OW + kIndex * OH * OW + col % (OH * OW);
    output[new_index] = sum;
}

void convolution(half *_I, half *_F, float *_O, half *_BUF1, float *_BUF2, int N,
                 int C, int H, int W, int K, int R, int S, int pad_h, int pad_w,
                 int stride_h, int stride_w, int dilation_h, int dilation_w,
                 int mpi_rank, int mpi_size) {


    // Broadcast F
    // MPI_Bcast(_F, K * C * R * S, mpi_half, 0, MPI_COMM_WORLD);
    MPI_Request F_br_req;
    MPI_Ibcast(_F, K * C * R * S, mpi_half, 0, MPI_COMM_WORLD, &F_br_req);

    // MPI_Bcast(_I, N * C * H * W, mpi_half, 0, MPI_COMM_WORLD);

    // Scatter I
    // int MPI_N_start[mpi_size], MPI_N_end[mpi_size];
    // for (int i = 0; i < mpi_size; ++i) {
    //     MPI_N_start[i] = i * N / mpi_size;
    //     MPI_N_end[i] = min((i + 1) * N / mpi_size, N);
    // }

    // MPI_Scatterv(I, MPI_N_end, MPI_N_start, mpi_half, I, MPI_N_end[mpi_rank] - MPI_N_start[mpi_rank], mpi_half, 0, MPI_COMM_WORLD);
    // MPI_Scatter(_I, N * C * H * W, mpi_half, _I, N * C * H * W, mpi_half, 0, MPI_COMM_WORLD); 

     
    // Can't scatter, guess it is MPI_Send time

    int MPI_N_starts[mpi_size], MPI_N_ends[mpi_size];
    for (int i = 0; i < mpi_size; ++i) {
        MPI_N_starts[i] = i * N / mpi_size;
        MPI_N_ends[i] = min((i + 1) * N / mpi_size, N);
    }

    MPI_Request I_send_req[mpi_size], I_recv_req[mpi_size];

    if (mpi_rank == 0) {
        for (int i = 1; i < mpi_size; ++i) {
            // MPI_Send(_I + MPI_N_starts[i] * C * H * W, (MPI_N_ends[i] - MPI_N_starts[i]) * C * H * W, mpi_half, i, 0, MPI_COMM_WORLD);
            MPI_Isend(_I + MPI_N_starts[i] * C * H * W, (MPI_N_ends[i] - MPI_N_starts[i]) * C * H * W, mpi_half, i, 0, MPI_COMM_WORLD, &I_send_req[i]);
        }
    } else {
        // MPI_Recv(_I, (MPI_N_ends[mpi_rank] - MPI_N_starts[mpi_rank]) * C * H * W, mpi_half, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Irecv(_I, (MPI_N_ends[mpi_rank] - MPI_N_starts[mpi_rank]) * C * H * W, mpi_half, 0, 0, MPI_COMM_WORLD, &I_recv_req[mpi_rank]);
    }

    // Wait for broadcast and scatter to finish
    MPI_Wait(&F_br_req, MPI_STATUS_IGNORE);
    if (mpi_rank != 0) {
        MPI_Wait(&I_recv_req[mpi_rank], MPI_STATUS_IGNORE);
    }

    if (mpi_rank == 0) {
        MPI_Waitall(mpi_size - 1, I_send_req + 1, MPI_STATUS_IGNORE);
    }
    

    const int thread_bound = MPI_N_ends[mpi_rank] - MPI_N_starts[mpi_rank];

    half *I = _I, *F = _F, *BUF1 = _BUF1;
    float *O = _O;
    
    cudaStream_t upload[n_gpus], compute[n_gpus], download[n_gpus];

    #pragma omp parallel for shared(upload, compute, download)
    for (int i = 0; i < n_gpus; ++i) {
        CHECK_CUDA(cudaSetDevice(i));
        CHECK_CUDA(cudaStreamCreate(&upload[i]));
        CHECK_CUDA(cudaStreamCreate(&compute[i]));
        CHECK_CUDA(cudaStreamCreate(&download[i]));
    }

    cudaEvent_t upload_done[n_gpus], compute_done[n_gpus], download_done[n_gpus];

    const int OH = 1 + (H + 2 * pad_h - (((R - 1) * dilation_h) + 1)) / stride_h;
    const int OW = 1 + (W + 2 * pad_w - (((S - 1) * dilation_w) + 1)) / stride_w;


    int N_begin[n_gpus], N_end[n_gpus];

    #pragma omp parallel for shared(N_begin, N_end, I, F, O, BUF1, upload_done, compute_done, download_done, upload, compute, download)
    for (int i = 0; i < n_gpus; ++i) {
        CHECK_CUDA(cudaSetDevice(i));

        cudaEventCreate(&upload_done[i]);
        cudaEventCreate(&compute_done[i]);
        cudaEventCreate(&download_done[i]);

        N_begin[i] = i * thread_bound / n_gpus;
        N_end[i] = min((i + 1) * thread_bound / n_gpus, thread_bound);

        CHECK_CUDA(cudaMemcpyAsync(F_gpu[i], F, K * C * R * S * sizeof(half), cudaMemcpyHostToDevice, upload[i]));
    }

    #pragma omp parallel for shared(N_begin, N_end, I, F, O, BUF1, upload_done, compute_done, download_done, upload, compute, download)
    for (int i = 0; i < n_gpus; ++i) {
        CHECK_CUDA(cudaSetDevice(i));
        CHECK_CUDA(cudaMemcpyAsync(I_gpu[i], I + N_begin[i] * C * H * W, (N_end[i] - N_begin[i]) * C * H * W * sizeof(half), cudaMemcpyHostToDevice, upload[i]));
        CHECK_CUDA(cudaEventRecord(upload_done[i], upload[i]));
    }


    #pragma omp parallel for shared(N_begin, N_end, I, F, O, BUF1, upload_done, compute_done, download_done, upload, compute, download)
    for (int i = 0; i < n_gpus; ++i) {
        CHECK_CUDA(cudaSetDevice(i));
        CHECK_CUDA(cudaStreamWaitEvent(compute[i], upload_done[i], 0));
        dim3 im2col_block(768);
        dim3 im2col_grid(((N_end[i] - N_begin[i]) * OH * OW + im2col_block.x - 1) / im2col_block.x);
        gpu_im2col<<<im2col_grid, im2col_block, 0, compute[i]>>>(I_gpu[i], BUF1_gpu[i], N_end[i] - N_begin[i], C, H, W, R, S, pad_h, pad_w, stride_h, stride_w, dilation_h, dilation_w);

        
        dim3 matmul_block(32, 32);
        dim3 matmul_grid(((N_end[i] - N_begin[i]) * OH * OW + matmul_block.x - 1) / matmul_block.x, (K + matmul_block.y - 1) / matmul_block.y);
        tiling_gpu_matmul<32><<<matmul_grid, matmul_block, 0, compute[i]>>>(F_gpu[i], BUF1_gpu[i], O_gpu[i], K, (N_end[i] - N_begin[i]) * OH * OW, C * R * S, OH, OW, K);

        CHECK_CUDA(cudaEventRecord(compute_done[i], compute[i]));
    }

    #pragma omp parallel for shared(N_begin, N_end, I, F, O, BUF1, upload_done, compute_done, download_done, upload, compute, download)
    for (int i = 0; i < n_gpus; ++i) {
        // CHECK_CUDA(cudaStreamWaitEvent(download, compute_done[i], 0));
        // CHECK_CUDA(cudaMemcpyAsync(O + N_begin[i] * K * OH * OW, O_gpu + N_begin[i] * K * OH * OW, (N_end[i] - N_begin[i]) * K * OH * OW * sizeof(float), cudaMemcpyDeviceToHost, download));
        // CHECK_CUDA(cudaEventRecord(download_done[i], download));

        CHECK_CUDA(cudaSetDevice(i));
        CHECK_CUDA(cudaStreamWaitEvent(download[i], compute_done[i], 0));
        CHECK_CUDA(cudaMemcpyAsync(O + N_begin[i] * K * OH * OW, O_gpu[i], (N_end[i] - N_begin[i]) * K * OH * OW * sizeof(float), cudaMemcpyDeviceToHost, download[i]));
        CHECK_CUDA(cudaEventRecord(download_done[i], download[i]));
    }

    #pragma omp parallel for shared(N_begin, N_end, I, F, O, BUF1, upload_done, compute_done, download_done, upload, compute, download)
    for (int i = 0; i < n_gpus; ++i) {
        CHECK_CUDA(cudaSetDevice(i));
        // CHECK_CUDA(cudaStreamWaitEvent(download[i], download_done[i], 0));
        CHECK_CUDA(cudaDeviceSynchronize());
    }

    // Gather O (should work, it's float32)
    int *recvcounts = new int[mpi_size];
    int *displs = new int[mpi_size];

    for (int i = 0; i < mpi_size; ++i) {
        recvcounts[i] = (MPI_N_ends[i] - MPI_N_starts[i]) * K * OH * OW;
        displs[i] = MPI_N_starts[i] * K * OH * OW;
    }

    MPI_Gatherv(O, (MPI_N_ends[mpi_rank] - MPI_N_starts[mpi_rank]) * K * OH * OW, MPI_FLOAT, O, recvcounts, displs, MPI_FLOAT, 0, MPI_COMM_WORLD);
}

void convolution_initialize(int N, int C, int H, int W, int K, int R, int S,
                            int pad_h, int pad_w, int stride_h, int stride_w,
                            int dilation_h, int dilation_w) {
    const int OH = 1 + (H + 2 * pad_h - (((R - 1) * dilation_h) + 1)) / stride_h;
    const int OW = 1 + (W + 2 * pad_w - (((S - 1) * dilation_w) + 1)) / stride_w;

    CHECK_CUDA(cudaGetDeviceCount(&n_gpus));
    
    for (int i = 0; i < n_gpus; ++i) {
        /*
            While we can definitely divide by n_gpu here to save some (massive sometimes) memory,
            the issue is that it's gonna need special handling when n_gpu doesn't divide N. 
            Either waste memory or risk your career, pick your poison, the deadline was 5 days.
        */
        CHECK_CUDA(cudaSetDevice(i));
        CHECK_CUDA(cudaMalloc(&I_gpu[i], N * C * H * W * sizeof(half)));
        CHECK_CUDA(cudaMalloc(&F_gpu[i], K * C * R * S * sizeof(half)));
        CHECK_CUDA(cudaMalloc(&BUF1_gpu[i], N * C * R * S * OH * OW * sizeof(half)));

        CHECK_CUDA(cudaMalloc(&O_gpu[i], N * K * OH * OW * sizeof(float)));
        
    }
    // create a new data type for MPI to use
    MPI_Type_contiguous(sizeof(half), MPI_BYTE, &mpi_half);
    MPI_Type_commit(&mpi_half);
}

void convolution_cleanup(half *_I, half *_F, float *_O, int N, int C, int H,
                         int W, int K, int R, int S, int pad_h, int pad_w,
                         int stride_h, int stride_w, int dilation_h,
                         int dilation_w) {
                            
    for (int i = 0; i < n_gpus; ++i) {
        CHECK_CUDA(cudaSetDevice(i));
        CHECK_CUDA(cudaFree(I_gpu[i]));
        CHECK_CUDA(cudaFree(F_gpu[i]));
        CHECK_CUDA(cudaFree(BUF1_gpu[i]));
        CHECK_CUDA(cudaFree(O_gpu[i]));
    }
}
