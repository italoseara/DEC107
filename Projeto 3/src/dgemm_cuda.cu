#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include "../include/dgemm.h"

// Versão básica: cada thread calcula um elemento de C
__global__ void dgemm_kernel_basic(const double* __restrict__ A,
                                   const double* __restrict__ B,
                                   double* __restrict__ C,
                                   int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row >= M || col >= N) return;

    double acc = 0.0;
    for (int k = 0; k < K; ++k) {
        acc += A[(size_t)row * (size_t)K + (size_t)k] *
               B[(size_t)k * (size_t)N + (size_t)col];
    }
    C[(size_t)row * (size_t)N + (size_t)col] = acc;
}

// Kernel de multiplicação de matrizes em precisão dupla com tiling em memória compartilhada.
// Kernel otimizado: tiling com memória compartilhada, tamanho de tile definido por blockDim
__global__ void dgemm_kernel_shared(const double* __restrict__ A,
                                    const double* __restrict__ B,
                                    double* __restrict__ C,
                                    int M, int N, int K) {
    const int T = blockDim.x; // assume bloco T x T
    int row = blockIdx.y * T + threadIdx.y;
    int col = blockIdx.x * T + threadIdx.x;

    double acc = 0.0;

    extern __shared__ double smem[]; // tamanho: 2 * T*T
    double* As = smem;
    double* Bs = smem + T*T;

    int numTiles = (K + T - 1) / T;
    for (int t = 0; t < numTiles; ++t) {
        int kA = t * T + threadIdx.x; // coluna dentro de A
        int kB = t * T + threadIdx.y; // linha dentro de B

        // Carrega tile de A em [ty*T + tx]
        if (row < M && kA < K) {
            As[threadIdx.y * T + threadIdx.x] = A[(size_t)row * (size_t)K + kA];
        } else {
            As[threadIdx.y * T + threadIdx.x] = 0.0;
        }
        // Carrega tile de B em [ty*T + tx]
        if (kB < K && col < N) {
            Bs[threadIdx.y * T + threadIdx.x] = B[(size_t)kB * (size_t)N + col];
        } else {
            Bs[threadIdx.y * T + threadIdx.x] = 0.0;
        }
        __syncthreads();

        // Acumula produto parcial
        #pragma unroll
        for (int k = 0; k < T; ++k) {
            acc += As[threadIdx.y * T + k] * Bs[k * T + threadIdx.x];
        }
        __syncthreads();
    }

    if (row < M && col < N) {
        C[(size_t)row * (size_t)N + col] = acc;
    }
}

extern "C" double* dgemm_parallel_cuda_shared(const double* A, const double* B, int M, int N, int K, int tile) {
    if (M <= 0 || N <= 0 || K <= 0 || !A || !B) {
        return NULL;
    }

    if (tile <= 0 || tile > 32) tile = 32; // limita a 32 por segurança

    size_t sizeA = (size_t)M * (size_t)K * sizeof(double);
    size_t sizeB = (size_t)K * (size_t)N * sizeof(double);
    size_t sizeC = (size_t)M * (size_t)N * sizeof(double);

    double *dA = nullptr, *dB = nullptr, *dC = nullptr;
    cudaError_t err;

    err = cudaMalloc((void**)&dA, sizeA);
    if (err != cudaSuccess) return NULL;
    err = cudaMalloc((void**)&dB, sizeB);
    if (err != cudaSuccess) { cudaFree(dA); return NULL; }
    err = cudaMalloc((void**)&dC, sizeC);
    if (err != cudaSuccess) { cudaFree(dA); cudaFree(dB); return NULL; }

    err = cudaMemcpy(dA, A, sizeA, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) { cudaFree(dA); cudaFree(dB); cudaFree(dC); return NULL; }
    err = cudaMemcpy(dB, B, sizeB, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) { cudaFree(dA); cudaFree(dB); cudaFree(dC); return NULL; }

    dim3 block(tile, tile);
    dim3 grid((N + tile - 1) / tile, (M + tile - 1) / tile);

    size_t shmemBytes = 2ull * (size_t)tile * (size_t)tile * sizeof(double);
    dgemm_kernel_shared<<<grid, block, shmemBytes>>>(dA, dB, dC, M, N, K);
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        cudaFree(dA); cudaFree(dB); cudaFree(dC);
        return NULL;
    }

    double* C = (double*)malloc(sizeC);
    if (!C) {
        cudaFree(dA); cudaFree(dB); cudaFree(dC);
        return NULL;
    }
    err = cudaMemcpy(C, dC, sizeC, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        free(C); C = NULL;
    }
    cudaFree(dA); cudaFree(dB); cudaFree(dC);
    return C;
}

extern "C" double* dgemm_parallel_cuda_basic(const double* A, const double* B, int M, int N, int K, int tile) {
    if (M <= 0 || N <= 0 || K <= 0 || !A || !B) return NULL;
    if (tile <= 0 || tile > 32) tile = 32; // limitar

    size_t sizeA = (size_t)M * (size_t)K * sizeof(double);
    size_t sizeB = (size_t)K * (size_t)N * sizeof(double);
    size_t sizeC = (size_t)M * (size_t)N * sizeof(double);

    double *dA = nullptr, *dB = nullptr, *dC = nullptr;
    cudaError_t err;

    err = cudaMalloc((void**)&dA, sizeA); if (err != cudaSuccess) return NULL;
    err = cudaMalloc((void**)&dB, sizeB); if (err != cudaSuccess) { cudaFree(dA); return NULL; }
    err = cudaMalloc((void**)&dC, sizeC); if (err != cudaSuccess) { cudaFree(dA); cudaFree(dB); return NULL; }

    err = cudaMemcpy(dA, A, sizeA, cudaMemcpyHostToDevice); if (err != cudaSuccess) { cudaFree(dA); cudaFree(dB); cudaFree(dC); return NULL; }
    err = cudaMemcpy(dB, B, sizeB, cudaMemcpyHostToDevice); if (err != cudaSuccess) { cudaFree(dA); cudaFree(dB); cudaFree(dC); return NULL; }

    dim3 block(tile, tile);
    dim3 grid((N + tile - 1) / tile, (M + tile - 1) / tile);

    dgemm_kernel_basic<<<grid, block>>>(dA, dB, dC, M, N, K);
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) { cudaFree(dA); cudaFree(dB); cudaFree(dC); return NULL; }

    double* C = (double*)malloc(sizeC);
    if (!C) { cudaFree(dA); cudaFree(dB); cudaFree(dC); return NULL; }
    err = cudaMemcpy(C, dC, sizeC, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) { free(C); C = NULL; }
    cudaFree(dA); cudaFree(dB); cudaFree(dC);
    return C;
}
