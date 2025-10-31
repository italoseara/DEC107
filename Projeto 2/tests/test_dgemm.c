/**
 * Teste de corretude para as implementações de DGEMM.
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <assert.h>
#include <cblas.h>
#include <mpi.h>

#include "../include/dgemm.h"

// Variáveis de teste (aumentadas)
static const int TEST_M = 8;
static const int TEST_N = 8;
static const int TEST_K = 8;
static const double A_init[64] = {
    1.0,  2.0,  3.0,  4.0,  5.0,  6.0,  7.0,  8.0,
    9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0,
    17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0,
    25.0, 26.0, 27.0, 28.0, 29.0, 30.0, 31.0, 32.0,
    33.0, 34.0, 35.0, 36.0, 37.0, 38.0, 39.0, 40.0,
    41.0, 42.0, 43.0, 44.0, 45.0, 46.0, 47.0, 48.0,
    49.0, 50.0, 51.0, 52.0, 53.0, 54.0, 55.0, 56.0,
    57.0, 58.0, 59.0, 60.0, 61.0, 62.0, 63.0, 64.0
};
static const double B_init[64] = {
    64.0, 63.0, 62.0, 61.0, 60.0, 59.0, 58.0, 57.0,
    56.0, 55.0, 54.0, 53.0, 52.0, 51.0, 50.0, 49.0,
    48.0, 47.0, 46.0, 45.0, 44.0, 43.0, 42.0, 41.0,
    40.0, 39.0, 38.0, 37.0, 36.0, 35.0, 34.0, 33.0,
    32.0, 31.0, 30.0, 29.0, 28.0, 27.0, 26.0, 25.0,
    24.0, 23.0, 22.0, 21.0, 20.0, 19.0, 18.0, 17.0,
    16.0, 15.0, 14.0, 13.0, 12.0, 11.0, 10.0,  9.0,
    8.0,  7.0,  6.0,  5.0,  4.0,  3.0,  2.0,  1.0
};

// Adaptadores para as funções dgemm
typedef double* (*dgemm_adapter_fn)(const double *A, const double *B, int M, int N, int K, int threads);

static double* serial_adapter(const double *A, const double *B, int M, int N, int K, int threads) {
    (void)threads;
    return dgemm_serial((double*)A, (double*)B, M, N, K);
}

static double* openmp_adapter(const double *A, const double *B, int M, int N, int K, int threads) {
    return dgemm_parallel_openmp((double*)A, (double*)B, M, N, K, threads);
}

static double* mpi_adapter(const double *A, const double *B, int M, int N, int K, int threads) {
    return dgemm_parallel_mpi((double*)A, (double*)B, M, N, K);
}

// Funções de teste
static void compare_and_report(const double *C, const double *C_ref, size_t elems, const char *name) {
    const double tol = 1e-9;
    for (size_t i = 0; i < elems; ++i) {
        double diff = fabs(C[i] - C_ref[i]);
        if (diff > tol) {
            fprintf(stderr, "%s failed at index %zu: got %.17g, expected %.17g, diff %.17g\n",
                    name, i, C[i], C_ref[i], diff);
            assert(0);
        }
    } 
}

static void run_test(dgemm_adapter_fn fn, const char *name, int threads) {
    const int M = TEST_M, N = TEST_N, K = TEST_K;
    size_t size_A = (size_t)M * (size_t)K;
    size_t size_B = (size_t)K * (size_t)N;
    size_t size_C = (size_t)M * (size_t)N;

    double *A = (double*)malloc(size_A * sizeof(double));
    double *B = (double*)malloc(size_B * sizeof(double));
    assert(A && B);
    memcpy(A, A_init, size_A * sizeof(double));
    memcpy(B, B_init, size_B * sizeof(double));

    double *C = fn(A, B, M, N, K, threads);
    assert(C != NULL);

    double *C_ref = (double*)malloc(size_C * sizeof(double));
    assert(C_ref);
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                M, N, K,
                1.0, A, (int)K,
                B, (int)N,
                0.0, C_ref, (int)N);

    compare_and_report(C, C_ref, size_C, name);

    free(C);
    free(C_ref);
    free(A);
    free(B);
    printf("%s passed.\n", name);
}

void test_dgemm_serial() {
    run_test(serial_adapter, "test_dgemm_serial", 0);
}

void test_dgemm_parallel_openmp() {
    run_test(openmp_adapter, "test_dgemm_parallel_openmp", 2);
}

void test_dgemm_parallel_mpi() {
    int initialized = 0;
    MPI_Initialized(&initialized);
    if (!initialized) {
        MPI_Init(NULL, NULL);
    }

    int world_rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    run_test(mpi_adapter, "test_dgemm_parallel_mpi", 0);

    if (!initialized) {
        MPI_Finalize();
    }
}

// Função main para executar os testes
int main() {
    test_dgemm_serial();
    test_dgemm_parallel_openmp();
    test_dgemm_parallel_mpi();
    printf("All tests passed.\n");
    return 0;
}
