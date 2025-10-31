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

// Variáveis de teste
static const int TEST_M = 4;
static const int TEST_N = 4;
static const int TEST_K = 4;
static const double A_init[16] = {
    1.0, 2.0, 3.0, 4.0,
    5.0, 6.0, 7.0, 8.0,
    9.0, 10.0, 11.0, 12.0,
    13.0, 14.0, 15.0, 16.0
};
static const double B_init[16] = {
    16.0, 15.0, 14.0, 13.0,
    12.0, 11.0, 10.0, 9.0,
    8.0, 7.0, 6.0, 5.0,
    4.0, 3.0, 2.0, 1.0
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
    double A[16], B[16];
    memcpy(A, A_init, sizeof(A));
    memcpy(B, B_init, sizeof(B));

    double *C = fn(A, B, M, N, K, threads);
    assert(C != NULL);

    double C_ref[16];
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                M, N, K,
                1.0, A, K,
                B, N,
                0.0, C_ref, N);

    compare_and_report(C, C_ref, (size_t)M * N, name);

    free(C);
    printf("%s passed.\n", name);
}

void test_dgemm_serial() {
    run_test(serial_adapter, "test_dgemm_serial", 0);
}

void test_dgemm_parallel_openmp() {
    run_test(openmp_adapter, "test_dgemm_parallel_openmp", 2);
}

void test_dgemm_parallel_mpi() {
    int provided = 0, initialized = 0;
    MPI_Initialized(&initialized);
    if (!initialized) {
        MPI_Init_thread(NULL, NULL, MPI_THREAD_FUNNELED, &provided);
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
