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
#include <omp.h>

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

// Métrica de diferença relativa máxima conforme relatório
static double max_relative_diff(const double *X, const double *Y, size_t elems, double eps, size_t *idx_out) {
    double max_delta = 0.0;
    size_t max_idx = 0;
    for (size_t i = 0; i < elems; ++i) {
        double denom = fabs(Y[i]) + eps; // evitar divisão por zero
        double delta = fabs(X[i] - Y[i]) / denom;
        if (delta > max_delta) {
            max_delta = delta;
            max_idx = i;
        }
    }
    if (idx_out) *idx_out = max_idx;
    return max_delta;
}

void test_dgemm_serial() {
    const int M = TEST_M, N = TEST_N, K = TEST_K;
    const size_t size_A = (size_t)M * (size_t)K;
    const size_t size_B = (size_t)K * (size_t)N;
    const size_t size_C = (size_t)M * (size_t)N;

    double *A = (double*)malloc(size_A * sizeof(double));
    double *B = (double*)malloc(size_B * sizeof(double));
    assert(A && B);
    memcpy(A, A_init, size_A * sizeof(double));
    memcpy(B, B_init, size_B * sizeof(double));

    // Resultado sequencial
    double *C_seq = dgemm_serial((double*)A, (double*)B, M, N, K);
    assert(C_seq != NULL);

    // Referência BLAS
    double *C_blas = (double*)malloc(size_C * sizeof(double));
    assert(C_blas);
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                M, N, K,
                1.0, A, (int)K,
                B, (int)N,
                0.0, C_blas, (int)N);

    // Diferença relativa máxima
    size_t idx = 0;
    const double eps = 1e-12;
    const double tol = 1e-9;
    double delta = max_relative_diff(C_seq, C_blas, size_C, eps, &idx);
    if (delta > tol) {
        fprintf(stderr, "test_dgemm_serial failed at index %zu: got %.17g, ref %.17g, rel-diff %.17g\n",
                idx, C_seq[idx], C_blas[idx], delta);
        assert(0);
    }

    free(C_seq);
    free(C_blas);
    free(A);
    free(B);
    printf("test_dgemm_serial passed.\n");
}

void test_dgemm_parallel_openmp() {
    const int M = TEST_M, N = TEST_N, K = TEST_K;
    const size_t size_A = (size_t)M * (size_t)K;
    const size_t size_B = (size_t)K * (size_t)N;
    const size_t size_C = (size_t)M * (size_t)N;

    double *A = (double*)malloc(size_A * sizeof(double));
    double *B = (double*)malloc(size_B * sizeof(double));
    assert(A && B);
    memcpy(A, A_init, size_A * sizeof(double));
    memcpy(B, B_init, size_B * sizeof(double));

    // Referência: versão sequencial
    double *C_seq = dgemm_serial((double*)A, (double*)B, M, N, K);
    assert(C_seq != NULL);

    // Versão OpenMP
    int threads = 2; // número de threads para o teste
    omp_set_num_threads(threads);
    double *C_omp = dgemm_parallel_openmp((double*)A, (double*)B, M, N, K);
    assert(C_omp != NULL);

    // Diferença relativa máxima
    size_t idx = 0;
    const double eps = 1e-12;
    const double tol = 1e-9;
    double delta = max_relative_diff(C_omp, C_seq, size_C, eps, &idx);
    if (delta > tol) {
        fprintf(stderr, "test_dgemm_parallel_openmp failed at index %zu: got %.17g, seq %.17g, rel-diff %.17g\n",
                idx, C_omp[idx], C_seq[idx], delta);
        assert(0);
    }

    free(C_seq);
    free(C_omp);
    free(A);
    free(B);
    printf("test_dgemm_parallel_openmp passed.\n");
}

void test_dgemm_parallel_mpi() {
    int initialized = 0;
    MPI_Initialized(&initialized);
    if (!initialized) {
        MPI_Init(NULL, NULL);
    }

    int world_rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    const int M = TEST_M, N = TEST_N, K = TEST_K;
    const size_t size_A = (size_t)M * (size_t)K;
    const size_t size_B = (size_t)K * (size_t)N;
    const size_t size_C = (size_t)M * (size_t)N;

    double *A = (double*)malloc(size_A * sizeof(double));
    double *B = (double*)malloc(size_B * sizeof(double));
    assert(A && B);
    memcpy(A, A_init, size_A * sizeof(double));
    memcpy(B, B_init, size_B * sizeof(double));

    // Referência: calcular C_seq apenas no rank 0
    double *C_seq = NULL;
    if (world_rank == 0) {
        C_seq = dgemm_serial((double*)A, (double*)B, M, N, K);
        assert(C_seq != NULL);
    }

    // Versão MPI (apenas rank 0 recebe o resultado completo)
    double *C_mpi = dgemm_parallel_mpi((double*)A, (double*)B, M, N, K);

    if (world_rank == 0) {
        assert(C_mpi != NULL);
        size_t idx = 0;
        const double eps = 1e-12;
        const double tol = 1e-9;
        double delta = max_relative_diff(C_mpi, C_seq, size_C, eps, &idx);
        if (delta > tol) {
            fprintf(stderr, "test_dgemm_parallel_mpi failed at index %zu: got %.17g, seq %.17g, rel-diff %.17g\n",
                    idx, C_mpi[idx], C_seq[idx], delta);
            assert(0);
        }
        printf("test_dgemm_parallel_mpi passed.\n");
    }

    if (C_seq) free(C_seq);
    if (world_rank == 0 && C_mpi) free(C_mpi);
    free(A);
    free(B);

    if (!initialized) {
        MPI_Finalize();
    }
}

void test_dgemm_parallel_cuda_basic() {
    const int M = TEST_M, N = TEST_N, K = TEST_K;
    const size_t size_A = (size_t)M * (size_t)K;
    const size_t size_B = (size_t)K * (size_t)N;
    const size_t size_C = (size_t)M * (size_t)N;

    double *A = (double*)malloc(size_A * sizeof(double));
    double *B = (double*)malloc(size_B * sizeof(double));
    assert(A && B);
    memcpy(A, A_init, size_A * sizeof(double));
    memcpy(B, B_init, size_B * sizeof(double));

    double *C_seq = dgemm_serial((double*)A, (double*)B, M, N, K);
    assert(C_seq != NULL);

    double *C_cuda = dgemm_parallel_cuda_basic((double*)A, (double*)B, M, N, K, 16);
    assert(C_cuda != NULL);

    size_t idx = 0;
    const double eps = 1e-12;
    const double tol = 1e-9;
    double delta = max_relative_diff(C_cuda, C_seq, size_C, eps, &idx);
    if (delta > tol) {
        fprintf(stderr, "test_dgemm_parallel_cuda_basic failed at index %zu: got %.17g, seq %.17g, rel-diff %.17g\n",
                idx, C_cuda[idx], C_seq[idx], delta);
        assert(0);
    }
    free(C_seq);
    free(C_cuda);
    free(A);
    free(B);
    printf("test_dgemm_parallel_cuda_basic passed.\n");
}

void test_dgemm_parallel_cuda_shared() {
    const int M = TEST_M, N = TEST_N, K = TEST_K;
    const size_t size_A = (size_t)M * (size_t)K;
    const size_t size_B = (size_t)K * (size_t)N;
    const size_t size_C = (size_t)M * (size_t)N;

    double *A = (double*)malloc(size_A * sizeof(double));
    double *B = (double*)malloc(size_B * sizeof(double));
    assert(A && B);
    memcpy(A, A_init, size_A * sizeof(double));
    memcpy(B, B_init, size_B * sizeof(double));

    double *C_seq = dgemm_serial((double*)A, (double*)B, M, N, K);
    assert(C_seq != NULL);

    double *C_cuda = dgemm_parallel_cuda_shared((double*)A, (double*)B, M, N, K, 16);
    assert(C_cuda != NULL);

    size_t idx = 0;
    const double eps = 1e-12;
    const double tol = 1e-9;
    double delta = max_relative_diff(C_cuda, C_seq, size_C, eps, &idx);
    if (delta > tol) {
        fprintf(stderr, "test_dgemm_parallel_cuda_shared failed at index %zu: got %.17g, seq %.17g, rel-diff %.17g\n",
                idx, C_cuda[idx], C_seq[idx], delta);
        assert(0);
    }
    free(C_seq);
    free(C_cuda);
    free(A);
    free(B);
    printf("test_dgemm_parallel_cuda_shared passed.\n");
}

// Função main para executar os testes
int main() {
    test_dgemm_serial();
    test_dgemm_parallel_openmp();
    test_dgemm_parallel_mpi();
    test_dgemm_parallel_cuda_basic();
    test_dgemm_parallel_cuda_shared();

    int initialized = 0;
    MPI_Initialized(&initialized);
    if (!initialized) {
        // MPI nunca foi inicializado; é seguro imprimir
        printf("All tests passed.\n");
    } else {
        int finalized = 0;
        MPI_Finalized(&finalized);
        if (!finalized) {
            int world_rank = 0;
            MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
            if (world_rank == 0) {
                printf("All tests passed.\n");
            }
        } // se já finalizado, evitar chamadas e impressões relacionadas a MPI
    }
    return 0;
}
