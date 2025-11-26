#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <omp.h>
#include <mpi.h>

#include "../include/dgemm.h"

#ifndef MIN
#define MIN(a,b) ((a) < (b) ? (a) : (b))
#endif

double* dgemm_serial(const double* A, const double* B, int M, int N, int K) {
    // Verifica entradas inválidas.
    if (M <= 0 || N <= 0 || K <= 0 || !A || !B) {
        return NULL;
    }

    // Aloca espaço para a matriz resultado C (dimensões M x N).
    double* C = malloc((size_t)M * (size_t)N * sizeof(double));
    if (!C) {
        fprintf(stderr, "Erro: malloc C\n");
        return NULL;
    }

    // Transpor B para acesso contíguo. Bt tem dimensões N x K (linha j é a coluna j de B).
    double* Bt = malloc((size_t)N * (size_t)K * sizeof(double));
    if (!Bt) {
        fprintf(stderr, "Erro: malloc Bt\n");
        free(C);
        return NULL;
    }

    for (int p = 0; p < K; ++p) {
        for (int j = 0; j < N; ++j) {
            Bt[j * K + p] = B[p * N + j];
        }
    }

    // Inicializa C com zeros.
    for (int i = 0; i < N * M; ++i) {
        C[i] = 0.0;
    }

    // Tamanho do bloco (tile size)
    const int BS = 64;

    // Multiplicação em blocos.
    for (int ii = 0; ii < M; ii += BS) {
        for (int jj = 0; jj < N; jj += BS) {
            int i_max = MIN(M, ii + BS);
            int j_max = MIN(N, jj + BS);

            // Percorre o bloco (ii,jj) de C
            for (int kk = 0; kk < K; kk += BS) {
                int k_max = MIN(K, kk + BS);

                // Multiplica o bloco A[ii:i_max, kk:k_max] com o bloco Bt[jj:j_max, kk:k_max]
                for (int i = ii; i < i_max; ++i) {
                    const double* Ai = A + i * K + kk;
                    double* Ci = C + i * N;

                    for (int j = jj; j < j_max; ++j) {
                        const double* Bj = Bt + j * K + kk;
                        double sum = 0.0;

                        for (int p = 0; p < k_max - kk; ++p) {
                            sum += Ai[p] * Bj[p];
                        }
                        Ci[j] += sum;
                    }
                }
            }
        }
    }

    free(Bt);
    return C;
}

double* dgemm_parallel_openmp(const double* A, const double* B, int M, int N, int K) {
    // Verifica entradas inválidas.
    if (M <= 0 || N <= 0 || K <= 0 || !A || !B) {
        return NULL;
    }

    // Aloca espaço para a matriz resultado C (dimensões M x N).
    double* C = malloc((size_t)M * (size_t)N * sizeof(double));
    if (!C) {
        fprintf(stderr, "Erro: malloc C\n");
        return NULL;
    }

    // Transpor B para acesso contíguo. Bt tem dimensões N x K (linha j é a coluna j de B).
    double* Bt = malloc((size_t)N * (size_t)K * sizeof(double));
    if (!Bt) {
        fprintf(stderr, "Erro: malloc Bt\n");
        free(C);
        return NULL;
    }

    #pragma omp parallel for// Paraleliza a transposição de B
    for (int p = 0; p < K; ++p) {
        for (int j = 0; j < N; ++j) {
            Bt[j * K + p] = B[p * N + j];
        }
    }

    // Inicializa C com zeros.
    #pragma omp parallel for // Paraleliza a inicialização de C
    for (int i = 0; i < N * M; ++i) {
        C[i] = 0.0;
    }

    // Tamanho do bloco (tile size)
    const int BS = 64;

    // Multiplicação em blocos paralela.
    #pragma omp parallel // Inicia a região paralela com num_threads
    {
        // collapse(2) para paralelizar os dois loops externos (ii, jj)
        // schedule(static) para distribuir blocos de forma estática entre threads
        #pragma omp for collapse(2) schedule(static)
        for (int ii = 0; ii < M; ii += BS) {
            for (int jj = 0; jj < N; jj += BS) {
                int i_max = MIN(M, ii + BS);
                int j_max = MIN(N, jj + BS);

                // Percorre o bloco (ii,jj) de C
                for (int kk = 0; kk < K; kk += BS) {
                    int k_max = MIN(K, kk + BS);

                    // Multiplica o bloco A[ii:i_max, kk:k_max] com o bloco Bt[jj:j_max, kk:k_max]
                    for (int i = ii; i < i_max; ++i) {
                        const double* Ai = A + i * K + kk;
                        double* Ci = C + i * N;

                        for (int j = jj; j < j_max; ++j) {
                            const double* Bj = Bt + j * K + kk;
                            double sum = 0.0;

                            #pragma omp simd reduction(+:sum) // Vetoriza o loop interno e garante que a soma seja correta
                            for (int p = 0; p < k_max - kk; ++p) {
                                sum += Ai[p] * Bj[p];
                            }
                            Ci[j] += sum;
                        }
                    }
                }
            }
        }

        #pragma omp barrier // Sincroniza as threads antes de sair da região paralela
    }

    free(Bt);
    return C;
}

double* dgemm_parallel_mpi(const double* A, const double* B, int M, int N, int K) {
    int initialized = 0;
    MPI_Initialized(&initialized);
    if (!initialized) {
        fprintf(stderr, "MPI must be initialized before calling dgemm_parallel_mpi\n");
        return NULL;
    }

    int world_rank = 0, world_size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    /* validação básica: broadcast de "ok" a partir do root */
    int ok = (M > 0 && N > 0 && K > 0);
    if (world_rank == 0) ok = ok && (A != NULL) && (B != NULL);
    MPI_Bcast(&ok, 1, MPI_INT, 0, MPI_COMM_WORLD);
    if (!ok) return NULL;

    /* Distribuição de linhas */
    int base = M / world_size;
    int rem = M % world_size;
    int local_rows = base + (world_rank < rem ? 1 : 0);

    /* prepara vetores de counts e displacements para scatter/gather (em número de doubles) */
    int *sendcounts_A = (int*)malloc((size_t)world_size * sizeof(int));
    int *displs_A     = (int*)malloc((size_t)world_size * sizeof(int));
    int *recvcounts_C = (int*)malloc((size_t)world_size * sizeof(int));
    int *displs_C     = (int*)malloc((size_t)world_size * sizeof(int));
    if (!sendcounts_A || !displs_A || !recvcounts_C || !displs_C) {
        fprintf(stderr, "malloc for MPI arrays failed (rank %d)\n", world_rank);
        free(sendcounts_A); free(displs_A); free(recvcounts_C); free(displs_C);
        return NULL;
    }

    int offsetA = 0, offsetC = 0;
    for (int r = 0; r < world_size; ++r) {
        int rows_r = base + (r < rem ? 1 : 0);
        sendcounts_A[r] = rows_r * K;         /* número de doubles */
        displs_A[r]     = offsetA;
        offsetA        += sendcounts_A[r];

        recvcounts_C[r] = rows_r * N;         /* número de doubles */
        displs_C[r]     = offsetC;
        offsetC        += recvcounts_C[r];
    }

    /* aloca buffer local A e faz scatter das linhas */
    double* A_local = NULL;
    if (local_rows > 0) {
        A_local = (double*)malloc((size_t)local_rows * (size_t)K * sizeof(double));
        if (!A_local) {
            fprintf(stderr, "malloc A_local failed (rank %d)\n", world_rank);
            free(sendcounts_A); free(displs_A); free(recvcounts_C); free(displs_C);
            return NULL;
        }
    }

    MPI_Scatterv((void*)A, sendcounts_A, displs_A, MPI_DOUBLE,
                 A_local, local_rows * K, MPI_DOUBLE,
                 0, MPI_COMM_WORLD);

    /* Broadcast de B (K x N). Usa um buffer contíguo B_full para que a root possa copiar de B. */
    double* B_full = (double*)malloc((size_t)K * (size_t)N * sizeof(double));
    if (!B_full) {
        fprintf(stderr, "malloc B_full failed (rank %d)\n", world_rank);
        free(A_local); free(sendcounts_A); free(displs_A); free(recvcounts_C); free(displs_C);
        return NULL;
    }
    if (world_rank == 0) {
        /* copia B para B_full (cópia defensiva caso B não seja contígua) */
        memcpy(B_full, B, (size_t)K * (size_t)N * sizeof(double));
    }
    MPI_Bcast(B_full, K * N, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    /* Cria Bt transposta (N x K) em cada rank para acesso contíguo por coluna da B original */
    double* Bt = (double*)malloc((size_t)N * (size_t)K * sizeof(double));
    if (!Bt) {
        fprintf(stderr, "malloc Bt failed (rank %d)\n", world_rank);
        free(B_full); free(A_local); free(sendcounts_A); free(displs_A); free(recvcounts_C); free(displs_C);
        return NULL;
    }
    for (int p = 0; p < K; ++p) {
        const double* Brow = B_full + (size_t)p * (size_t)N;
        for (int j = 0; j < N; ++j) {
            Bt[(size_t)j * (size_t)K + p] = Brow[j];
        }
    }

    /* Computa C_local = A_local * B  (dimensões local_rows x N)
     * Usando Bt para acesso contíguo nos loops internos.
     */
    double* C_local = NULL;
    if (local_rows > 0) {
        C_local = (double*)malloc((size_t)local_rows * (size_t)N * sizeof(double));
        if (!C_local) {
            fprintf(stderr, "malloc C_local failed (rank %d)\n", world_rank);
            free(Bt); free(B_full); free(A_local); free(sendcounts_A); free(displs_A); free(recvcounts_C); free(displs_C);
            return NULL;
        }
        for (size_t i = 0; i < (size_t)local_rows * (size_t)N; ++i) C_local[i] = 0.0;

        const int BS = 64; /* parâmetro de ajuste */

        for (int ii = 0; ii < local_rows; ii += BS) {
            int i_max = MIN(local_rows, ii + BS);
            for (int jj = 0; jj < N; jj += BS) {
                int j_max = MIN(N, jj + BS);
                for (int kk = 0; kk < K; kk += BS) {
                    int k_max = MIN(K, kk + BS);
                    for (int i = ii; i < i_max; ++i) {
                        const double* Ai = A_local + (size_t)i * (size_t)K + kk;
                        double* Ci = C_local + (size_t)i * (size_t)N;
                        for (int j = jj; j < j_max; ++j) {
                            const double* Bj = Bt + (size_t)j * (size_t)K + kk; /* bloco K contíguo */
                            double sum = 0.0;
                            for (int p = 0; p < (k_max - kk); ++p) {
                                sum += Ai[p] * Bj[p];
                            }
                            Ci[j] += sum;
                        }
                    }
                }
            }
        }
    } else {
        /* local_rows == 0: participa nas coletivas mas não computa */
        C_local = NULL;
    }

    /* Gatherv dos resultados para a raiz */
    double* C_full = NULL;
    if (world_rank == 0) {
        C_full = (double*)malloc((size_t)M * (size_t)N * sizeof(double));
        if (!C_full) {
            fprintf(stderr, "malloc C_full failed (rank 0)\n");
            /* libera tudo e retorna NULL */
            free(C_local); free(Bt); free(B_full); free(A_local);
            free(sendcounts_A); free(displs_A); free(recvcounts_C); free(displs_C);
            return NULL;
        }
    }

    MPI_Gatherv(C_local, local_rows * N, MPI_DOUBLE,
                C_full, recvcounts_C, displs_C, MPI_DOUBLE,
                0, MPI_COMM_WORLD);

    /* Limpa buffers locais (mantém C_full no rank 0) */
    if (C_local) free(C_local);
    free(Bt);
    free(B_full);
    if (A_local) free(A_local);
    free(sendcounts_A); free(displs_A); free(recvcounts_C); free(displs_C);

    return C_full; /* válido apenas no rank 0 */
}
