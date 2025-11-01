#include <stdlib.h>
#include <stdio.h>
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

double* dgemm_parallel_openmp(const double* A, const double* B, int M, int N, int K, int num_threads) {
    // Verifica entradas inválidas.
    if (M <= 0 || N <= 0 || K <= 0 || !A || !B) {
        return NULL;
    }

    if (num_threads <= 0) {
        num_threads = 1;
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

    #pragma omp parallel for num_threads(num_threads) // Paraleliza a transposição de B
    for (int p = 0; p < K; ++p) {
        for (int j = 0; j < N; ++j) {
            Bt[j * K + p] = B[p * N + j];
        }
    }

    // Inicializa C com zeros.
    #pragma omp parallel for num_threads(num_threads) // Paraleliza a inicialização de C
    for (int i = 0; i < N * M; ++i) {
        C[i] = 0.0;
    }

    // Tamanho do bloco (tile size)
    const int BS = 64;

    // Multiplicação em blocos paralela.
    #pragma omp parallel num_threads(num_threads) // Inicia a região paralela com num_threads
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
        // MPI deve ser inicializado pelo chamador (main). Caso contrário, falha.
        fprintf(stderr, "Erro: MPI não inicializado antes de dgemm_parallel_mpi.\n");
        return NULL;
    }

    int world_rank = 0, world_size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    // Validar parâmetros de entrada no rank 0 e comunicar aos demais.
    int ok = (M > 0 && N > 0 && K > 0);
    if (world_rank == 0) {
        ok = ok && (A != NULL) && (B != NULL);
    }
    MPI_Bcast(&ok, 1, MPI_INT, 0, MPI_COMM_WORLD);
    if (!ok) {
        return NULL;
    }

    // Distribuição de linhas de A entre os processos.
    int base = M / world_size;
    int rem  = M % world_size;

    // Linhas atribuídas a este rank
    int local_rows = base + (world_rank < rem ? 1 : 0);

    // Vetores de counts e displs para Scatterv/Gatherv
    int* sendcounts_A = (int*)malloc((size_t)world_size * sizeof(int));
    int* displs_A     = (int*)malloc((size_t)world_size * sizeof(int));
    int* recvcounts_C = (int*)malloc((size_t)world_size * sizeof(int));
    int* displs_C     = (int*)malloc((size_t)world_size * sizeof(int));
    if (!sendcounts_A || !displs_A || !recvcounts_C || !displs_C) {
        fprintf(stderr, "Erro: malloc para vetores MPI\n");
        free(sendcounts_A); 
        free(displs_A); 
        free(recvcounts_C); 
        free(displs_C);
        return NULL;
    }

    int offsetA = 0;
    int offsetC = 0;
    for (int r = 0; r < world_size; ++r) {
        int rows_r = base + (r < rem ? 1 : 0);
        sendcounts_A[r] = rows_r * K;  // elementos double
        displs_A[r]     = offsetA;
        offsetA        += sendcounts_A[r];

        recvcounts_C[r] = rows_r * N;  // elementos double
        displs_C[r]     = offsetC;
        offsetC        += recvcounts_C[r];
    }

    // Alocar buffers locais
    double* A_local = NULL;
    if (local_rows > 0) {
        A_local = (double*)malloc((size_t)local_rows * (size_t)K * sizeof(double));
        if (!A_local) {
            fprintf(stderr, "Erro: malloc A_local\n");
            free(sendcounts_A); free(displs_A); free(recvcounts_C); free(displs_C);
            return NULL;
        }
    }

    // Scatterv das linhas de A (cada processo recebe local_rows*K elementos)
    MPI_Scatterv((void*)A, sendcounts_A, displs_A, MPI_DOUBLE,
                 A_local, local_rows * K, MPI_DOUBLE,
                 0, MPI_COMM_WORLD);

    // Broadcast da matriz B inteira (K x N)
    double* B_full = (double*)malloc((size_t)K * (size_t)N * sizeof(double));
    if (!B_full) {
        fprintf(stderr, "Erro: malloc B_full\n");
        free(A_local);
        free(sendcounts_A); 
        free(displs_A); 
        free(recvcounts_C); 
        free(displs_C);
        return NULL;
    }
    if (world_rank == 0) {
        // Copiar dados de B para o buffer de broadcast
        for (int i = 0; i < K * N; ++i) B_full[i] = B[i];
    }
    MPI_Bcast(B_full, K * N, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // Computa C_local = A_local * B_full (dimensões local_rows x N)
    double* C_local = NULL;
    if (local_rows > 0) {
        C_local = (double*)malloc((size_t)local_rows * (size_t)N * sizeof(double));
        if (!C_local) {
            fprintf(stderr, "Erro: malloc C_local\n");
            free(B_full);
            free(A_local);
            free(sendcounts_A); 
            free(displs_A);
            free(recvcounts_C);
            free(displs_C);
            return NULL;
        }
        // Inicializa com zero
        for (int i = 0; i < local_rows * N; ++i) C_local[i] = 0.0;

        const int BS = 64; // bloco simples para cache
        for (int ii = 0; ii < local_rows; ii += BS) {
            int i_max = MIN(local_rows, ii + BS);
            for (int kk = 0; kk < K; kk += BS) {
                int k_max = MIN(K, kk + BS);
                for (int jj = 0; jj < N; jj += BS) {
                    int j_max = MIN(N, jj + BS);
                    for (int i = ii; i < i_max; ++i) {
                        const double* Ai = A_local + (size_t)i * (size_t)K + kk;
                        double* Ci = C_local + (size_t)i * (size_t)N + jj;
                        for (int j = jj; j < j_max; ++j) {
                            double sum = 0.0;
                            const double* Bcol = B_full + (size_t)kk * (size_t)N + j;
                            for (int p = 0; p < (k_max - kk); ++p) {
                                sum += Ai[p] * Bcol[(size_t)p * (size_t)N];
                            }
                            Ci[j - jj] += sum;
                        }
                    }
                }
            }
        }
    }

    // Alocar C no rank 0 e reunir C_local
    double* C_full = NULL;
    if (world_rank == 0) {
        C_full = (double*)malloc((size_t)M * (size_t)N * sizeof(double));
        if (!C_full) {
            fprintf(stderr, "Erro: malloc C_full\n");
            free(C_local);
            free(B_full);
            free(A_local);
            free(sendcounts_A);
            free(displs_A);
            free(recvcounts_C);
            free(displs_C);
            return NULL;
        }
    }

    MPI_Gatherv(C_local, local_rows * N, MPI_DOUBLE,
                 C_full, recvcounts_C, displs_C, MPI_DOUBLE,
                 0, MPI_COMM_WORLD);

    // Limpeza de buffers locais
    free(C_local);
    free(B_full);
    free(A_local);
    free(sendcounts_A);
    free(displs_A);
    free(recvcounts_C);
    free(displs_C);

    // Retornar C apenas no rank 0; demais retornam NULL.
    return C_full;
}