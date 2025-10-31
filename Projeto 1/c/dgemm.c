#include "dgemm.h"
#include <stdlib.h>
#include <stdio.h>

#include <omp.h>

#ifndef MIN
#define MIN(a,b) ((a) < (b) ? (a) : (b))
#endif

/**
 * Versão sequencial de DGEMM para comparação.
 */
void dgemm_sequencial(const double* A, const double* B, double* C, int n, int k, int m) {
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < m; ++j) {
            double sum = 0.0;
            for (int p = 0; p < k; ++p) {
                sum += A[i * k + p] * B[p * m + j];
            }
            C[i * m + j] = sum;
        }
    }
}

/**
 * Versão paralela simples de DGEMM usando OpenMP
 * Cada thread calcula um subconjunto das linhas de C.
 */
void dgemm_parallel(const double* A, const double* B, double* C, int n, int k, int m) {
    #pragma omp parallel // Inicia a região paralela
    {
        #pragma omp for // Divide as iterações do loop entre as threads
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < m; ++j) {
                double sum = 0.0;
                for (int p = 0; p < k; ++p) {
                    sum += A[i * k + p] * B[p * m + j];
                }
                C[i * m + j] = sum;
            }
        }

        #pragma omp barrier // Sincroniza as threads antes de sair da região paralela
    }
}

/**
 * Versão sequencial otimizada de DGEMM usando transposição de B e tiling.
 */
void dgemm_sequencial_opt(const double* A, const double* B, double* C, int n, int k, int m) 
{
    // Transpor B para acesso contíguo. Bt tem dimensões m x k (linha j é a coluna j de B).
    double* Bt = malloc((size_t)m * (size_t)k * sizeof(double));
    if (!Bt) {
        fprintf(stderr, "Erro: malloc Bt\n");
        return;
    }

    for (int p = 0; p < k; ++p) {
        for (int j = 0; j < m; ++j) {
            Bt[j * k + p] = B[p * m + j];
        }
    }

    // Inicializa C com zeros.
    for (int i = 0; i < n * m; ++i) {
        C[i] = 0.0;
    }

    // Tamanho do bloco (tile size)
    const int BS = 64;

    // Multiplicação em blocos.
    for (int ii = 0; ii < n; ii += BS) {
        for (int jj = 0; jj < m; jj += BS) {
            int i_max = MIN(n, ii + BS);
            int j_max = MIN(m, jj + BS);

            // Percorre o bloco (ii,jj) de C
            for (int kk = 0; kk < k; kk += BS) {
                int k_max = MIN(k, kk + BS);

                // Multiplica o bloco A[ii:i_max, kk:k_max] com o bloco Bt[jj:j_max, kk:k_max]
                for (int i = ii; i < i_max; ++i) {
                    const double* Ai = A + i * k + kk;
                    double* Ci = C + i * m;

                    for (int j = jj; j < j_max; ++j) {
                        const double* Bj = Bt + j * k + kk;
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
}

/**
 * Versão paralela otimizada de DGEMM usando OpenMP, transposição de B e tiling.
 */
void dgemm_parallel_opt(const double* A, const double* B, double* C, int n, int k, int m)
{
    // Transpor B para acesso contíguo. Bt tem dimensões m x k (linha j é a coluna j de B).
    double* Bt = malloc((size_t)m * (size_t)k * sizeof(double));
    if (!Bt) {
        fprintf(stderr, "Erro: malloc Bt\n");
        return;
    }

    #pragma omp parallel for // Paraleliza a transposição de B
    for (int p = 0; p < k; ++p) {
        for (int j = 0; j < m; ++j) {
            Bt[j * k + p] = B[p * m + j];
        }
    }

    // Inicializa C com zeros.
    #pragma omp parallel for // Paraleliza a inicialização de C
    for (int i = 0; i < n * m; ++i) {
        C[i] = 0.0;
    }

    // Tamanho do bloco (tile size)
    const int BS = 64;

    // Multiplicação em blocos paralela.
    #pragma omp parallel // Inicia a região paralela
    {
        // collapse(2) para paralelizar os dois loops externos (ii, jj)
        // schedule(static) para distribuir blocos de forma estática entre threads
        // Cada thread calcula blocos de C
        #pragma omp for collapse(2) schedule(static)
        for (int ii = 0; ii < n; ii += BS) {
            for (int jj = 0; jj < m; jj += BS) {
                int i_max = MIN(n, ii + BS);
                int j_max = MIN(m, jj + BS);

                // Percorre o bloco (ii,jj) de C
                for (int kk = 0; kk < k; kk += BS) {
                    int k_max = MIN(k, kk + BS);

                    // Multiplica o bloco A[ii:i_max, kk:k_max] com o bloco Bt[jj:j_max, kk:k_max]
                    for (int i = ii; i < i_max; ++i) {
                        const double* Ai = A + i * k + kk;
                        double* Ci = C + i * m;

                        for (int j = jj; j < j_max; ++j) {
                            const double* Bj = Bt + j * k + kk;
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
}