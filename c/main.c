#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>
#include "dgemm.h"

/**
 * Lê uma matriz do arquivo 'filename'.
 * Aloca dinamicamente um vetor double (row-major) e retorna ponteiro.
 * Preenche rows_out e cols_out com as dimensões.
 * Em caso de erro retorna NULL.
 */
double* read_matrix(const char* filename, int* rows_out, int* cols_out) {
    FILE* f = fopen(filename, "r");
    if (!f) {
        fprintf(stderr, "Erro: nao foi possivel abrir '%s'\n", filename);
        return NULL;
    }

    int rows, cols;
    if (fscanf(f, "%d %d", &rows, &cols) != 2) {
        fprintf(stderr, "Erro: formato invalido na primeira linha de '%s'\n", filename);
        fclose(f);
        return NULL;
    }

    // Aloca uma matriz representada como um vetor unidimensional
    double* mat = (double*) malloc((size_t)rows * (size_t)cols * sizeof(double));
    if (!mat) {
        fprintf(stderr, "Erro: malloc falhou para matriz %s (%d x %d)\n", filename, rows, cols);
        fclose(f);
        return NULL;
    }

    // Lê valores (espera rows*cols valores). fscanf ignora whitespace.
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            double val;
            if (fscanf(f, "%lf", &val) != 1) {
                fprintf(stderr, "Erro: faltam valores em '%s' (esperado %d x %d)\n", filename, rows, cols);
                free(mat);
                fclose(f);
                return NULL;
            }
            mat[i * cols + j] = val; // row-major
        }
    }

    fclose(f);
    *rows_out = rows;
    *cols_out = cols;
    return mat;
}

int main(int argc, char* argv[]) {
    const char* fileA = "A.txt";
    const char* fileB = "B.txt";
    const char* method = "sequencial";
    int num_threads = 1; // padrão: 1 thread

    if (argc >= 4) {
        method = argv[1];
        fileA = argv[2];
        fileB = argv[3];
        if (argc >= 5) {
            num_threads = atoi(argv[4]);
            if (num_threads < 1) num_threads = 1;
        }
    } else {
        fprintf(stderr, "Aviso: usando arquivos padrao: %s %s\n", fileA, fileB);
        fprintf(stderr, "Uso: %s [sequencial|parallel|parallel_opt] A.txt B.txt [num_threads]\n", argv[0]);
    }

    int A_rows, A_cols, B_rows, B_cols;
    double* A = read_matrix(fileA, &A_rows, &A_cols);
    if (!A) return EXIT_FAILURE;
    double* B = read_matrix(fileB, &B_rows, &B_cols);
    if (!B) { free(A); return EXIT_FAILURE; }

    if (A_cols != B_rows) {
        fprintf(stderr, "Erro: dimensoes incompatíveis para multiplicacao: A é %d x %d, B é %d x %d\n", A_rows, A_cols, B_rows, B_cols);
        free(A); free(B);
        return EXIT_FAILURE;
    }

    int n = A_rows;
    int k = A_cols;
    int m = B_cols;

    double* C = (double*) malloc((size_t)n * (size_t)m * sizeof(double));
    if (!C) {
        fprintf(stderr, "Erro: malloc falhou para matriz resultado (%d x %d)\n", n, m);
        free(A); free(B);
        return EXIT_FAILURE;
    }

    // Tempo inicial
    double start_time = omp_get_wtime();

    // Seleciona o método de multiplicação
    if (strcmp(method, "sequencial") == 0) {
        dgemm_sequencial(A, B, C, n, k, m);
    } else if (strcmp(method, "parallel") == 0) {
        omp_set_num_threads(num_threads);

        fprintf(stderr, "Threads utilizados = %d\n", omp_get_max_threads());
        dgemm_parallel(A, B, C, n, k, m);
    } else if (strcmp(method, "parallel_opt") == 0) {
        omp_set_num_threads(num_threads);

        fprintf(stderr, "Threads utilizados = %d\n", omp_get_max_threads());
        dgemm_parallel_opt(A, B, C, n, k, m);
    } else {
        fprintf(stderr, "Erro: método desconhecido '%s'. Use 'sequencial', 'parallel' ou 'parallel_opt'.\n", method);
        free(A); free(B); free(C);
        return EXIT_FAILURE;
    }

    // Tempo final
    double end_time = omp_get_wtime();
    double elapsed = end_time - start_time;

    printf("%.6f", elapsed);

    free(A); free(B); free(C);

    return EXIT_SUCCESS;
}