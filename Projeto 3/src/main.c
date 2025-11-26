/**
 * Executor simples para variantes de DGEMM (serial, openmp, mpi)
 * 
 * Uso (prefira via Makefile):
 *  ./dgemm --alg [serial|openmp|mpi] --threads <n> --m <M> --n <N> --k <K>
 * 
 * Imprime apenas o tempo decorrido em segundos no stdout.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <mpi.h>
#include <omp.h>

#include "../include/dgemm.h"

typedef enum { ALG_SERIAL = 0, ALG_OPENMP = 1, ALG_MPI = 2, ALG_CUDA = 3 } algorithm_t;

static void* xmalloc(size_t nbytes) {
	void* p = malloc(nbytes);
	if (!p) {
		fprintf(stderr, "malloc failed for %zu bytes\n", nbytes);
		exit(1);
	}
	return p;
}

static void fill_matrix(double* A, int rows, int cols, int seed) {
	// Padrão simples determinístico para evitar custo elevado de inicialização
	// A[i,j] = (i*cols + j + seed) % 100 / 10.0
	size_t idx = 0;
	for (int i = 0; i < rows; ++i) {
		for (int j = 0; j < cols; ++j) {
			A[idx++] = ((double)(((i * cols + j + seed) % 100))) / 10.0;
		}
	}
}

int main(int argc, char** argv) {
	// Padrões
	algorithm_t alg = ALG_SERIAL;
	int threads = 1; // para OpenMP ou como dica de ranks
	int tile = 32;   // tamanho do tile para CUDA (Option A)
	int M = 512, N = 512, K = 512;

	// Parseia argumentos (parser bem simples)
	for (int i = 1; i < argc; ++i) {
		if (!strcmp(argv[i], "--alg") && i + 1 < argc) {
			++i;
			if (!strcmp(argv[i], "serial")) alg = ALG_SERIAL;
			else if (!strcmp(argv[i], "openmp")) alg = ALG_OPENMP;
			else if (!strcmp(argv[i], "mpi")) alg = ALG_MPI;
			else if (!strcmp(argv[i], "cuda")) alg = ALG_CUDA;
			else {
				fprintf(stderr, "Unknown --alg value: %s\n", argv[i]);
				return 2;
			}
		} else if (!strcmp(argv[i], "--threads") && i + 1 < argc) {
			threads = atoi(argv[++i]);
			if (threads <= 0) threads = 1;
		} else if (!strcmp(argv[i], "--m") && i + 1 < argc) {
			M = atoi(argv[++i]);
		} else if (!strcmp(argv[i], "--n") && i + 1 < argc) {
			N = atoi(argv[++i]);
		} else if (!strcmp(argv[i], "--k") && i + 1 < argc) {
			K = atoi(argv[++i]);
		} else if (!strcmp(argv[i], "--tile") && i + 1 < argc) {
			tile = atoi(argv[++i]);
			if (tile <= 0) tile = 32;
		} else if (!strcmp(argv[i], "-h") || !strcmp(argv[i], "--help")) {
			fprintf(stderr, "Usage: %s --alg [serial|openmp|mpi|cuda] --threads N --m M --n N --k K [--tile T]\n", argv[0]);
			return 0;
		} else {
			fprintf(stderr, "Unknown argument: %s\n", argv[i]);
			return 2;
		}
	}

	// Aloca e inicializa entradas A (M x K) e B (K x N)
	double* A = (double*)xmalloc((size_t)M * (size_t)K * sizeof(double));
	double* B = (double*)xmalloc((size_t)K * (size_t)N * sizeof(double));
	fill_matrix(A, M, K, 1);
	fill_matrix(B, K, N, 7);

	double elapsed = 0.0;
	double* C = NULL;

	if (alg == ALG_MPI) {
		int initialized = 0;
		MPI_Initialized(&initialized);
		if (!initialized) {
			MPI_Init(&argc, &argv);
		}
		int world_rank = 0, world_size = 1;
		MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
		MPI_Comm_size(MPI_COMM_WORLD, &world_size);

		double t0 = MPI_Wtime();
		C = dgemm_parallel_mpi(A, B, M, N, K);
		double t1 = MPI_Wtime();
		elapsed = t1 - t0;

		// Apenas o rank 0 recebe C e deve imprimir o tempo
		if (world_rank == 0) {
			if (!C) {
				fprintf(stderr, "dgemm_parallel_mpi retornou NULL\n");
				free(A);
				free(B);
				MPI_Finalize();
				return 1;
			}
			printf("%.9f\n", elapsed);
			fflush(stdout);
			free(C);
		}

		free(A);
		free(B);
		MPI_Finalize();
		return 0;
	}

	// Caminho Serial: medir com clock_gettime; OpenMP: medir com omp_get_wtime
	double t0 = omp_get_wtime();
	if (alg == ALG_SERIAL) {
		C = dgemm_serial(A, B, M, N, K);
	} else if (alg == ALG_OPENMP) {
		omp_set_num_threads(threads);
		C = dgemm_parallel_openmp(A, B, M, N, K);
	} else if (alg == ALG_CUDA) {
#ifdef ENABLE_CUDA
		C = dgemm_parallel_cuda(A, B, M, N, K, tile);
#else
		fprintf(stderr, "CUDA selecionado mas compilação sem suporte (ENABLE_CUDA ausente)\n");
		free(A); free(B);
		return 3;
#endif
	}
	double t1 = omp_get_wtime();
	elapsed = t1 - t0;

	if (!C) {
		fprintf(stderr, "dgemm returned NULL (allocation or input error)\n");
		free(A);
		free(B);
		return 1;
	}

	// Imprime apenas o tempo decorrido em segundos
	printf("%.9f\n", elapsed);
	fflush(stdout);

	free(A);
	free(B);
	free(C);
	return 0;
}
