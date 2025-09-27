#ifndef DGEMM_H
#define DGEMM_H

// Multiplicacao sequencial
void dgemm_sequencial(const double* A, const double* B, double* C, int n, int k, int m);

// Multiplicacao paralela usando OpenMP (versão simples, sem otimizações).
void dgemm_parallel(const double* A, const double* B, double* C, int n, int k, int m);

// Multiplicacao paralela usando OpenMP com otimizações (transposição de B e tiling).
void dgemm_parallel_opt(const double* A, const double* B, double* C, int n, int k, int m);

#endif /* DGEMM_H */
