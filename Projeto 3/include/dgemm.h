#ifndef DGEMM_H
#define DGEMM_H

#ifdef __cplusplus // Força a ligação C para compatibilidade com C++
extern "C" {
#endif

/**
 * @brief Computa o produto de matrizes C = A * B usando uma implementação de thread única.
 *
 * Realiza uma multiplicação geral de matrizes em precisão dupla onde
 * A possui dimensões M x K e B possui dimensões K x N. O resultado C possui
 * dimensões M x N e é armazenado em ordem por linhas (row-major).
 *
 * @param A Ponteiro para a primeira matriz de entrada A (tamanho M*K), em ordem por linhas.
 * @param B Ponteiro para a segunda matriz de entrada B (tamanho K*N), em ordem por linhas.
 * @param M Número de linhas de A e C.
 * @param N Número de colunas de B e C.
 * @param K Número de colunas de A e linhas de B.
 * @return Ponteiro para um array recém-alocado de comprimento M*N contendo C em
 *         ordem por linhas, ou NULL em caso de falha de alocação ou entrada inválida.
 *         O chamador é responsável por liberar o buffer retornado.
 */
double* dgemm_serial(const double* A, const double* B, int M, int N, int K);

/**
 * @brief Computa o produto de matrizes C = A * B usando paralelização com OpenMP.
 *
 * Versão paralela da multiplicação geral de matrizes. A possui
 * dimensões M x K e B possui dimensões K x N. O resultado C possui dimensões
 * M x N e é armazenado em ordem por linhas (row-major). O cálculo é paralelizado
 * usando OpenMP com o número solicitado de threads.
 *
 * @param A Ponteiro para a primeira matriz de entrada A (tamanho M*K), em ordem por linhas.
 * @param B Ponteiro para a segunda matriz de entrada B (tamanho K*N), em ordem por linhas.
 * @param M Número de linhas de A e C.
 * @param N Número de colunas de B e C.
 * @param K Número de colunas de A e linhas de B.
 * @return Ponteiro para um array recém-alocado de comprimento M*N contendo C em
 *         ordem por linhas, ou NULL em caso de falha de alocação ou entrada inválida.
 *         O chamador é responsável por liberar o buffer retornado.
 */
double* dgemm_parallel_openmp(const double* A, const double* B, int M, int N, int K);

/**
 * @brief Computa o produto de matrizes C = A * B usando paralelização com MPI.
 *
 * Versão paralela da multiplicação geral de matrizes. A possui
 * dimensões M x K e B possui dimensões K x N. O resultado C possui dimensões
 * M x N e é armazenado em ordem por linhas (row-major). O cálculo é paralelizado
 * usando MPI com o número solicitado de processos.
 *
 * @param A Ponteiro para a primeira matriz de entrada A (tamanho M*K), em ordem por linhas.
 * @param B Ponteiro para a segunda matriz de entrada B (tamanho K*N), em ordem por linhas.
 * @param M Número de linhas de A e C.
 * @param N Número de colunas de B e C.
 * @param K Número de colunas de A e linhas de B.
 * @return Ponteiro para um array recém-alocado de comprimento M*N contendo C em
 *         ordem por linhas, ou NULL em caso do rank não ser o root ou em caso de 
 *         falha de alocação ou entrada inválida. O chamador é responsável por 
 *         liberar o buffer retornado.
 */
double* dgemm_parallel_mpi(const double* A, const double* B, int M, int N, int K);

/**
 * @brief Computa o produto de matrizes C = A * B usando uma implementação em GPU (CUDA).
 *
 * Executa a multiplicação geral de matrizes em precisão dupla onde
 * A possui dimensões M x K e B possui dimensões K x N. O resultado C possui
 * dimensões M x N em ordem por linhas. Internamente utiliza tiling em
 * memória compartilhada na GPU para reduzir acessos à memória global.
 *
 * @param A Ponteiro para matriz A (M*K) em ordem por linhas.
 * @param B Ponteiro para matriz B (K*N) em ordem por linhas.
 * @param M Número de linhas de A e C.
 * @param N Número de colunas de B e C.
 * @param K Número de colunas de A e linhas de B.
 * @param tile Tamanho do tile (bloco) a ser usado na multiplicação.
 * @return Ponteiro recém-alocado contendo C (M*N) ou NULL em caso de erro.
 *         O chamador deve liberar o buffer retornado com free().
 */
double* dgemm_parallel_cuda_shared(const double* A, const double* B, int M, int N, int K, int tile);

/**
 * @brief Computa o produto de matrizes C = A * B usando uma implementação em GPU (CUDA) básica.
 * 
 * Executa a multiplicação geral de matrizes em precisão dupla onde
 * A possui dimensões M x K e B possui dimensões K x N. O resultado C possui
 * dimensões M x N em ordem por linhas. Cada thread calcula um elemento de C
 * sem otimizações adicionais.
 * 
 * @param A Ponteiro para matriz A (M*K) em ordem por linhas.
 * @param B Ponteiro para matriz B (K*N) em ordem por linhas.
 * @param M Número de linhas de A e C.
 * @param N Número de colunas de B e C.
 * @param K Número de colunas de A e linhas de B.
 * @param tile Tamanho do tile (bloco) a ser usado na multiplicação.
 * @return Ponteiro recém-alocado contendo C (M*N) ou NULL em caso de erro.
 *         O chamador deve liberar o buffer retornado com free().
 */
double* dgemm_parallel_cuda_basic(const double* A, const double* B, int M, int N, int K, int tile);

#ifdef __cplusplus
} // extern "C"
#endif

#endif // DGEMM_H