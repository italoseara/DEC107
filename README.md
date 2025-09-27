# Multiplicação de Matrizes (DGEMM)

Este repositório contém uma implementação de multiplicação de matrizes (DGEMM) em C, juntamente com scripts Python para gerar matrizes de teste e um Makefile para compilar o código.

**Disciplina**: DEC107 — Processamento Paralelo\
**Professor**: Esbel Tomas Valero Orellana\
**Alunos**: Ítalo Seara, Wilson Filho

## Conteúdo

- [`c/dgemm.c`](c/dgemm.c): Implementações em C da multiplicação de matrizes.
- [`c/main.c`](c/main.c): Programa principal para testar as implementações de DGEMM.
- [`c/Makefile`](c/Makefile): Arquivo Makefile para compilar o código C.
- [`python/generate_matrices.py`](python/generate_matrices.py): Script Python para gerar matrizes de teste.
- [`python/benchmark.py`](python/benchmark.py): Script Python para executar benchmarks das implementações de DGEMM.
- [`python/charts.ipynb`](python/charts.ipynb): Notebook Jupyter para visualizar os resultados dos benchmarks em gráficos.
- [`latex`](latex): Diretório contendo o código fonte do relatório do projeto em LaTeX, incluindo o PDF gerado.

## Requisitos

- [Python 3.12+](https://www.python.org/downloads/)
- [GCC 13.3.0+ (ou outro compilador C compatível)](https://gcc.gnu.org/)
- [Make](https://www.gnu.org/software/make/)
- [OpenMP](https://www.openmp.org/)

## Como Usar

> [!NOTE] Todos os script foram feitos e testados em um ambiente Linux. Pode ser necessário fazer ajustes para rodar em outros sistemas operacionais.

### Gerando Matrizes de Teste

1. Navegue até o diretório `python`:

   ```bash
   cd python
   ```

2. Instale as dependências necessárias (se ainda não estiverem instaladas):

   ```bash
   pip install -r requirements.txt
   ```

3. Execute o script para gerar matrizes de teste:

   ```bash
   python generate_matrices.py --k <valor> --outA <arquivo_A> --outB <arquivo_B>
   ```

   Sendo `k` o tamanho das matrizes (k x k).

   Exemplo:

   ```bash
   python generate_matrices.py --k 512 --outA A.txt --outB B.txt
   ```

   Isso gerará duas matrizes quadradas de tamanho 512x512 e as salvará em `A.txt` e `B.txt`.

### Compilando o Código C

1. Navegue até o diretório `c`:

   ```bash
   cd ../c
   ```

2. Compile o código usando o Makefile:

   ```bash
   make
   ```

### Executando o Programa

1. Execute o programa com as matrizes geradas:

   ```bash
   ./dgemm <arquivo_A> <arquivo_B> <implementação>
   ```

   Onde `<implementação>` pode ser:

   - `sequencial`: Implementação sequencial.
   - `sequencial_opt`: Implementação sequencial otimizada.
   - `parallel`: Implementação paralela usando OpenMP.
   - `parallel_opt`: Implementação paralela otimizada usando OpenMP.

   Exemplo:

   ```bash
   ./dgemm A.txt B.txt parallel_opt
   ```

   O resultado será o tempo de execução da multiplicação de matrizes.

### Executando Benchmarks

1. No diretório `python`, execute o script de benchmark:

   ```bash
   python benchmark.py --exe <path_to_executable> --method <implementação> --threads <n_threads> --sizes <tamanhos> --reps <repetições> --out <arquivo_saida>
   ```

   Onde `<tamanhos>` é uma lista de tamanhos de matrizes separados por vírgulas (ex: `128,256,512`) ou um intervalo (ex: `128:1024:128`, que gera tamanhos de 128 a 1024 com passo 128).

   Exemplo:

   ```bash
   python benchmark.py --exe ../c/main --method parallel_opt --threads 4 --sizes 128:1024:128 --reps 5 --out results.csv
   ```

   Isso executará o benchmark da implementação `parallel_opt` com 4 threads para tamanhos de matrizes de 128 a 1024, repetindo cada teste 5 vezes, e salvará os resultados em `results.csv`.

## Visualizando Resultados

1. Abra o notebook Jupyter `charts.ipynb` no diretório `python` utilizando o programa de sua preferencia.

2. Execute as células do notebook para carregar os resultados do benchmark e gerar gráficos de desempenho (Observação: você pode precisar ajustar o caminho do arquivo CSV no notebook).

## Autores

- [Ítalo Seara](https://github.com/italoseara)
- [Wilson Filho](https://github.com/Wssfilho)
