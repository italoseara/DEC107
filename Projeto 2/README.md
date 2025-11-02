# Multiplicação de Matrizes (DGEMM) — Projeto 2

Implementação da multiplicação de matrizes (DGEMM) em C com três variantes: sequencial, paralela com OpenMP e paralela distribuída com MPI. Inclui Makefile para compilação, binários de testes de corretude e ferramentas em Python para benchmark e visualização.

Disciplina: DEC107 — Processamento Paralelo \
Professor: Esbel Tomas Valero Orellana \
Alunos: Ítalo Seara, Wilson Filho

## Conteúdo

- `src/dgemm.c`: Implementações em C (sequencial, OpenMP e MPI).
- `src/main.c`: Executor/driver que aloca entradas sintéticas, chama a variante escolhida e imprime o tempo decorrido.
- `include/dgemm.h`: Interface das funções DGEMM.
- `Makefile`: Alvos para compilar, rodar e testar (com variáveis configuráveis).
- `tests/test_dgemm.c`: Testes de corretude (comparam com BLAS e entre variantes).
- `tools/benchmark.py`: Script de benchmark do executável para vários tamanhos e repetições.
- `tools/requirements.txt`: Dependências Python para análise e gráficos.
- `tools/charts.ipynb`: Notebook opcional para gerar gráficos a partir do CSV de resultados.
- `doc/main.tex`: Relatório técnico em LaTeX sobre o projeto.

Observação: Diferente do Projeto 1 (que lia matrizes de arquivos), este projeto gera matrizes de entrada sintéticas e determinísticas em memória, adequadas para medição de desempenho.

## Requisitos

- GCC 13.3.0+ (ou outro compilador C compatível)
- Make
- OpenMP (suporte via flags do compilador)
- MPI (OpenMPI ou MPICH) para a variante MPI e para executar testes MPI
- BLAS para os testes de corretude (padrão: OpenBLAS)
- Python 3.12+ (para as ferramentas em `tools/`)

Sugestão (Linux): instalar OpenBLAS e uma implementação MPI via gerenciador de pacotes da sua distro.

## Como compilar

Na raiz de `Projeto 2`:

```bash
make
```

Isso gera o executável em `build/dgemm`.

Outros comandos úteis:

- `make help`: mostra variáveis configuráveis e alvos disponíveis.
- `make clean`: remove artefatos de build.
- `make test`: compila e executa `build/test_dgemm` (ver seção Testes). Pode exigir BLAS instalado.

Variáveis de build e execução (podem ser passadas na linha do `make`):

- `ALG=serial|openmp|mpi` (padrão: `serial`)
- `THREADS=<int>` (padrão: `1`, usado por OpenMP)
- `RANKS=<int>` (padrão: `2`, usado por MPI ao rodar pelo alvo `run`)
- `M=<int> N=<int> K=<int>` (padrão: `512 512 512`)
- `BLAS_LIBS=...` (padrão: `-lopenblas`; ex.: `-lcblas -lblas`)

## Como executar

O executável imprime apenas o tempo decorrido (em segundos) no stdout.

Via Makefile (recomendado):

```bash
# OpenMP com 4 threads e matrizes 1024x1024
make run ALG=openmp THREADS=4 M=1024 N=1024 K=1024

# MPI com 4 processos e matrizes 1024x1024
make run ALG=mpi RANKS=4 M=1024 N=1024 K=1024
```

Execução direta do binário:

```bash
# Sequencial
./build/dgemm --alg serial --m 1024 --n 1024 --k 1024

# OpenMP
./build/dgemm --alg openmp --threads 4 --m 1024 --n 1024 --k 1024

# MPI (usar mpirun/mpiexec)
mpirun -np 4 ./build/dgemm --alg mpi --m 1024 --n 1024 --k 1024
```

Parâmetros suportados pelo executável (`src/main.c`):

- `--alg serial|openmp|mpi`
- `--threads <n>` (para OpenMP)
- `--m <M> --n <N> --k <K>` (dimensões)
- `--help` exibe o uso

## Testes de corretude

O alvo `test` compila e executa `build/test_dgemm`, que valida:

- `dgemm_serial` vs. BLAS (`cblas_dgemm`)
- `dgemm_parallel_openmp` vs. `dgemm_serial`
- `dgemm_parallel_mpi` vs. `dgemm_serial` (resultado verificado no rank 0)

Para rodar:

```bash
make test
```

Para rodar os testes MPI com múltiplos processos:

```bash
mpirun -np 4 ./build/test_dgemm
```

Caso sua instalação BLAS não seja o OpenBLAS padrão, você pode sobrescrever a variável `BLAS_LIBS` ao compilar o binário de teste:

```bash
make test BLAS_LIBS="-lcblas -lblas"
```

## Benchmarks (Python)

No diretório `tools/`:

```bash
pip install -r tools/requirements.txt

# Exemplo: OpenMP, tamanhos 128..1024 passo 128, 5 repetições
python tools/benchmark.py \
	--exe build/dgemm \
	--alg openmp \
	--threads 4 \
	--sizes 128:1024:128 \
	--reps 5 \
	--output results_openmp.csv

# Exemplo: MPI com 4 processos
python tools/benchmark.py \
	--exe build/dgemm \
	--alg mpi \
	--ranks 4 \
	--sizes 256,512,1024 \
	--reps 5 \
	--output results_mpi.csv
```

Notas:

- Para `--alg mpi`, o script usa `mpirun/mpiexec` automaticamente (requer que esteja no PATH).
- O executável imprime apenas o tempo; o script coleta mín./máx./média por tamanho e salva em CSV.

### Visualizando resultados

- Abra `tools/charts.ipynb` e ajuste o caminho do CSV, se necessário, para gerar gráficos.

## Relatório (LaTeX)

O relatório técnico do projeto está em `doc/main.tex`. Você pode abri-lo no seu editor LaTeX preferido. Opcionalmente, para compilar:

```bash
# Exemplo (se você utiliza latexmk)
latexmk -pdf -interaction=nonstopmode -halt-on-error -outdir=doc/build doc/main.tex
```

## Observações

- Este projeto foi desenvolvido e testado em ambiente Linux. Em outros sistemas, ajustes de ferramentas e flags podem ser necessários.
- Para medições reprodutíveis, execute em máquina ociosa, fixe frequência quando possível e repita múltiplas vezes (o script já ajuda com isso).

## Autores

- [Ítalo Seara](https://github.com/italoseara)
- [Wilson Filho](https://github.com/Wssfilho)
