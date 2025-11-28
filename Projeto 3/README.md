# Multiplicação de Matrizes (DGEMM) — Projeto 3 (CPU + GPU CUDA)

Extensão dos Projetos 1 e 2 adicionando variantes em GPU (CUDA) à multiplicação de matrizes em precisão dupla (DGEMM). Reúne agora quatro abordagens: sequencial, paralela com OpenMP, paralela distribuída com MPI e duas variantes CUDA (básica e otimizada com memória compartilhada/tiling). Inclui também testes de corretude (comparando com BLAS) e ferramentas de benchmark e visualização.

Disciplina: DEC107 — Processamento Paralelo \
Professor: Esbel Tomas Valero Orellana \
Alunos: Ítalo Seara, Wilson Filho

## Conteúdo

- `src/dgemm.c`: Implementações CPU (serial, OpenMP, MPI helper).
- `src/dgemm_cuda.cu`: Kernels CUDA (variante básica e variante com tiling em memória compartilhada).
- `src/main.c`: Driver que gera matrizes sintéticas determinísticas, seleciona a variante e mede tempo (CPU via `omp_get_wtime`, GPU via eventos CUDA).
- `include/dgemm.h`: Interface das funções (CPU + CUDA).
- `tests/test_dgemm.c`: Testes de corretude (serial vs BLAS, OpenMP/MPI vs serial, CUDA básica/tiling vs serial).
- `Makefile`: Alvos de build e execução com variáveis configuráveis (`ALG`, `THREADS`, `RANKS`, `TILE`, `VARIANT`).
- `tools/benchmark.py`: Script de benchmark para rodar múltiplos tamanhos e repetições (suporta `serial|openmp|mpi|cuda`).
- `tools/charts.ipynb`: Notebook para gerar gráficos a partir do CSV de resultados.
- `doc/main.tex`: Relatório técnico em LaTeX.

## Diferenças em relação ao Projeto 2

Este projeto adiciona suporte a execução em GPU (CUDA) com:

- Variante `basic`: cada thread computa um elemento de C sem uso de memória compartilhada.
- Variante `shared`: usa tiling e memória compartilhada para reduzir acessos à memória global (padrão no driver).

O parâmetro de runtime `--tile` (ou variável `TILE` via Makefile) controla o tamanho do bloco quadrado usado nos kernels.

## Requisitos

- GCC 13.3.0+ (ou compatível) e Make
- NVCC (CUDA Compiler) 13.0+
- OpenMP (flags do compilador)
- MPI (OpenMPI ou MPICH) para a variante MPI
- CUDA Toolkit (nvcc + driver compatível) para as variantes GPU
- BLAS (OpenBLAS recomendado) para testes (`cblas_dgemm`)
- Python 3.12+ para ferramentas em `tools/`
- GPU compatível com CUDA (Compute Capability 3.0+ recomendado)

Sugestão (Linux): instalar `openblas`, `openmpi`/`mpich` e `nvidia-cuda-toolkit` (nome pode variar conforme a distro). Testes e benchmarks foram feitos em ambiente Linux.

## Build

Na raiz de `Projeto 3`:

```bash
make
```

Gera `build/dgemm`. Principais alvos:

- `make run` — compila (se necessário) e executa com parâmetros escolhidos.
- `make test` — compila e roda `build/test_dgemm` (inclui testes CUDA; requer GPU disponível e BLAS instalado).
- `make clean` — remove artefatos de build.
- `make help` — lista variáveis e alvos.

Variáveis sobregraváveis na linha de comando do `make`:

- `ALG=serial|openmp|mpi|cuda` (default `serial`)
- `THREADS=<int>` (OpenMP)
- `RANKS=<int>` (MPI)
- `M=<int> N=<int> K=<int>` (dimensões; default 512)
- `TILE=<int>` (lado do tile para CUDA; default 32)
- `VARIANT=shared|basic` (variante CUDA; default implícito `shared` no driver; via Makefile usar `VARIANT=basic` se desejar)
- `BLAS_LIBS=...` (linkagem alternativa para testes)

Exemplos:

```bash
# OpenMP com 8 threads, matrizes 1024x1024
make run ALG=openmp THREADS=8 M=1024 N=1024 K=1024

# MPI com 4 processos
make run ALG=mpi RANKS=4 M=1024 N=1024 K=1024

# CUDA variante otimizada (shared) tile 32
make run ALG=cuda TILE=32 VARIANT=shared M=1024 N=1024 K=1024

# CUDA variante básica tile 16
make run ALG=cuda TILE=16 VARIANT=basic M=1024 N=1024 K=1024
```

## Execução direta do binário

Após `make`, o binário fica em `build/dgemm` e aceita:

```
./build/dgemm --alg serial|openmp|mpi|cuda \
			  --m M --n N --k K \
			  [--threads T] [--tile TILE] [--variant shared|basic]
```

Notas:

- `--threads` só para `openmp`.
- `mpi` deve ser invocado com `mpirun -np <ranks>`.
- `cuda` aceita `--tile` e `--variant` (padrão `shared`).

Exemplos:

```bash
# Sequencial
./build/dgemm --alg serial --m 1024 --n 1024 --k 1024

# OpenMP
./build/dgemm --alg openmp --threads 8 --m 1024 --n 1024 --k 1024

# MPI (4 processos)
mpirun -np 4 ./build/dgemm --alg mpi --m 1024 --n 1024 --k 1024

# CUDA (tiling)
./build/dgemm --alg cuda --m 1024 --n 1024 --k 1024 --tile 32 --variant shared
```

Saída: apenas o tempo decorrido (segundos) em stdout com precisão de nanossegundos (formato float). Erros vão para stderr.

## Testes de Corretude

Compilação e execução:

```bash
make test
```

Valida:

- CPU serial vs BLAS
- OpenMP vs serial
- MPI vs serial (apenas rank 0 verifica e imprime)
- CUDA básica vs serial
- CUDA shared (tiling) vs serial

Se necessário ajustar BLAS:

```bash
make test BLAS_LIBS="-lcblas -lblas"
```

Notas:

- Requer GPU funcional para testes CUDA; caso contrário, falhará em inicializar runtime.
- Para exercitar MPI diretamente no teste: `mpirun -np 4 ./build/test_dgemm`.

## Benchmarks (Python)

Instalar dependências:

```bash
pip install -r tools/requirements.txt
```

Exemplos:

```bash
# OpenMP 8 threads, tamanhos 128..1024 passo 128
python tools/benchmark.py --exe build/dgemm --alg openmp --threads 8 --sizes 128:1024:128 --reps 5 --output openmp.csv

# MPI 4 ranks
python tools/benchmark.py --exe build/dgemm --alg mpi --ranks 4 --sizes 256,512,1024 --reps 5 --output mpi.csv

# CUDA (shared) tile 32
python tools/benchmark.py --exe build/dgemm --alg cuda --tile 32 --variant shared --sizes 256:1024:256 --reps 5 --output cuda_shared.csv

# CUDA (basic) tile 16
python tools/benchmark.py --exe build/dgemm --alg cuda --tile 16 --variant basic --sizes 256:1024:256 --reps 5 --output cuda_basic.csv
```

O script coleta mínimo, máximo e média por tamanho e grava CSV. Para `mpi` usa automaticamente `mpirun` se encontrado no PATH.

### Visualização

Abra `tools/charts.ipynb`, ajuste caminho dos CSV e execute para gerar gráficos comparativos (ex.: speedup vs serial, impacto do tile, etc.).

## Relatório

`doc/main.tex` contém o relatório técnico. Para compilar (se tiver `latexmk`):

```bash
latexmk -pdf -interaction=nonstopmode -halt-on-error -outdir=doc/build doc/main.tex
```

## Observações

- Ambiente alvo: Linux. Em outros sistemas podem ser necessários ajustes de flags e path de CUDA.
- Para reprodutibilidade: isolar máquina, desativar cargas concorrentes, repetir medições (já suportado via `--reps`).
- Ajustar `TILE` pode alterar ocupação e uso de memória compartilhada; valores típicos: 16, 32.
- A variante `shared` deve oferecer melhor throughput que `basic` em tamanhos moderados/grandes.

## Autores

- [Ítalo Seara](https://github.com/italoseara)
- [Wilson Filho](https://github.com/Wssfilho)
