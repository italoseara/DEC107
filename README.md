# Multiplicação de Matrizes (DGEMM)

Este repositório contém uma implementação de multiplicação de matrizes (DGEMM) em C, juntamente com scripts Python para gerar matrizes de teste e um Makefile para compilar o código.

**Disciplina**: DEC107 — Processamento Paralelo\
**Professor**: Esbel Tomas Valero Orellana

## Python Scripts

Instale as dependências necessárias com:

```bash
pip install -r requirements.txt
```

O repositório inclui os seguintes scripts:

- `gen_matricies.py`: Gera duas matrizes quadradas aleatórias A (k x k) e B (k x k) e salva em arquivos de texto. Use o comando:

  ```bash
  python3 gen_matricies.py --k <colunas_A_linhas_B> --outA <arquivo_A> --outB <arquivo_B>
  ```

  Por padrão, gera matrizes de tamanho 1024x1024.
  Esse script só é necessário para fazer testes rápidos e manuais.

- `benchmark.py`: Executa benchmarks da implementação DGEMM para tamanhos de matrizes especificados e salva os resultados em um arquivo com formato CSV. Use o comando:

  ```bash
  python3 benchmark.py --exe <nome_do_arquivo> --sizes <tamanhos> --reps <n_repeticoes> --out <arquivo_resultados>
  ```

  Onde `<tamanhos>` pode ser uma lista de tamanhos separados por vírgulas (ex: `128,256,512`) ou intervalos (ex: `128:512:128` para 128, 256, 384, 512). O padrão é `128, 256, 512`. O número de repetições padrão é 5 e o arquivo de saída padrão é `benchmark.dat`.

## Compilação

Use o Makefile para compilar o código C. Execute:

```bash
make TARGET=<nome_do_arquivo>
```

Substitua `<nome_do_arquivo>` pelo nome do arquivo C que contém a implementação DGEMM (sem a extensão `.c`).

## Execução

Após compilar, execute o programa com:

```bash
./<nome_do_arquivo> <arquivo_A> <arquivo_B>
```

Substitua `<nome_do_arquivo>` pelo nome do arquivo compilado e `<arquivo_A>`, `<arquivo_B>` pelos arquivos de entrada gerados pelo script Python. A saída será a matriz resultante da multiplicação no console. Você pode redirecionar a saída para um arquivo, se desejar:

```bash
./<nome_do_arquivo> <arquivo_A> <arquivo_B> > C.txt
```

## Limpeza

Para limpar os arquivos compilados, execute:

```bash
make clean
```

## Requisitos

- [Python 3](https://www.python.org/downloads/)
- [GCC (ou outro compilador C compatível)](https://gcc.gnu.org/)
- [Make](https://www.gnu.org/software/make/)
- [OpenMP](https://www.openmp.org/)

## Autores

- [Ítalo Seara](https://github.com/italoseara)
- [Wilson Filho](https://github.com/Wssfilho)
