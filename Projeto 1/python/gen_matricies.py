# Esse arquivo só é necessário para fazer testes rápidos e manuais. Para fazer o benchmark completo,
# use o script benchmark.py no mesmo diretório.

import argparse
import numpy as np

def write_matrix(filename: str, mat: np.ndarray) -> None:
    rows, cols = mat.shape
    with open(filename, 'w') as f:
        f.write(f"{rows} {cols}\n")
        np.savetxt(f, mat, fmt='%.17g', delimiter=' ', newline='\n', header='', comments='')

def main() -> None:
    parser = argparse.ArgumentParser(description="Gerar duas matrizes quadradas aleatórias e salvar em arquivos de texto.")
    parser.add_argument('--k', type=int, default=1024, help='k (número de linhas e colunas das matrizes)')
    parser.add_argument('--outA', type=str, default='A.txt', help='arquivo de saída para a matriz A')
    parser.add_argument('--outB', type=str, default='B.txt', help='arquivo de saída para a matriz B')
    args = parser.parse_args()

    A = np.random.rand(args.k, args.k)
    B = np.random.rand(args.k, args.k)

    # Escreve em arquivos
    write_matrix(args.outA, A)
    write_matrix(args.outB, B)

    print(f"Matrizes geradas:")
    print(f"  A: {args.k} x {args.k}  -> {args.outA}")
    print(f"  B: {args.k} x {args.k}  -> {args.outB}")

if __name__ == "__main__":
    main()
