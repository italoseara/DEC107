import subprocess
import argparse
import sys
import csv
import os

import numpy as np


def write_matrix_txt(filename: str, mat: np.ndarray) -> None:
    rows, cols = mat.shape
    with open(filename, 'w') as f:
        f.write(f"{rows} {cols}\n")
        np.savetxt(f, mat, fmt='%.17g', delimiter=' ', newline='\n', header='', comments='')

def generate_square_matrices(N: int, out_dir: str) -> tuple[str, str]:
    A = np.random.rand(N, N)
    B = np.random.rand(N, N)
    fileA = os.path.join(out_dir, f"A_{N}.txt")
    fileB = os.path.join(out_dir, f"B_{N}.txt")
    write_matrix_txt(fileA, A)
    write_matrix_txt(fileB, B)
    return fileA, fileB

def run_executable_and_time(exe_path: str, method: str, fileA: str, fileB: str, threads: int) -> float:
    completed = subprocess.run([exe_path, method, fileA, fileB, str(threads)], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if completed.returncode != 0:
        stderr = completed.stderr.decode('utf-8', errors='replace')
        raise subprocess.CalledProcessError(completed.returncode, completed.args, output=None, stderr=stderr)
    try:
        # O programa C imprime apenas o tempo (float) no stdout
        elapsed = float(completed.stdout.decode('utf-8').strip())
    except Exception as e:
        raise RuntimeError(f"Falha ao ler tempo do programa: {e}\nstdout: {completed.stdout}")
    return elapsed

def parse_sizes_arg(sizes_arg: str) -> list[int]:
    """Converte string '128,256,512' em lista de inteiros."""

    sizes = set()
    for part in sizes_arg.split(','):
        part = part.strip()
        if not part:
            continue
        if ':' in part:
            parts = part.split(':')
            if len(parts) == 2:
                start, end = map(int, parts)
                step = 1
            elif len(parts) == 3:
                start, end, step = map(int, parts)
            else:
                raise ValueError(f"Formato inválido para intervalo: '{part}'")
            sizes.update(range(start, end + 1, step))
        else:
            sizes.add(int(part))
    return sorted(sizes)

def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark DGEMM executável para vários tamanhos de matriz.")
    parser.add_argument('--exe', required=True, help='Caminho do executável (ex: ./dgemm_seq)')
    parser.add_argument('--method', type=str, default='sequencial', choices=['sequencial', 'sequencial_opt', 'parallel', 'parallel_opt'], help='Método de multiplicação (default: sequencial)')
    parser.add_argument('--threads', type=int, default=1, help='Número de threads (apenas para métodos paralelos)')
    parser.add_argument('--sizes', type=str, default=None, help='Tamanhos separados por vírgula, ex: "128,256,512"')
    parser.add_argument('--reps', type=int, default=5, help='Repetições por tamanho (default 5)')
    parser.add_argument('--out', type=str, default='benchmark.dat', help='Arquivo CSV de saída')
    args = parser.parse_args()

    if not (os.path.isfile(args.exe) and os.access(args.exe, os.X_OK)):
        sys.exit(f"Erro: executável '{args.exe}' não encontrado ou não executável.")

    sizes = parse_sizes_arg(args.sizes) if args.sizes else [128, 256, 512]
    if not sizes:
        sys.exit("Erro: nenhuma dimensão selecionada. Use --sizes.")

    tmpdir = './tmp.' + str(os.getpid())
    os.makedirs(tmpdir, exist_ok=True)
    header = ["Mat Size", "Min Time (s)", "Max Time (s)", "Mean Time (s)"]

    # Cria arquivo e escreve cabeçalho
    out_file = open(args.out, 'w', newline='')
    writer = csv.writer(out_file)
    writer.writerow(header)

    print("Iniciando benchmark")
    print(f"Executável: {args.exe}")
    print(f"Método: {args.method}")
    if args.method not in ['sequencial', 'sequencial_opt']:
        os.environ['OMP_NUM_THREADS'] = str(args.threads)
        print(f"Número de threads: {args.threads}")
    print(f"Tamanhos: {sizes}")
    print(f"Repetições por tamanho: {args.reps}")
    print(f"Arquivo de saída: {args.out}")

    for N in sizes:
        print(f"\n=== Tamanho: {N} x {N} ===")
        fileA, fileB = generate_square_matrices(N, tmpdir)
        print(f"Arquivos: {fileA}, {fileB}")
        samples = []
        for rep in range(args.reps):
            print(f" Rodada {rep+1}/{args.reps} ... ", end='', flush=True)
            try:
                t = run_executable_and_time(args.exe, args.method, fileA, fileB, args.threads)
                samples.append(t)
                print(f"{t:.6f} s")
            except subprocess.TimeoutExpired:
                print(" TIMEOUT")
                break
            except subprocess.CalledProcessError as e:
                print(" ERRO!", file=sys.stderr)
                print(" stderr:\n" + str(e.stderr), file=sys.stderr)
                break

        min_time = min(samples) if samples else 0.0
        max_time = max(samples) if samples else 0.0
        mean_time = sum(samples) / len(samples) if samples else 0.0
        writer.writerow([N, f"{min_time:.6f}", f"{max_time:.6f}", f"{mean_time:.6f}"])
        out_file.flush()

        # Remove os arquivos temporários que já não são mais necessários
        for fname in os.listdir(tmpdir):
            os.remove(os.path.join(tmpdir, fname))

    out_file.close()
    os.rmdir(tmpdir)

    print(f"\nBenchmark concluído. Resultados salvos em: {args.out}")

if __name__ == "__main__":
    main()
