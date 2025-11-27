import subprocess
import argparse
import sys
import csv
import os
import shutil

def run_executable_and_time(exe_path: str, algorithm: str, m: int, n: int, k: int, threads: int = 1, ranks: int = 1, tile: int = 32, variant: str = "shared") -> float:
    # build common arguments
    cmd = [
        exe_path,
        "--alg", algorithm,
        "--m", str(m),
        "--n", str(n),
        "--k", str(k)
    ]

    if algorithm == 'mpi':
        mpiexec = shutil.which('mpirun') or shutil.which('mpiexec')
        if mpiexec is None:
            raise FileNotFoundError("mpirun/mpiexec não encontrado no PATH, mas 'mpi' foi solicitado")
        cmd = [mpiexec, "-np", str(ranks)] + cmd
    elif algorithm == 'cuda':
        cmd += ["--tile", str(tile), "--variant", variant]
    else:
        cmd += ["--threads", str(threads)]

    completed = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    
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
    parser.add_argument('--exe', type=str, required=True, help="Caminho para o executável a ser testado.")
    parser.add_argument('--alg', type=str, required=True, help="Algoritmo de multiplicação a ser usado.")
    parser.add_argument('--sizes', type=str, required=True, help="Tamanhos das matrizes (ex: '128,256,512' ou '128:512:128').")
    parser.add_argument('--threads', type=int, default=1, help="Número de threads a serem usadas.")
    parser.add_argument('--ranks', type=int, default=1, help="Número de ranks MPI a serem usados (apenas para 'mpi').")
    parser.add_argument('--tile', type=int, default=32, help="Tamanho do tile (apenas para 'cuda').")
    parser.add_argument('--variant', type=str, default='shared', choices=['shared','basic'], help="Variante CUDA: shared ou basic.")
    parser.add_argument('--reps', type=int, default=5, help='Repetições por tamanho (default 5)')
    parser.add_argument('--output', type=str, required=True, help="Arquivo CSV de saída para os resultados.")
    
    args = parser.parse_args()

    if not (os.path.isfile(args.exe) and os.access(args.exe, os.X_OK)):
        sys.exit(f"Erro: executável '{args.exe}' não encontrado ou não executável.")
    
    sizes = parse_sizes_arg(args.sizes)
    if args.alg not in ['serial', 'openmp', 'mpi', 'cuda']:
        sys.exit(f"Erro: algoritmo '{args.alg}' inválido.")

    out_file = open(args.output, 'w', newline='')
    writer = csv.writer(out_file)
    writer.writerow(["Mat Size", "Min Time (s)", "Max Time (s)", "Mean Time (s)"])
    
    print(f"Executando benchmark")
    print(f"Executável: {args.exe}")
    print(f"Algoritmo: {args.alg}")
    print(f"Tamanhos: {sizes}")
    if args.alg == "openmp":
        print(f"Threads: {args.threads}")
    elif args.alg == "mpi":
        print(f"Ranks MPI: {args.ranks}")
    elif args.alg == "cuda":
        print(f"Tile: {args.tile} | Variante: {args.variant}")
    print(f"Repetições por tamanho: {args.reps}")
    print(f"Saída: {args.output}")

    for N in sizes:
        print(f"\n=== Tamanho: {N} x {N} ===")

        samples = []
        for rep in range(args.reps):
            print(f" Rodada {rep+1}/{args.reps} ... ", end='', flush=True)
            try:
                t = run_executable_and_time(args.exe, args.alg, N, N, N, args.threads, args.ranks, args.tile, args.variant)
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

    out_file.close()

    print(f"\nBenchmark concluído. Resultados salvos em: {args.output}")

if __name__ == "__main__":
    main()