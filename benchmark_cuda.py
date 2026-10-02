"""同じ形状・粒子・乱数で帯電計算全体の CPU/CUDA 時間を比較する。

uv run --extra cuda python benchmark_cuda.py --n_batches 20 --repeats 3
初期 CUDA コンテキスト・コンパイルをウォームアップに分離し、図の描画を含めず計測する。
"""
import argparse
import json
import time
from dataclasses import asdict, replace
from pathlib import Path
from statistics import median

import numpy as np
import trench_charging_2d as T


def benchmark(p, repeats=3):
    if repeats < 1 or p.n_batches < 1 or p.n_per_batch < 1:
        raise ValueError("反復回数・バッチ数・粒子数は 1 以上にしてください")
    solid, _, _, _ = T.build_geometry(p)
    from trench_cuda import CudaTracer
    tracer = CudaTracer(~solid, p.cuda_device)
    print(f"GPU: {tracer.name}, float64", flush=True)
    print("初期化・コンパイルのウォームアップを行います…", flush=True)
    for backend in ("cpu", "cuda"):
        T.run(replace(p, backend=backend, n_batches=1), log=lambda _: None)
    times = {"cpu": [], "cuda": []}
    results = {}
    for k in range(repeats):
        for backend in times:
            start = time.perf_counter()
            results[backend] = T.run(replace(p, backend=backend), log=lambda _: None)
            elapsed = time.perf_counter() - start
            times[backend].append(elapsed)
            print(f"{k+1}/{repeats} {backend}: {elapsed:.4f} s", flush=True)
    cpu, gpu = results["cpu"], results["cuda"]
    equal = all(np.allclose(cpu[k], gpu[k], rtol=1e-9, atol=1e-8) for k in ("rho_raw", "phi", "Ex", "Ey"))
    report = dict(gpu=tracer.name, dtype="float64", params=asdict(p), repeats=repeats,
                  seconds=times, median_seconds={k: median(v) for k, v in times.items()},
                  speedup=median(times["cpu"])/median(times["cuda"]),
                  results_close=equal, max_abs_phi_difference=float(np.max(np.abs(cpu["phi"]-gpu["phi"]))),
                  warmup_excluded=True, plotting_excluded=True)
    print(f"中央値 CPU {report['median_seconds']['cpu']:.4f} s / CUDA {report['median_seconds']['cuda']:.4f} s "
          f"= {report['speedup']:.2f} 倍", flush=True)
    print(f"結果の照合: {equal}, 最大電位差 {report['max_abs_phi_difference']:.3g} V", flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n_batches", type=int, default=20)
    parser.add_argument("--n_per_batch", type=int, default=3000)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--cuda_device", type=int, default=0)
    parser.add_argument("--json", type=Path, help="計測条件と結果の JSON 保存先")
    args = parser.parse_args()
    # DC で回路・1 次元シースの初回計算を含めず、帯電計算の高速化を測る。
    p = replace(T.Params(), bias="dc", n_batches=args.n_batches,
                n_per_batch=args.n_per_batch, cuda_device=args.cuda_device)
    report = benchmark(p, args.repeats)
    if args.json:
        args.json.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
